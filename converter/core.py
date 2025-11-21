# Конвертирует HF или GGUF → GGUF нужного формата

# core.py
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

from . import formats


class CoreError(RuntimeError):
    """Общая ошибка конвертера."""

def _get_project_root() -> Path:
    # core.py лежит в converter/, выше — корень проекта
    return Path(__file__).resolve().parent.parent

def _get_llama_cpp_dir(kind: str = "default") -> Path:
    """
    Ищет папку llama.cpp так, чтобы работало и в обычном python,
    и в PyInstaller .exe.
    """
    candidates: list[Path] = []

    # если exe — рядом с exe
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "llama.cpp")

    # обычный режим — корень проекта
    project_root = Path(__file__).resolve().parent.parent

    if kind == "vision":
        candidates.append(project_root / "llama_cpp_vision")

    candidates.append(project_root / "llama.cpp")

    # переменная окружения
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        candidates.insert(0, Path(env_dir).expanduser())

    # поиск
    for p in candidates:
        if p.is_dir():
            return p

    checked = ", ".join(str(p) for p in candidates)
    raise CoreError(f"Папка llama.cpp не найдена. Проверены пути: {checked}")

def _get_convert_script_path() -> Path:
    llama_dir = _get_llama_cpp_dir()
    script = llama_dir / "convert_hf_to_gguf.py"
    if not script.is_file():
        raise CoreError(f"convert_hf_to_gguf.py не найден в {llama_dir}")
    return script


def _get_quantize_path() -> Path:
    llama_dir = _get_llama_cpp_dir()
    binary_name = "quantize.exe" if os.name == "nt" else "quantize"
    binary = llama_dir / binary_name
    if not binary.is_file():
        raise CoreError(f"binary quantize не найден в {llama_dir} (ожидался {binary_name})")
    return binary


def _load_formats_registry() -> Dict[str, Dict[str, Any]]:
    """
    Строит реестр форматов по содержимому formats.py.
    Любой новый формат добавляется только там, core не меняем.
    """
    registry: Dict[str, Dict[str, Any]] = {}

    required_keys = {"name", "kind", "outtype", "quantize_type", "need_f16_first"}

    for attr_name in dir(formats):
        value = getattr(formats, attr_name)
        if isinstance(value, dict) and required_keys.issubset(value.keys()):
            name = value["name"]
            if not isinstance(name, str):
                continue
            registry[name] = value

    if not registry:
        raise CoreError("В formats.py не найдено ни одного валидного формата")

    return registry


_FORMATS_REGISTRY = _load_formats_registry()


def _get_format_by_name(name: str) -> Dict[str, Any]:
    try:
        return _FORMATS_REGISTRY[name]
    except KeyError:
        raise CoreError(f"Формат '{name}' не найден в formats.py") from None


def _detect_model_type(hf_dir: Path) -> str:
    """
    Грубая автоопределялка text/vision по config.json.
    Если что-то не так — считаем текстовой.
    """
    config_path = hf_dir / "config.json"
    if not config_path.is_file():
        # Нет конфига — считаем обычной текстовой
        return "text"

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        # Конфиг битый/нечитаемый — считаем текстовой, но не замалчиваем проблему
        raise CoreError(f"Не удалось прочитать config.json в {hf_dir}: {e}")

    text = json.dumps(cfg).lower()

    # Очень простой хак: если в конфиге явно мелькают vision/image — считаем vision
    vision_markers = [
        "vision_tower",
        "vision_config",
        "mm_vision",
        "image_token_index",
        "image_size",
        '"vision"',
        '"image"',
    ]

    for marker in vision_markers:
        if marker in text:
            return "vision"

    return "text"


def _ensure_gguf_is_f16(path: Path) -> None:
    """
    Проверяет, что входной GGUF — F16.
    Если библиотека gguf/reader недоступна или тип не F16 → исключение.
    """
    try:
        from gguf.gguf_reader import GGUFReader, GGMLQuantizationType  # type: ignore
    except Exception as e:
        raise CoreError(
            f"Не удалось импортировать GGUFReader для проверки типа весов: {e}"
        )

    try:
        reader = GGUFReader(str(path))
    except Exception as e:
        raise CoreError(f"Не удалось прочитать GGUF-файл '{path}': {e}")

    if not reader.tensors:
        raise CoreError(f"GGUF-файл '{path}' не содержит тензоров")

    # Берём первый тензор как индикатор типа квантовки
    first_tensor = reader.tensors[0]
    qtype = first_tensor.tensor_type

    # Ожидаем F16
    try:
        is_f16 = qtype.name == "F16" or qtype == GGMLQuantizationType.F16
    except Exception:
        # На случай, если enum устроен иначе
        is_f16 = False

    if not is_f16:
        raise CoreError(f"GGUF '{path}' не имеет тип F16, квантовать в Q4 нельзя")


def _run_subprocess(cmd, cwd: Path) -> None:
    """
    Запускает внешнюю команду синхронно.
    Любой ненулевой код возврата → CoreError с выводом stderr.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        raise CoreError(f"Не удалось запустить команду {cmd}: {e}")

    if result.returncode != 0:
        raise CoreError(
            f"Команда {cmd} завершилась с кодом {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _build_basename(
    input_path: Path,
    input_type: str,
    output_basename: str | None,
) -> str:
    if output_basename:
        return output_basename

    if input_type == "hf":
        return input_path.name
    elif input_type == "gguf":
        return input_path.stem
    else:
        raise CoreError(f"Неподдерживаемый input_type: {input_type}")


def convert_model(
    input_path: str,
    input_type: str,
    target_format_name: str,
    output_dir: str,
    output_basename: str | None = None,
) -> str:
    """
    Центральная функция ядра:
    HF/GGUF → GGUF нужного формата.

    Возвращает путь к итоговому GGUF.
    При любой проблеме бросает CoreError.
    """
    in_path = Path(input_path)
    out_dir = Path(output_dir)

    if not in_path.exists():
        raise CoreError(f"Входной путь не найден: {in_path}")

    if input_type not in ("hf", "gguf"):
        raise CoreError(f"input_type должен быть 'hf' или 'gguf', получено: {input_type}")

    # Готовим выходную папку
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise CoreError(f"Не удалось создать выходную папку {out_dir}: {e}")

    # Берём описание формата
    fmt = _get_format_by_name(target_format_name)
    kind = fmt["kind"]              # "base" или "q4"
    outtype = fmt["outtype"]        # "f16", "q8_0", ...
    quantize_type = fmt["quantize_type"]
    need_f16_first = fmt["need_f16_first"]

       # Определяем базовое имя и итоговый путь
    basename = _build_basename(in_path, input_type, output_basename)
    final_filename = f"{basename}_{target_format_name}.gguf"
    final_path = out_dir / final_filename

    llama_dir = _get_llama_cpp_dir()
    convert_script = _get_convert_script_path()
    quantize_bin = _get_quantize_path()

    # выбираем python из venv (один раз на всю HF-ветку)
    if getattr(sys, "frozen", False):
        project_root = Path(sys.executable).resolve().parent
    else:
        project_root = _get_project_root()

    if os.name == "nt":
        python_exe = project_root / "venv" / "Scripts" / "python.exe"
    else:
        python_exe = project_root / "venv" / "bin" / "python"

    if not python_exe.is_file():
        raise CoreError(f"Не найден интерпретатор venv: {python_exe}")

    # Ветка: HF-вход
    if input_type == "hf":
        if not in_path.is_dir():
            raise CoreError(f"Ожидалась HF-папка, но это не папка: {in_path}")

        model_type = _detect_model_type(in_path)

        # Сценарий HF → base
        if kind == "base":
            cmd = [
                str(python_exe),
                str(convert_script),
                str(in_path),
                "--outfile",
                str(final_path),
                "--outtype",
                str(outtype),
            ]
            _run_subprocess(cmd, cwd=llama_dir)
            return str(final_path)

        # Сценарий HF → Q4 (через F16)
        if kind == "q4":
            if not need_f16_first or outtype.lower() != "f16":
                raise CoreError(
                    f"Формат '{target_format_name}' помечен как q4, но не требует F16 первым шагом. "
                    f"Ядро такому сценарию не доверяет."
                )

            temp_f16_path = out_dir / f"{basename}_F16_TEMP.gguf"

            # Шаг 1: HF → F16
            cmd_convert = [
                str(python_exe),
                str(convert_script),
                str(in_path),
                "--outfile",
                str(temp_f16_path),
                "--outtype",
                "f16",
            ]
            _run_subprocess(cmd_convert, cwd=llama_dir)

            # Шаг 2: F16 → Q4
            if not isinstance(quantize_type, str) or not quantize_type:
                raise CoreError(f"Для формата '{target_format_name}' не указан quantize_type")

            cmd_quant = [
                str(quantize_bin),
                str(temp_f16_path),
                str(final_path),
                quantize_type,
            ]
            try:
                _run_subprocess(cmd_quant, cwd=llama_dir)
            finally:
                if temp_f16_path.exists():
                    try:
                        temp_f16_path.unlink()
                    except OSError:
                        pass

            return str(final_path)

        # Другие kind для HF пока не поддерживаем
        raise CoreError(
            f"Неподдерживаемый kind '{kind}' для input_type='hf' "
            f"и формата '{target_format_name}'"
        )


    # Ветка: GGUF-вход
    if input_type == "gguf":
        if not in_path.is_file():
            raise CoreError(f"Ожидался GGUF-файл, но это не файл: {in_path}")

        if in_path.suffix.lower() != ".gguf":
            raise CoreError(f"Файл '{in_path}' не имеет расширение .gguf")

        # Сейчас поддерживаем только GGUF → Q4
        if kind != "q4":
            raise CoreError(
                f"Для input_type='gguf' поддерживаются только q4-форматы, "
                f"а '{target_format_name}' имеет kind='{kind}'"
            )

        # Строгая проверка: входной GGUF должен быть F16
        _ensure_gguf_is_f16(in_path)

        if not isinstance(quantize_type, str) or not quantize_type:
            raise CoreError(f"Для формата '{target_format_name}' не указан quantize_type")

        cmd_quant = [
            str(quantize_bin),
            str(in_path),
            str(final_path),
            quantize_type,
        ]
        _run_subprocess(cmd_quant, cwd=llama_dir)
        return str(final_path)

    # До сюда доходить не должны, но на всякий случай
    raise CoreError(
        f"Комбинация input_type='{input_type}' и формата '{target_format_name}' не поддерживается"
    )

def detect_model_type(hf_path) -> str:
    """
    Публичный детектор типа модели для main.py.
    Принимает строку или Path, возвращает "text" или "vision".
    """
    return _detect_model_type(Path(hf_path))