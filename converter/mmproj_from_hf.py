# mmproj_from_hf.py

import json
import shutil
import uuid
import sys
from pathlib import Path
from typing import Optional, Union

from .core import CoreError, _get_project_root, _get_llama_cpp_dir, _run_subprocess


def build_mmproj_from_hf(
    hf_dir: Union[str, Path],
    dest_dir: Union[str, Path],
) -> Optional[Path]:
    """
    Универсальный конвертер vision-части HF-модели в mmproj GGUF.

    :param hf_dir: путь к HF-папке модели
    :param dest_dir: папка, куда нужно вынести готовый mmproj*.gguf
                     (main передаёт сюда свой временный каталог для мультимодели)
    :return: Path к файлу mmproj*.gguf в dest_dir или None (если vision нет)
    :raises CoreError: при любой ошибке логики/поддержки/запуска внешних скриптов
    """
    hf_path = Path(hf_dir).resolve()

    if not hf_path.is_dir():
        raise CoreError(f"HF-папка не найдена: {hf_path}")

    config = _load_config(hf_path)
    if not _has_vision(config):
        # Мультимодальности нет — честно возвращаем None
        return None

    family = _detect_family(config)
    if family is None:
        raise CoreError(
            f"Обнаружены признаки мультимодели, но архитектуру определить не удалось "
            f"(config.json в {hf_path})"
        )

    llama_dir = _get_llama_cpp_dir("vision")
    script_path = _find_vision_converter_script(llama_dir, family)

    project_root = _get_project_root()
    temp_root = project_root / "mmproj_temp"
    temp_root.mkdir(parents=True, exist_ok=True)

    temp_dir = temp_root / f"mmproj_{uuid.uuid4().hex}"
    tmp_hf_dir = temp_dir  # копия HF живёт прямо в temp_dir

    # куда main хочет положить итоговый mmproj
    dest_dir_path = Path(dest_dir).resolve()
    dest_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        # Копируем всю HF-папку целиком
        shutil.copytree(hf_path, tmp_hf_dir)

        # Запускаем конвертер в копии HF
        cmd = [str(_python_executable()), str(script_path), "."]
        _run_subprocess(cmd, cwd=tmp_hf_dir)

        # Ищем результат mmproj*.gguf во временной папке
        mmproj_temp_path = _find_mmproj_result(tmp_hf_dir)

        # Имя mmproj в стиле LM Studio / llama.cpp
        # LLM-файл может называться как угодно (Q4_K_M, Q6_K, ...),
        # projector всегда кладём как mmproj-model-f16.gguf
        final_name = "mmproj-model-f16.gguf"
        final_path = dest_dir_path / final_name

        if final_path.exists():
            raise CoreError(
                f"Файл mmproj уже существует в целевой папке: {final_path}"
            )

        shutil.copy2(mmproj_temp_path, final_path)
        return final_path

    finally:
        # Полная очистка временной директории
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Удаляем корень mmproj_temp, если он пустой
        if temp_root.exists():
            try:
                next(temp_root.iterdir())
            except StopIteration:
                try:
                    temp_root.rmdir()
                except OSError:
                    pass




def _load_config(hf_path: Path) -> dict:
    config_path = hf_path / "config.json"
    if not config_path.is_file():
        raise CoreError(f"В HF-папке нет config.json: {hf_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        raise CoreError(f"Не удалось прочитать config.json в {hf_path}: {e}")

    if not isinstance(cfg, dict):
        raise CoreError(f"config.json в {hf_path} имеет некорректный формат (ожидается объект JSON)")

    return cfg


def _has_vision(cfg: dict) -> bool:
    # Ключи, которые часто присутствуют у мультимоделей
    vision_keys = {
        "vision_tower",
        "vision_config",
        "vision_encoder",
        "mm_projector",
        "mm_projection",
        "image_size",
        "vision_hidden_size",
    }

    if any(k in cfg for k in vision_keys):
        return True

    # Признаки в типах/архитектурах
    names: list[str] = []

    arch = cfg.get("architectures")
    if isinstance(arch, str):
        names.append(arch)
    elif isinstance(arch, list):
        names.extend(str(a) for a in arch if isinstance(a, str))

    for key in ("model_type", "model_name"):
        v = cfg.get(key)
        if isinstance(v, str):
            names.append(v)

    joined = " ".join(names).lower()
    if any(s in joined for s in ("vision", "vl", "llava", "mmproj", "multimodal")):
        return True

    return False


def _detect_family(cfg: dict) -> Optional[str]:
    """
    Грубое определение семейства мультимодели по config.json.

    Возвращает строку семейства или None, если определить нельзя.
    """
    names: list[str] = []

    arch = cfg.get("architectures")
    if isinstance(arch, str):
        names.append(arch)
    elif isinstance(arch, list):
        names.extend(str(a) for a in arch if isinstance(a, str))

    for key in ("model_type", "model_name"):
        v = cfg.get(key)
        if isinstance(v, str):
            names.append(v)

    joined = " ".join(names).lower()

    if "gemma" in joined:
        return "gemma"

    if "llava" in joined:
        return "llava"

    if "qwen" in joined and "vl" in joined:
        return "qwen_vl"

    if "phi" in joined and "vision" in joined:
        return "phi_vision"

    if "mistral" in joined and ("vision" in joined or "vl" in joined):
        return "mistral_vision"

    return None


def _find_vision_converter_script(llama_dir: Path, family: str) -> Path:
    """
    Ищет подходящий конвертер vision→mmproj внутри llama.cpp.

    По ТЗ:
      - поиск по реальной структуре llama.cpp
      - без жёстких путей
    """
    family = family.lower()

    def predicate(fname: str) -> bool:
        name = fname.lower()

        if family == "gemma":
            return "gemma" in name and "encoder" in name and "gguf" in name

        if family == "llava":
            return "llava" in name and ("mmproj" in name or "encoder" in name) and "gguf" in name

        if family == "qwen_vl":
            return "qwen" in name and ("vl" in name or "vision" in name) and "gguf" in name

        if family == "phi_vision":
            return "phi" in name and "vision" in name and "gguf" in name

        if family == "mistral_vision":
            return "mistral" in name and ("vision" in name or "vl" in name) and "gguf" in name

        return False

    candidates: list[Path] = []
    for path in llama_dir.rglob("*.py"):
        if predicate(path.name):
            candidates.append(path)

    if not candidates:
        raise CoreError(
            f"Для семейства мультимодели '{family}' не найден конвертер vision→mmproj в {llama_dir}"
        )

    # Если кандидатов несколько — берём самый специфичный (самое длинное имя файла)
    candidates.sort(key=lambda p: len(p.name), reverse=True)
    return candidates[0]


def _find_mmproj_result(tmp_hf_dir: Path) -> Path:
    mmproj_files = [p for p in tmp_hf_dir.rglob("mmproj*.gguf") if p.is_file()]

    if not mmproj_files:
        raise CoreError(
            f"Конвертер vision→mmproj отработал без явной ошибки, но mmproj*.gguf не найден "
            f"во временной папке {tmp_hf_dir}"
        )

    if len(mmproj_files) > 1:
        names = ", ".join(str(p) for p in mmproj_files)
        raise CoreError(
            f"Найдено несколько кандидатов mmproj*.gguf во временной папке {tmp_hf_dir}: {names}. "
            f"Невозможно однозначно выбрать один файл."
        )

    return mmproj_files[0]


def _python_executable() -> Path:
    """
    Возвращает исполняемый Python, которым запущен конвертер.
    Вынесено отдельно на случай, если потом понадобится переопределять интерпретатор.
    """
    return Path(sys.executable)
