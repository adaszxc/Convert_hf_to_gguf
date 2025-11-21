import os
import sys
from pathlib import Path
import shutil
import tkinter as tk
from tkinter import filedialog, ttk
import multiprocessing

from converter.core import CoreError, convert_model, detect_model_type
from converter.mmproj_from_hf import build_mmproj_from_hf

LM_STUDIO_INTEGRATION = 1
# LM_STUDIO_INTEGRATION ==================================================================================


def _detect_lm_studio_models_dir() -> Path:
    """
    Грубая эвристика: стандартный путь LM Studio.
    Для точного контроля можно будет доработать отдельно.
    """
    if os.name != "nt":
        raise RuntimeError("Интеграция с LM Studio поддерживается только на Windows")

    env_path = os.environ.get("LMSTUDIO_MODELS_DIR")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_dir():
            return p

    # Документация LM Studio: ~/.lmstudio/models
    home_dir = Path.home()
    candidate = home_dir / ".lmstudio" / "models"
    if candidate.is_dir():
        return candidate

    # Альтернативный путь, встречающийся у части установок
    alt_candidate = home_dir / ".cache" / "lm-studio" / "models"
    if alt_candidate.is_dir():
        return alt_candidate

    raise RuntimeError(
        "Папка моделей LM Studio не найдена. "
        "Создайте её (~/.lmstudio/models) или задайте LMSTUDIO_MODELS_DIR."
    )


def _normalize_lm_filename(name: str) -> str:
    # ничего не удаляем, возвращаем имя как есть
    return name



def sync_with_lm_studio(path: str, is_dir: bool) -> None:
    """
    Копирует результат в LM Studio, не перезаписывая существующие файлы/папки.
    Для файлов, чьё имя оканчивается на '_M.gguf', убирает подчёркивание:
    '..._M.gguf' -> '...M.gguf', чтобы LM Studio видела такие квантовки.
    """
    try:
        models_dir = _detect_lm_studio_models_dir().resolve()
    except Exception as e:
        print(f"Ошибка LM Studio: {e}")
        return

    # работаем с ~/.lmstudio/models/Custom
    custom_dir = models_dir / "Custom"
    custom_dir.mkdir(parents=True, exist_ok=True)

    src = Path(path).resolve()

    # если результат уже лежит внутри Custom — ничего не делаем
    try:
        inside_custom = (src == custom_dir) or src.is_relative_to(custom_dir)
    except AttributeError:
        # Python < 3.9: нет is_relative_to
        try:
            src.relative_to(custom_dir)
            inside_custom = True
        except ValueError:
            inside_custom = False

    if inside_custom:
        print(f"LM Studio: исходный путь уже внутри каталога Custom: {src}")
        return

    # ----- ветка для папки (мультимодальная модель) -----
    if is_dir:
        dest_dir = custom_dir / src.name

        if dest_dir.exists():
            print(f"Лм студио НЕ СОХРАНЕНО: файл уже существует — {dest_dir}")
            return

        shutil.copytree(src, dest_dir)
        print(f"Лм студио сохранено в: {dest_dir}")
        return

    # ----- ветка для одного файла (.gguf) -----
    folder = custom_dir / src.stem
    folder.mkdir(parents=True, exist_ok=True)

    dest_file = folder / src.name

    if dest_file.exists():
        print(f"Лм студио не сохранено: файл уже существует — {dest_file}")
        return

    shutil.copy2(src, dest_file)
    print(f"Лм студио сохранено в: {dest_file}")


def _ask_lm_copy(root: tk.Tk) -> bool | None:
    """
    Спрашивает, нужно ли копировать модель в LM Studio.
    True  - копировать
    False - не копировать
    None  - пользователь нажал Отмена
    """
    selected = {"value": True, "done": False, "ok": False}

    win = tk.Toplevel(root)
    win.title("LM Studio")
    win.resizable(False, False)

    win.update_idletasks()
    win.lift()
    win.attributes("-topmost", True)
    win.after(200, lambda: win.attributes("-topmost", False))

    frame = tk.Frame(win)
    frame.pack(padx=10, pady=10)

    var = tk.BooleanVar(value=True)
    cb = tk.Checkbutton(frame, text="Сохранить модель в LM Studio", variable=var)
    cb.pack(anchor="w")

    def on_ok() -> None:
        selected["value"] = var.get()
        selected["ok"] = True
        selected["done"] = True
        win.destroy()
        root.quit()

    def on_cancel() -> None:
        selected["done"] = True
        selected["ok"] = False
        win.destroy()
        root.quit()

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=(5, 0))

    ok_btn = tk.Button(btn_frame, text="OK", command=on_ok)
    ok_btn.pack(side=tk.LEFT, padx=5)

    cancel_btn = tk.Button(btn_frame, text="Отмена", command=on_cancel)
    cancel_btn.pack(side=tk.LEFT, padx=5)

    win.protocol("WM_DELETE_WINDOW", on_cancel)

    root.mainloop()

    if not selected["ok"]:
        return None

    return selected["value"]

#=========================================================================================================

def _build_default_basename(hf_path: Path) -> str:
    return hf_path.name


def _load_format_names() -> list[str]:
    from importlib import import_module

    formats = import_module("converter.formats")

    names: dict[str, None] = {}
    required_keys = {"name", "kind", "outtype", "quantize_type", "need_f16_first"}

    for attr_name in dir(formats):
        value = getattr(formats, attr_name)
        if isinstance(value, dict) and required_keys.issubset(value.keys()):
            name = value.get("name")
            if isinstance(name, str):
                names[name] = None

    if not names:
        raise RuntimeError("В formats.py не найдено ни одного формата")

    return sorted(names.keys())


def _ask_hf_dir(root: tk.Tk) -> Path | None:
    default = Path(r"C:\Programming\LLM")
    initial = default if default.exists() else Path("C:\\")
    path_str = filedialog.askdirectory(
        parent=root,
        title="Выберите папку HF-модели",
        initialdir=str(initial),
    )
    if not path_str:
        return None
    return Path(path_str)


def _ask_output_dir(root: tk.Tk) -> Path | None:
    default = Path(r"C:\Programming\LLM\GGUF")
    initial = default if default.exists() else Path("C:\\")
    path_str = filedialog.askdirectory(
        parent=root,
        title="Выберите папку для сохранения результата",
        initialdir=str(initial),
    )
    if not path_str:
        return None
    return Path(path_str)


def _ask_format(
    root: tk.Tk,
    formats: list[str],
    default_basename: str,
    is_multimodal: bool,
) -> tuple[str, str, str | None, bool, bool, bool] | None:
    """
    Возвращает (format_name, basename, folder_name, save_to_custom, save_to_lm, build_mmproj).
    Если пользователь нажал Отмена — None.
    folder_name и build_mmproj используются только для мультимодели.
    """
    if not formats:
        return None

    selected: dict[str, object] = {
        "format": None,
        "basename": default_basename,
        "folder_name": None,
        "save_custom": True,
        "save_lm": False,
        "build_mmproj": False,
        "ok": False,
    }

    win = tk.Toplevel(root)
    win.title("Выбор формата")
    win.resizable(False, False)

    win.update_idletasks()
    win.lift()
    win.attributes("-topmost", True)
    win.after(200, lambda: win.attributes("-topmost", False))

    label = tk.Label(win, text="Выберите формат и варианты сохранения:")
    label.pack(padx=10, pady=(10, 5))

    var = tk.StringVar(value=formats[0])
    combo = ttk.Combobox(win, textvariable=var, values=formats, state="readonly")
    combo.pack(padx=10, pady=5)
    combo.current(0)

    # поле имени папки — только для мультимодели
    if is_multimodal:
        folder_label = tk.Label(win, text="Имя папки:")
        folder_label.pack(anchor="w", padx=10, pady=(5, 0))

        folder_var = tk.StringVar(value=default_basename)
        folder_entry = tk.Entry(win, textvariable=folder_var, width=30)
        folder_entry.pack(fill="x", padx=10, pady=(0, 5))
    else:
        folder_var = None

    # поле имени файла
    name_label = tk.Label(win, text="Имя файла (без .gguf):")
    name_label.pack(anchor="w", padx=10, pady=(5, 0))

    name_var = tk.StringVar(value=default_basename)
    name_entry = tk.Entry(win, textvariable=name_var, width=30)
    name_entry.pack(fill="x", padx=10, pady=(0, 5))

    # галка "сохранить в выбранную папку"
    save_custom_var = tk.BooleanVar(value=True)
    chk_custom = tk.Checkbutton(
        win,
        text="Сохранить в выбранную папку",
        variable=save_custom_var,
    )
    chk_custom.pack(anchor="w", padx=10, pady=(5, 0))

    # галка LM Studio (показываем только если интеграция включена)
    if LM_STUDIO_INTEGRATION:
        save_lm_var = tk.BooleanVar(value=True)
        chk_lm = tk.Checkbutton(
            win,
            text="Сохранить в LM Studio",
            variable=save_lm_var,
        )
        chk_lm.pack(anchor="w", padx=10, pady=(0, 5))
    else:
        save_lm_var = tk.BooleanVar(value=False)

    # галка "создавать mmproj" — только для мультимодели
    if is_multimodal:
        build_mmproj_var = tk.BooleanVar(value=True)
        chk_mmproj = tk.Checkbutton(
            win,
            text="Создавать vision mmproj",
            variable=build_mmproj_var,
        )
        chk_mmproj.pack(anchor="w", padx=10, pady=(0, 5))
    else:
        build_mmproj_var = tk.BooleanVar(value=False)

    def on_ok() -> None:
        fmt = var.get()
        if not fmt:
            return

        # имя файла
        raw_name = name_var.get().strip()
        if raw_name:
            base = raw_name
        else:
            base = default_basename

        # имя папки для мультимодели
        if is_multimodal and folder_var is not None:
            raw_folder = folder_var.get().strip()
            if raw_folder:
                folder_name = raw_folder
            else:
                folder_name = base
        else:
            folder_name = None

        c = save_custom_var.get()
        l = save_lm_var.get()
        b = build_mmproj_var.get() if is_multimodal else False

        if not c and not l:
            return

        selected["format"] = fmt
        selected["basename"] = base
        selected["folder_name"] = folder_name
        selected["save_custom"] = c
        selected["save_lm"] = l
        selected["build_mmproj"] = b
        selected["ok"] = True
        win.destroy()
        root.quit()

    def on_cancel() -> None:
        selected["ok"] = False
        win.destroy()
        root.quit()

    btn_frame = tk.Frame(win)
    btn_frame.pack(padx=10, pady=(5, 10))

    ok_btn = tk.Button(btn_frame, text="OK", command=on_ok)
    ok_btn.pack(side=tk.LEFT, padx=5)

    cancel_btn = tk.Button(btn_frame, text="Отмена", command=on_cancel)
    cancel_btn.pack(side=tk.LEFT, padx=5)

    win.protocol("WM_DELETE_WINDOW", on_cancel)

    root.mainloop()

    if not selected["ok"]:
        return None

    return (
        selected["format"],
        selected["basename"],
        selected["folder_name"],
        bool(selected["save_custom"]),
        bool(selected["save_lm"]),
        bool(selected["build_mmproj"]),
    )


def _mmproj_worker(hf_dir_str: str, final_output_dir_str: str) -> None:
    """
    Отдельный процесс для сборки mmproj, чтобы реально грузить отдельное ядро.
    """
    hf_dir = Path(hf_dir_str)
    final_output_dir = Path(final_output_dir_str)

    try:
        mmproj_path = build_mmproj_from_hf(hf_dir, final_output_dir)
        if mmproj_path is not None:
            print(f"Собран vision mmproj (процесс): {mmproj_path}")
        else:
            print(
                "Внимание (процесс mmproj): detect_model_type вернул 'vision', "
                "но build_mmproj_from_hf не обнаружил vision-часть (None)"
            )
    except CoreError as e:
        print(f"Ошибка при сборке mmproj (процесс): {e}")
        # тут только лог, LLM всё равно может быть собрана


def main() -> None:
    root = tk.Tk()
    root.withdraw()

    hf_dir = _ask_hf_dir(root)
    if hf_dir is None:
        return

    model_type = detect_model_type(hf_dir)

    if model_type == "vision":
        is_multimodal = True
        print("Обнаружена мультимодальная модель")
    else:
        is_multimodal = False
        print("Обнаружена текстовая модель")

    format_names = _load_format_names()
    default_basename = _build_default_basename(hf_dir)

    # для мультимодели по умолчанию убираем суффикс "_multimodal" из имени
    if is_multimodal and default_basename.endswith("_multimodal"):
        default_basename = default_basename[: -len("_multimodal")]
        
    fmt_result = _ask_format(root, format_names, default_basename, is_multimodal)
    if fmt_result is None:
        return

    (
        target_format,
        basename,
        folder_name,
        save_to_custom,
        copy_to_lm,
        build_mmproj,
    ) = fmt_result

    user_output_dir: Path | None = None
    if save_to_custom:
        user_output_dir = _ask_output_dir(root)
        if user_output_dir is None:
            if not (copy_to_lm and LM_STUDIO_INTEGRATION):
                return
            user_output_dir = None

    temp_root = Path(os.getenv("TEMP", ".")) / "gguf_temp"
    temp_root.mkdir(parents=True, exist_ok=True)

    if is_multimodal:
        folder_for_temp = folder_name if folder_name else basename
        temp_output_dir = temp_root / folder_for_temp
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        final_output_dir = temp_output_dir
    else:
        final_output_dir = temp_root
        print("Запускаю конвертацию...")

    if not is_multimodal:
        result_path = convert_model(
            str(hf_dir),
            "hf",
            target_format,
            str(final_output_dir),
            basename,
        )
    else:
        if build_mmproj:
            print("Мультимодаль: LLM и vision собираются параллельно")

            mm_proc = multiprocessing.Process(
                target=_mmproj_worker,
                args=(str(hf_dir), str(final_output_dir)),
            )
            mm_proc.start()
        else:
            print("Мультимодаль: собираю только LLM (без vision mmproj)")
            mm_proc = None

        result_path = convert_model(
            str(hf_dir),
            "hf",
            target_format,
            str(final_output_dir),
            basename,
        )

        if mm_proc is not None:
            mm_proc.join()

    if is_multimodal:
        temp_result_path = final_output_dir
        is_dir_for_lm = True
    else:
        temp_result_path = Path(result_path)
        is_dir_for_lm = False

    if user_output_dir is not None:
        dest_custom = user_output_dir / temp_result_path.name

        if dest_custom.exists():
            print(f"НЕ СОХРАНЕНО: файл уже существует в - {dest_custom}")
        else:
            if is_dir_for_lm:
                shutil.copytree(temp_result_path, dest_custom)
            else:
                shutil.copy2(temp_result_path, dest_custom)
            print(f"Cохранено в: {dest_custom}")

    if copy_to_lm and LM_STUDIO_INTEGRATION:
        sync_with_lm_studio(str(temp_result_path), is_dir_for_lm)

    print("Готово.")



if __name__ == "__main__":
    main()
