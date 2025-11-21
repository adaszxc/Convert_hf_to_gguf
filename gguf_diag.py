import os
import tkinter as tk
from tkinter import filedialog, messagebox
from gguf import GGUFReader

# человекочитаемые имена только для старых/частых типов
QT_NAMES = {
    0:  "F32 (без квантования)",
    1:  "F16 (без квантования)",

    2:  "Q4_0 (4-бит, legacy)",
    3:  "Q4_1 (4-бит, legacy)",

    6:  "Q5_0 (5-бит, legacy)",
    7:  "Q5_1 (5-бит, legacy)",

    8:  "Q8_0 (8-бит, legacy)",
    9:  "Q8_1 (8-бит, legacy)",

    10: "Q2_K (2-бит, K-quant)",
    11: "Q3_K (3-бит, K-quant)",
    12: "Q4_K (4-бит, K-quant)",
    13: "Q5_K (5-бит, K-quant)",
    14: "Q6_K (6-бит, K-quant)",
    15: "Q8_K (8-бит, K-quant)",
}

# порог доминирования квантовки в ядре (по байтам/элементам)
MAIN_QT_THRESHOLD_BYTES = 0.85  # 85% по объёму байт (для "почти чистой")
MAIN_QT_THRESHOLD_ELEMS = 0.75  # 75% по числу элементов (для "почти чистой")

# доля общего объёма весов, которая считается "ядром"
CORE_FRACTION = 0.85  # 85% по объёму байт

# учитывать только "весовые" тензоры (обычно .weight / .w*) во всей статистике
USE_WEIGHT_ONLY = False

# текущий открытый GGUF
current_reader = None
current_path = None
current_tensor_txt_path = None  # путь к txt с тензорами


def fmt_bytes(n: int) -> str:
    for unit in ["Б", "КБ", "МБ", "ГБ", "ТБ"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} ПБ"


def qt_name_from_type(t) -> str:
    """
    Получить человекочитаемое имя квантовки:
    - сначала пытаемся взять из QT_NAMES по коду
    - потом из Enum.name (Q4_0, Q4_K, IQ2_XS, BF16, ...)
    - в крайнем случае просто "код N"
    """
    tt = getattr(t, "tensor_type", None)
    if tt is None:
        return "неизвестный тип"

    code = int(tt)
    if code in QT_NAMES:
        return QT_NAMES[code]

    name_enum = getattr(tt, "name", None)
    if name_enum:
        return name_enum

    return f"код {code}"


def is_weight_tensor(name: str) -> bool:
    """
    Эвристика для выбора "основных весов".
    Усилена: добавлены частые паттерны и фильтр того, что весами считать НЕ надо.
    """
    if not name:
        return False
    name = name.lower()

    # то, что мы НЕ хотим считать основными весами
    bad_patterns = [
        "norm.weight", "layernorm", "ln_f", "rms_norm",
        ".bias", "bias_", "embedding", ".emb", "position", "pos_emb",
    ]
    if any(pat in name for pat in bad_patterns):
        return False

    # то, что похоже на основные матрицы/проекции
    good_patterns = [
        ".weight",
        ".kernel", ".proj", ".projection",
        ".wq", ".wk", ".wv", ".wo",
        ".w1", ".w2", ".w3",
        ".up", ".down", ".gate",
        ".ffn",
        "attention.w",
    ]
    return any(pat in name for pat in good_patterns)


def compute_core_stats(tensor_records, core_fraction: float, total_bytes: int):
    """
    Выделить ядро модели:
    берём самые крупные по байтам тензоры,
    пока их суммарный объём не достигнет core_fraction * total_bytes.
    """
    if total_bytes <= 0:
        return {}, 0, 0, 0

    tensor_records_sorted = sorted(
        tensor_records,
        key=lambda tr: tr["bytes"],
        reverse=True
    )

    core_limit_bytes = total_bytes * core_fraction

    stats_core = {}
    total_core_tensors = 0
    total_core_elems = 0
    total_core_bytes = 0

    for tr in tensor_records_sorted:
        if total_core_bytes >= core_limit_bytes and total_core_bytes > 0:
            break

        if tr["bytes"] <= 0:
            continue

        code = tr["code"]
        name_qt = tr["name"]

        info_q = stats_core.setdefault(code, {
            "name": name_qt,
            "count": 0,
            "elements": 0,
            "bytes": 0,
        })

        info_q["count"] += 1
        info_q["elements"] += tr["elements"]
        info_q["bytes"] += tr["bytes"]

        total_core_tensors += 1
        total_core_elems += tr["elements"]
        total_core_bytes += tr["bytes"]

    return stats_core, total_core_tensors, total_core_elems, total_core_bytes


def find_main_quant(stats_core, total_core_bytes, total_core_elems):
    """
    Найти наиболее крупные типы квантовки в ядре:
    возвращаем доли по байтам и по числу элементов без жёсткого решения.
    """
    if total_core_bytes <= 0 or total_core_elems <= 0 or not stats_core:
        return None

    best_code_bytes = None
    best_share_bytes = 0.0

    best_code_elems = None
    best_share_elems = 0.0

    for code, info in stats_core.items():
        bts = info["bytes"]
        elems = info["elements"]

        share_b = bts / total_core_bytes if total_core_bytes else 0.0
        share_e = elems / total_core_elems if total_core_elems else 0.0

        if share_b > best_share_bytes:
            best_share_bytes = share_b
            best_code_bytes = code

        if share_e > best_share_elems:
            best_share_elems = share_e
            best_code_elems = code

    return {
        "by_bytes": (best_code_bytes, best_share_bytes),
        "by_elems": (best_code_elems, best_share_elems),
    }


def classify_share(share: float, hard_threshold: float):
    """
    Мягкая классификация доли квантовки:
    - share >= hard_threshold -> "почти чистая"
    - 0.6 <= share < hard_threshold -> "основной тип, но модель заметно смешанная"
    - share < 0.6 -> "сильно смешанная"
    """
    if share >= hard_threshold:
        return "почти чистая", True
    if share >= 0.6:
        return "основной тип, но заметно смешанная", False
    return "сильно смешанная", False


def choose_file():
    global current_reader, current_path

    path = filedialog.askopenfilename(
        title="Выбери GGUF файл",
        filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
    )
    if not path:
        return

    try:
        r = GGUFReader(path)
        print(sorted({(int(t.tensor_type), t.tensor_type.name) for t in r.tensors}))
        current_reader = r
        current_path = path


        # статистика по всем тензорам
        stats_all = {}
        tensor_records_all = []
        total_tensors_all = 0
        total_elems_all = 0
        total_bytes_all = 0

        # статистика только по весам
        stats_weights = {}
        tensor_records_weights = []
        total_tensors_weights = 0
        total_elems_weights = 0
        total_bytes_weights = 0

        for t in r.tensors:
            name = getattr(t, "name", "") or ""
            tt = getattr(t, "tensor_type", None)
            if tt is None:
                code = -1
            else:
                code = int(tt)

            name_qt = qt_name_from_type(t)

            n_elems = int(getattr(t, "n_elements", 0))
            n_bytes = int(getattr(t, "n_bytes", 0))

            is_w = is_weight_tensor(name)

            # все тензоры (если не включён режим "только весовые")
            if not USE_WEIGHT_ONLY:
                info_all = stats_all.setdefault(code, {
                    "name": name_qt,
                    "count": 0,
                    "elements": 0,
                    "bytes": 0,
                })

                info_all["count"] += 1
                info_all["elements"] += n_elems
                info_all["bytes"] += n_bytes

                total_tensors_all += 1
                total_elems_all += n_elems
                total_bytes_all += n_bytes

                tensor_records_all.append({
                    "code": code,
                    "name": name_qt,
                    "elements": n_elems,
                    "bytes": n_bytes,
                })

            # только весовые тензоры
            if is_w:
                info_w = stats_weights.setdefault(code, {
                    "name": name_qt,
                    "count": 0,
                    "elements": 0,
                    "bytes": 0,
                })

                info_w["count"] += 1
                info_w["elements"] += n_elems
                info_w["bytes"] += n_bytes

                total_tensors_weights += 1
                total_elems_weights += n_elems
                total_bytes_weights += n_bytes

                tensor_records_weights.append({
                    "code": code,
                    "name": name_qt,
                    "elements": n_elems,
                    "bytes": n_bytes,
                })

        # если вообще ничего не набралось — выходим
        if USE_WEIGHT_ONLY:
            if not stats_weights:
                messagebox.showinfo(
                    "Квантовка",
                    "Весовые тензоры не найдены (либо все отфильтрованы эвристикой is_weight_tensor)."
                )
                return
        else:
            if not stats_all and not stats_weights:
                messagebox.showinfo(
                    "Квантовка",
                    "Тензоры не найдены."
                )
                return

        lines = []
        lines.append(f"Файл: {path}")
        lines.append("")

        if USE_WEIGHT_ONLY:
            lines.append("Режим: анализ только весовых тензоров (USE_WEIGHT_ONLY = True).")
            lines.append("")
        else:
            lines.append("Режим: анализ всех тензоров + отдельный анализ весов.")
            lines.append("")

        # --- общая картина по всем тензорам ---
        if not USE_WEIGHT_ONLY and stats_all:
            lines.append("=== Все тензоры (без фильтра по весам) ===")
            lines.append(f"Всего тензоров: {total_tensors_all}")
            lines.append(f"Всего элементов: {total_elems_all}")
            lines.append(f"Всего объём: {fmt_bytes(total_bytes_all)}")
            lines.append("")
            lines.append("По типам квантовки (все тензоры):")
            for code, info in sorted(stats_all.items(), key=lambda kv: kv[0]):
                cnt = info["count"]
                elems = info["elements"]
                bts = info["bytes"]

                p_e = (elems / total_elems_all * 100) if total_elems_all else 0.0
                p_b = (bts / total_bytes_all * 100) if total_bytes_all else 0.0

                lines.append(
                    f"{info['name']} [код {code}]: "
                    f"тензоров {cnt}, "
                    f"элементов {elems} ({p_e:.2f}%), "
                    f"объём {fmt_bytes(bts)} ({p_b:.2f}%)"
                )
            lines.append("")

        # --- анализ только по весам ---
        if stats_weights:
            lines.append("=== Весовые тензоры (по эвристике is_weight_tensor) ===")
            lines.append(f"Всего весовых тензоров: {total_tensors_weights}")
            lines.append(f"Всего элементов в весах: {total_elems_weights}")
            lines.append(f"Всего объём весов: {fmt_bytes(total_bytes_weights)}")
            lines.append("")
            lines.append("По типам квантовки (только весовые тензоры):")

            for code, info in sorted(stats_weights.items(), key=lambda kv: kv[0]):
                cnt = info["count"]
                elems = info["elements"]
                bts = info["bytes"]

                p_e = (elems / total_elems_weights * 100) if total_elems_weights else 0.0
                p_b = (bts / total_bytes_weights * 100) if total_bytes_weights else 0.0

                lines.append(
                    f"{info['name']} [код {code}]: "
                    f"тензоров {cnt}, "
                    f"элементов {elems} ({p_e:.2f}%), "
                    f"объём {fmt_bytes(bts)} ({p_b:.2f}%)"
                )
            lines.append("")
        else:
            lines.append("Весовые тензоры по эвристике не найдены — ядро модели по весам не выделяется.")
            show_big_message("\n".join(lines))
            return

        # --- ядро модели по весам (самые крупные весовые тензоры по объёму) ---
        lines.append("=== Ядро модели по весам (самые крупные весовые тензоры) ===")

        stats_core, total_core_tensors, total_core_elems, total_core_bytes = compute_core_stats(
            tensor_records_weights, CORE_FRACTION, total_bytes_weights
        )

        if total_core_bytes == 0:
            lines.append(
                "Ядро модели не выделено: суммарный объём весов равен нулю "
                "или все весовые тензоры нулевого размера."
            )
            main_info = None
        else:
            pct_core = (total_core_bytes / total_bytes_weights * 100.0) if total_bytes_weights else 0.0
            lines.append(
                f"В ядро включены весовые тензоры, которые суммарно дают "
                f"{pct_core:.2f}% объёма всех весов "
                f"(параметр CORE_FRACTION = {CORE_FRACTION * 100:.0f}%)."
            )
            lines.append(
                f"Всего тензоров в ядре: {total_core_tensors}, "
                f"элементов в ядре: {total_core_elems}, "
                f"объём ядра: {fmt_bytes(total_core_bytes)}"
            )

            for code, info in sorted(stats_core.items(), key=lambda kv: kv[0]):
                cnt = info["count"]
                elems = info["elements"]
                bts = info["bytes"]

                p_e = (elems / total_core_elems * 100) if total_core_elems else 0.0
                p_b = (bts / total_core_bytes * 100) if total_core_bytes else 0.0

                lines.append(
                    f"{info['name']} [код {code}]: "
                    f"тензоров {cnt}, "
                    f"элементов {elems} ({p_e:.2f}%), "
                    f"объём {fmt_bytes(bts)} ({p_b:.2f}%)"
                )

            main_info = find_main_quant(
                stats_core,
                total_core_bytes,
                total_core_elems,
            )

        # --- итоговый вывод по основной квантовке ---
        lines.append("")
        lines.append("=== Вывод по основной квантовке модели (по ядру весов) ===")

        if main_info is None:
            lines.append(
                "Основная квантовка модели не определена: ядро по весам не выделено."
            )
        else:
            code_b, share_b = main_info["by_bytes"]
            code_e, share_e = main_info["by_elems"]

            if code_b is not None:
                name_b = QT_NAMES.get(code_b, f"тип {code_b}")
                pct_b = share_b * 100.0
                text_b, hard_b = classify_share(share_b, MAIN_QT_THRESHOLD_BYTES)
                lines.append(
                    f"По объёму байт в ядре: {name_b} [код {code_b}], "
                    f"{pct_b:.2f}% байт, "
                    f"характеристика: {text_b} "
                    f"(порог 'почти чистой' {MAIN_QT_THRESHOLD_BYTES * 100:.0f}%)."
                )

            if code_e is not None:
                name_e = QT_NAMES.get(code_e, f"тип {code_e}")
                pct_e = share_e * 100.0
                text_e, hard_e = classify_share(share_e, MAIN_QT_THRESHOLD_ELEMS)
                lines.append(
                    f"По числу элементов в ядре: {name_e} [код {code_e}], "
                    f"{pct_e:.2f}% элементов, "
                    f"характеристика: {text_e} "
                    f"(порог 'почти чистой' {MAIN_QT_THRESHOLD_ELEMS * 100:.0f}%)."
                )

            if code_b is not None and code_e is not None and code_b != code_e:
                lines.append(
                    "Замечание: по байтам и по числу элементов доминируют разные типы "
                    "(разная плотность хранения / структурное смешение квантовок)."
                )

        out = "\n".join(lines)
        show_big_message(out)

    except Exception as e:
        messagebox.showerror("Ошибка", str(e))


def show_tensors():
    global current_tensor_txt_path

    if current_reader is None:
        messagebox.showinfo("Тензоры", "Сначала выбери GGUF-файл.")
        return

    r = current_reader

    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_name = os.path.splitext(os.path.basename(current_path))[0]
    txt_name = base_name + ".txt"
    txt_path = os.path.join(script_dir, txt_name)
    current_tensor_txt_path = txt_path

    lines = []
    lines.append(f"Файл: {current_path}")
    lines.append("")
    if USE_WEIGHT_ONLY:
        lines.append("Режим: выгружаются только весовые тензоры (USE_WEIGHT_ONLY = True).")
        lines.append("")
    else:
        lines.append("Режим: выгружаются все тензоры (USE_WEIGHT_ONLY = False).")
        lines.append("")
    lines.append("Имя тензора | элементов | форма | тип квантовки | объём")
    lines.append("")

    for t in r.tensors:
        name = getattr(t, "name", "") or ""

        if USE_WEIGHT_ONLY and not is_weight_tensor(name):
            continue

        n_elems = int(getattr(t, "n_elements", 0))
        n_bytes = int(getattr(t, "n_bytes", 0))
        tt = getattr(t, "tensor_type", None)
        code = int(tt) if tt is not None else -1
        qt_name = qt_name_from_type(t)

        shape = getattr(t, "shape", None)
        if shape is None:
            shape = getattr(t, "ne", None)

        if shape is None:
            shape_str = "?"
        else:
            try:
                shape_str = "x".join(str(x) for x in shape)
            except TypeError:
                shape_str = str(shape)

        lines.append(
            f"{name} | {n_elems} | {shape_str} | {qt_name} [код {code}] | {fmt_bytes(n_bytes)}"
        )

    text = "\n".join(lines)

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        messagebox.showinfo(
            "Тензоры",
            f"Файл с тензорами создан:\n{txt_path}"
        )
    except Exception as e:
        messagebox.showerror("Ошибка записи файла", str(e))


def show_big_message(text):
    win = tk.Toplevel()
    win.title("Диагностика квантовки GGUF")

    frame = tk.Frame(win)
    frame.pack(fill="both", expand=True)

    # Увеличение шрифта примерно на треть:
    # было 12 -> стало 16
    text_widget = tk.Text(frame, wrap="word", font=("Segoe UI", 16))
    scroll = tk.Scrollbar(frame, command=text_widget.yview)
    text_widget.configure(yscrollcommand=scroll.set)

    text_widget.insert("1.0", text)
    text_widget.configure(state="disabled")

    text_widget.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    # было 14 -> стало 19 (округлённое 14 * 4/3)
    btn = tk.Button(win, text="OK", font=("Segoe UI", 19), command=win.destroy)
    btn.pack(pady=10)

    win.geometry("1100x900")
    win.resizable(True, True)


def on_close():
    global current_tensor_txt_path
    if current_tensor_txt_path and os.path.exists(current_tensor_txt_path):
        try:
            os.remove(current_tensor_txt_path)
        except OSError:
            pass
    root.destroy()


root = tk.Tk()
root.title("GGUF квантовка — диагностика")
root.geometry("450x220")

# Увеличенный шрифт кнопок (добавлен, раньше был шрифт по умолчанию)
btn = tk.Button(root, text="Выбрать GGUF и проанализировать", command=choose_file,
                font=("Segoe UI", 18))
btn.pack(expand=True, padx=20, pady=(20, 10))

btn2 = tk.Button(root, text="Показать тензоры", command=show_tensors,
                 font=("Segoe UI", 18))
btn2.pack(expand=True, padx=20, pady=(0, 20))

root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()
