# описание, какие шаги нужны, чтобы получить GGUF заданного формата

# ---------- Базовые форматы (через convert_hf_to_gguf) ----------

F16_FORMAT = {
    "name": "F16",
    "kind": "base",          # базовый формат, только convert_hf_to_gguf
    "outtype": "f16",        # что передаём в --outtype
    "quantize_type": None,   # quantize не нужен
    "need_f16_first": False, # промежуточный F16 не нужен
}

Q8_0_FORMAT = {
    "name": "Q8_0",
    "kind": "base",
    "outtype": "q8_0",
    "quantize_type": None,
    "need_f16_first": False,
}

# ---------- Q4-форматы (через quantize) ----------

Q4_K_M_FORMAT = {
    "name": "Q4_K_M",
    "kind": "q4",            # квантованный формат, нужен quantize
    "outtype": "f16",        # сначала делаем F16 через convert_hf_to_gguf
    "quantize_type": "Q4_K_M",
    "need_f16_first": True,  # обязательно сначала F16
}

Q4_K_S_FORMAT = {
    "name": "Q4_K_S",
    "kind": "q4",
    "outtype": "f16",
    "quantize_type": "Q4_K_S",
    "need_f16_first": True,
}