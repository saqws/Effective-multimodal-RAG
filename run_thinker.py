import torch
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
    BitsAndBytesConfig,
)

# 4‑bit NF4 config
quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quant_cfg,
)

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")


save_directory = "./qwen2_5_omni_7b_4bit"
model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)