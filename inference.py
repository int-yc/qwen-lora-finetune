
import os
os.environ["HF_HOME"] = r"D:\LoRA\hf_cache"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "output/qwen-lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("加载基座模型...")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
)

print("加载 LoRA adapter...")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

QUESTIONS = [
    "什么是 RAG？",
    "LoRA 是什么？",
    "你能帮我做什么？",
]

for q in QUESTIONS:
    messages = [
        {"role": "system", "content": "你是一个严谨的大模型技术助手，回答简洁、准确、不编造。"},
        {"role": "user", "content": q},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n问：", q)
    print("答：", answer)
