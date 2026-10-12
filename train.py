
import os
os.environ["HF_HOME"] = r"D:\LoRA\hf_cache"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "output/qwen-lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_chat(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

dataset = dataset.map(format_chat)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=8,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=2,
    save_strategy="no",
    max_length=512,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=MODEL_NAME,
    args=args,
    train_dataset=dataset,
    peft_config=lora_config,
)

trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("训练完成，adapter 已保存到", OUTPUT_DIR)
