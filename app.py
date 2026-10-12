import os
os.environ["HF_HOME"] = r"D:\LoRA\hf_cache"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from threading import Thread
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "output/qwen-lora"
SYSTEM_PROMPT = "你是一个严谨的大模型技术助手，回答简洁、准确、不编造。"

st.set_page_config(page_title="大模型知识问答助手", layout="centered")

@st.cache_resource(show_spinner="正在加载模型，首次约 1 分钟...")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return tok, model

def ask_stream(tok, model, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs, streamer=streamer, max_new_tokens=200, do_sample=False
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for new_text in streamer:
        yield new_text

st.title("大模型知识问答助手")
st.caption("基于 Qwen2.5-1.5B 经 LoRA 微调，专攻大模型基础概念问答")

example = ["RAG 是什么？", "LoRA 是什么？", "向量检索和关键词检索有什么区别？"]
cols = st.columns(len(example))
for i, q in enumerate(example):
    with cols[i]:
        if st.button(q):
            st.session_state["q"] = q

question = st.text_input(
    "请输入你的问题",
    value=st.session_state.get("q", "RAG 是什么？"),
)

if st.button("提问", type="primary", use_container_width=True):
    tok, model = load_model()
    st.markdown("### 回答")
    with st.chat_message("assistant"):
        st.write_stream(ask_stream(tok, model, question))
