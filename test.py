import os
# 強制鎖定本地與魔搭通道，徹底隔絕外網檢查
os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"
os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.cn"

from unsloth import FastLanguageModel
from transformers import TextStreamer

max_seq_length = 2048
dtype = None
load_in_4bit = True

print("=== [正在直接載入你練好的新大腦皮層] ===")
# 直接讀取 outputs，Unsloth 會自動抓取裡面的配置並結合底模
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 切換到極速推理模式
FastLanguageModel.for_inference(model)

# 測試對話
messages = [
    {"role": "user", "content": "你是誰？請介紹一下你自己。"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
print("\n=== [全新大腦回答開始] ===")
_ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, temperature=0.3)
print("=== [全新大腦回答結束] ===\n")
