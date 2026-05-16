import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. 设定硬件与模型参数（2048上下文，对大显卡来说轻轻松松）
max_seq_length = 2048
dtype = None # 自动检测（大显卡会自动开启 bfloat16）
load_in_4bit = True # 开启 4-bit 量化加速

# 2. 加载基础模型与分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/qwen2.5-1.5b-bnb-4bit", # 锁定1.5B小钢炮
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. 设置 LoRA 微调参数
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. 加载你上传的本地数据集
# 提示：请确保 weibu_dataset.jsonl 文件和本脚本放在同一个文件夹下
dataset = load_dataset("json", data_files="weibu_dataset.jsonl", split="train")

# 5. 设置训练超参数
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # 确保你的 jsonl 数据集里每条数据的 key 是 "text"
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # 20条语料，跑 60 步左右人设就很稳了
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)

# 6. 开始特训！
print("⚡ 韦布正在好兄弟的显卡里全力冲锋... ⚡")
trainer_stats = trainer.train()

# 7. 终极打包：直接导出为 Ollama 专用的 GGUF 文件
print("📦 训练完成！正在打包并量化为 GGUF 格式...")
model.save_pretrained_gguf("weibu_model", tokenizer, quantization_method = "q4_k_m")
print("🎉 大功告成！请把生成的 'weibu_model-unsloth.Q4_K_M.gguf' 文件发给主人！")