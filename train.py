import os
# CRITICAL FIX: Force Unsloth to use ModelScope backend instead of HuggingFace
os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"
os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.cn"

import sys
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

def main():
    # 1. Configuration Constants
    max_seq_length = 2048 # Supports RoPE Scaling automatically
    dtype = None          # None for auto-detection
    load_in_4bit = True   # Use 4-bit quantization to save massive VRAM
    
    # Path to the model downloaded via download_qwen.py
    local_model_path = "./qwen3-3b/qwen/Qwen2.5-3B-Instruct"
    dataset_path = "weibu_dataset.jsonl"
    output_dir = "outputs"

    print("=== [Phase 1: Loading Local Model & Tokenizer] ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = local_model_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    print("\n=== [Phase 2: Setting up LoRA Architecture] ===")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    print("\n=== [Phase 3: Preparing Chat Dataset] ===")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5",
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    dataset = dataset.map(formatting_prompts_func, batched=True)

    print("\n=== [Phase 4: Launching SFT Trainer] ===")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 100, # Increased to 100 steps for even better intelligence!
            learning_rate = 2e-4,
            fp16 = False,
            bf16 = True,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
            report_to = "none",
            # CRITICAL FIX: Disable mid-training saving to bypass Pickle error
            save_strategy = "no",
            save_total_limit = 0,
        ),
    )

    # Trigger the training engine
    trainer_stats = trainer.train()
    print(f"\nTraining completed! Runtime: {trainer_stats.metrics['train_runtime']} seconds.")

    print("\n=== [Phase 5: Exporting to Perfect Completely Unified GGUF] ===")
    # Save the completely merged Q4_K_M GGUF format directly for Ollama
    model.save_pretrained_gguf(
        output_dir, 
        tokenizer, 
        quantization_method = "q4_k_m"
    )
    print(f"Success! Your target GGUF file is saved inside the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
