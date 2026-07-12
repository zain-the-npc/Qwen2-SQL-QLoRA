"""
QLoRA fine-tuning of Qwen2-7B-Instruct on the Gretel synthetic text-to-SQL dataset.

Designed to run on a single free-tier T4 GPU (16GB), e.g. Google Colab.
T4 is a Turing-generation GPU with no native bf16 tensor core support --
this script deliberately uses fp16 throughout, not bf16.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
OUTPUT_DIR = "./qwen2-7b-sql-qlora-final"
NUM_TRAIN_EXAMPLES = 4000
NUM_TEST_EXAMPLES = 300


def format_example(example, tokenizer):
    system_msg = "You are a helpful assistant that writes SQL queries given a database schema and a question."
    user_msg = f"### Database Schema:\n{example['sql_context']}\n\n### Question:\n{example['sql_prompt']}"
    assistant_msg = example["sql"]

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def main():
    print("Loading dataset...")
    dataset = load_dataset("philschmid/gretel-synthetic-text-to-sql")
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(NUM_TRAIN_EXAMPLES))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(NUM_TEST_EXAMPLES))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = dataset.map(lambda ex: format_example(ex, tokenizer))

    print("Loading base model in 4-bit (NF4, double quant, fp16 compute)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    print("Attaching LoRA adapter...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir="./qwen2-7b-sql-qlora-checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,   # base model is already fp16; avoid GradScaler dtype conflicts
        bf16=False,   # T4 has no native bf16 support
        max_length=512,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {OUTPUT_DIR} ...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done. Adapter saved.")


if __name__ == "__main__":
    main()
