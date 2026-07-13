# Qwen2-7B QLoRA — Natural Language to SQL

Fine-tuning **Qwen2-7B-Instruct** with **QLoRA** (4-bit quantization + Low-Rank Adaptation) to generate SQL queries from a database schema and a natural-language question.

Follow-up to my earlier [Adaptive-Gpt2-LoRA](link) project, where I hand-implemented LoRA at the weight level on GPT-2 to understand the mechanics. This project applies that understanding to a model 60x larger (7.6B vs 124M params) — a scale where quantization isn't optional, it's the only reason fine-tuning is possible on a single free-tier GPU.

## Results

| Model | Execution Accuracy |
|---|---|
| Base (Qwen2-7B-Instruct, no adapter) | 3.12% |
| Fine-tuned (QLoRA adapter) | **57.29%** |
| Improvement | +54.17% |

Evaluated on 100 held-out samples from [gretel-synthetic-text-to-sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql), execution accuracy (predicted SQL run against schema, output compared to gold).

## Why QLoRA, why this model

Qwen2-7B-Instruct in 4-bit takes ~4-5GB VRAM for frozen base weights — small enough to fit on a free Colab T4 (16GB), but too big to comfortably full-fine-tune or even LoRA-tune in fp16 on the same hardware. Unlike a 1-3B model (tunable without quantization), a 7B model on a T4 genuinely needs QLoRA to work.

Qwen2-7B-Instruct was chosen because it's already a strong general-purpose coding/reasoning model — making "before vs after" a fair test of whether fine-tuning improves *consistency on a narrow task*, not just "fixing a broken model."

## Example predictions

| Question | Base Output | Fine-tuned Output |
|---|---|---|
| Avg salary per department | *(base often malformed/incomplete SQL)* | `SELECT department, AVG(salary) FROM employees GROUP BY department;` |
| Products priced over 100 | *(often ignores schema columns)* | `SELECT * FROM products WHERE price > 100;` |

*(fill in 3-5 real pairs — run `scripts/compare_examples.py` or the Colab cell to regenerate)*

## Adapter

Trained LoRA adapter, hosted on Hugging Face: **[zain-the-npc/qwen2-7b-sql-qlora](https://huggingface.co/zain-the-npc/qwen2-7b-sql-qlora)** (verified upload, 31.6 MB — adapter weights + tokenizer, no base model included).

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", load_in_4bit=True)
model = PeftModel.from_pretrained(base, "zain-the-npc/qwen2-7b-sql-qlora")
```

## Repo structure
