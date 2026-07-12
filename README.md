# Qwen2-7B QLoRA — Natural Language to SQL

Fine-tuning **Qwen2-7B-Instruct** with **QLoRA** (4-bit quantization + Low-Rank Adaptation) to generate SQL queries from a database schema and a natural-language question.

This is the follow-up to my earlier [Adaptive-Gpt2-LoRA](https://github.com/zain-the-npc/Adaptive-Gpt2-LoRA) project, where I hand-implemented LoRA at the weight level on GPT-2 to understand the mechanics. This project builds on that understanding and applies QLoRA to a model 60x larger (7.6B vs 124M params) — a scale where quantization isn't optional, it's the only reason fine-tuning is possible on a single free-tier GPU.

> **Status: training complete, adapter included in `adapter/`.** Execution-accuracy numbers below are pending final eval run — see Evaluation section.

---

## Why QLoRA, why this model

Qwen2-7B-Instruct in 4-bit takes roughly 4-5GB of VRAM just for the frozen base weights — small enough to fit on a free Colab T4 (16GB), but *too big* to comfortably full-fine-tune or even LoRA-tune in fp16 on the same hardware. That's the actual point of this project: unlike a 1-3B model (which could be tuned without quantization at all), a 7B model on a T4 genuinely needs QLoRA to work. This isn't a toy demonstration — it's the constraint QLoRA was built to solve.

**Qwen2-7B-Instruct** was chosen because it's already a strong general-purpose coding/reasoning model — which makes "before vs after" a fair test of whether fine-tuning improves *consistency on a narrow task*, not just "fixing a broken model."

## Why text-to-SQL specifically

Instead of generic "coding Q&A" (where Qwen2 is already broadly capable and improvement is hard to measure objectively), text-to-SQL was chosen because it's:

- **Narrow enough** that a general-purpose model is inconsistent on it (wrong joins, wrong column references, ignoring schema constraints) even though it "can code."
- **Verifiable** — generated SQL can be executed against a real database and checked for a matching result, giving an objective, non-subjective before/after metric instead of eyeballed quality judgments.

## Method

| Component | Choice | Why |
|---|---|---|
| Base model | `Qwen/Qwen2-7B-Instruct` | Strong small-7B baseline, Apache 2.0 license, GQA for efficiency |
| Quantization | 4-bit NF4, double quant | QLoRA paper's recommended config, biggest memory win for smallest quality cost |
| Adapter | LoRA, r=16, alpha=32 | Targets `q_proj, k_proj, v_proj, o_proj` — attention only, ~10M trainable params (0.13% of total) |
| Dataset | [`philschmid/gretel-synthetic-text-to-sql`](https://huggingface.co/datasets/philschmid/gretel-synthetic-text-to-sql) | Synthetic but high-quality, includes schema + question + query + reasoning, 100k examples downsampled to 4,000 train / 300 test |
| Hardware | Free Colab T4 (16GB) | Consumer-accessible constraint, same tier as this repo's target audience |
| Precision | fp16 (not bf16) | T4 is a Turing-generation GPU — no native bf16 tensor core support; bf16 silently runs in slow software emulation |

## Evaluation

Fine-tuned and base model are compared on the **same 100 held-out test questions**, using **execution accuracy**: each generated SQL query is run against a real SQLite database built from that example's schema, and the result is compared to the result of the ground-truth query. This checks whether the SQL actually *works*, not just whether it looks plausible.

Base-model comparison uses `peft`'s `model.disable_adapter()` context manager — same weights, same prompts, adapter toggled on/off — to isolate the effect of fine-tuning from any other variable.

```
Execution Accuracy — Base Qwen2-7B-Instruct:      TBD
Execution Accuracy — Fine-tuned (QLoRA adapter):  TBD
Improvement:                                       TBD
```

## Repo structure

```
qwen2-sql-qlora/
├── README.md
├── requirements.txt
├── scripts/
│   ├── train.py           # QLoRA training script
│   ├── evaluate.py        # Execution-accuracy eval (base vs fine-tuned)
│   └── app_gradio.py       # Side-by-side demo UI
└── notebooks/
    └── qwen2_sql_qlora.ipynb   # Full Colab notebook (training + eval)
```

## Running it yourself

```bash
pip install -r requirements.txt
python scripts/train.py          # trains the adapter, saves to ./qwen2-7b-sql-qlora-final
python scripts/evaluate.py       # prints base vs fine-tuned execution accuracy
python scripts/app_gradio.py     # launches the side-by-side comparison demo
```

Trained on a free Colab T4. Full run: ~2.5 hours for 2 epochs on 4,000 examples.

## Honest limitations

- Trained on synthetic data (Gretel) — real-world schemas with messier naming conventions may perform differently.
- 300-example eval set is a reasonable sanity check, not a rigorous benchmark like Spider or BIRD.
- Complex multi-join queries are the hardest case for a 7B model regardless of fine-tuning; this project targets consistency gains, not solving arbitrarily hard SQL.
- 2 epochs / 4,000 examples was chosen to fit a realistic single-session training budget on free-tier hardware, not to maximize accuracy at any cost.

## Related work

- [Adaptive-Gpt2-LoRA](https://github.com/zain-the-npc/Adaptive-Gpt2-LoRA) — hand-written LoRA implementation (no `peft`) on GPT-2, the mechanistic prerequisite to this project.
