"""
Execution-accuracy evaluation: base Qwen2-7B-Instruct vs the QLoRA fine-tuned adapter.

For each test example, the generated SQL is executed against an in-memory SQLite
database built from that example's schema, and the result is compared against the
result of the ground-truth query. This checks whether the SQL actually *works*,
not just whether it superficially resembles the correct answer.

Base-model behavior is obtained via `peft`'s `model.disable_adapter()` context
manager -- same weights, same prompts, adapter toggled on/off -- so the only
variable between "before" and "after" is whether the adapter is active.
"""

import sqlite3

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
ADAPTER_PATH = "./qwen2-7b-sql-qlora-final"
NUM_EVAL_EXAMPLES = 100


def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def create_db_from_schema(schema_sql):
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    try:
        cursor.executescript(schema_sql)
        conn.commit()
    except Exception as e:
        return None, str(e)
    return conn, None


def run_sql(conn, sql):
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    except Exception as e:
        return f"ERROR: {e}"


def generate_sql(model, tokenizer, schema, question, max_new_tokens=200):
    system_msg = "You are a helpful assistant that writes SQL queries given a database schema and a question."
    user_msg = f"### Database Schema:\n{schema}\n\n### Question:\n{question}"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def evaluate_model(model, tokenizer, test_examples, num_samples=100):
    correct, total = 0, 0
    results = []

    for example in test_examples.select(range(min(num_samples, len(test_examples)))):
        schema = example["sql_context"]
        question = example["sql_prompt"]
        gold_sql = example["sql"]

        conn, err = create_db_from_schema(schema)
        if conn is None:
            continue

        predicted_sql = generate_sql(model, tokenizer, schema, question)
        gold_result = run_sql(conn, gold_sql)
        pred_result = run_sql(conn, predicted_sql)

        is_correct = (gold_result == pred_result) and not isinstance(pred_result, str)
        correct += int(is_correct)
        total += 1

        results.append({
            "question": question,
            "gold_sql": gold_sql,
            "predicted_sql": predicted_sql,
            "correct": is_correct,
        })
        conn.close()

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


def main():
    print("Loading model + adapter...")
    model, tokenizer = load_model_and_tokenizer()

    print("Loading test set...")
    dataset = load_dataset("philschmid/gretel-synthetic-text-to-sql")
    test_set = dataset["test"].shuffle(seed=42).select(range(NUM_EVAL_EXAMPLES))

    print("Evaluating fine-tuned model (adapter ON)...")
    ft_accuracy, ft_results = evaluate_model(model, tokenizer, test_set, NUM_EVAL_EXAMPLES)
    print(f"Execution Accuracy (fine-tuned): {ft_accuracy:.2%}")

    print("Evaluating base model (adapter OFF)...")
    with model.disable_adapter():
        base_accuracy, base_results = evaluate_model(model, tokenizer, test_set, NUM_EVAL_EXAMPLES)
    print(f"Execution Accuracy (base model): {base_accuracy:.2%}")

    print(f"\nImprovement: {ft_accuracy - base_accuracy:+.2%}")

    print("\nSample comparisons:")
    for r in ft_results[:3]:
        print("\n---")
        print("Q:", r["question"])
        print("Gold SQL:", r["gold_sql"])
        print("Predicted SQL:", r["predicted_sql"])
        print("Correct:", r["correct"])


if __name__ == "__main__":
    main()
