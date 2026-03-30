import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
INPUT_JSONL = "outputs/prompts/meld_train_prompts_k3.jsonl"
OUTPUT_JSONL = "outputs/prompts/meld_train_reasoning_qwen7b.jsonl"

MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7
TOP_P = 0.9

def count_existing_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    out_path = Path(OUTPUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = count_existing_lines(OUTPUT_JSONL)
    print(f"Resuming from line {done}")

    with open(INPUT_JSONL, "r", encoding="utf-8") as fin:
        all_lines = fin.readlines()

    remaining_lines = all_lines[done:]

    with open(out_path, "a", encoding="utf-8") as fout:
        for line in tqdm(remaining_lines, desc="Generating reasoning"):
            item = json.loads(line)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": item["prompt"]}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    eos_token_id=tokenizer.eos_token_id
                )

            gen_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            ).strip()

            item["reasoning"] = gen_text
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"✅ Reasoning saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()