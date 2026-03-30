import csv
import json
from pathlib import Path

TRAIN_CSV = "datasets/MELD/train_meld_emo.csv"
REASONING_JSONL = "outputs/prompts/meld_train_reasoning_qwen7b.jsonl"
OUT_CSV = "datasets/MELD/train_meld_emo_augmented.csv"

def load_reasoning_map(path):
    reasoning_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            key = (int(item["dialogue_id"]), int(item["utterance_id"]))
            reasoning_map[key] = item["reasoning"].strip()
    return reasoning_map

def main():
    reasoning_map = load_reasoning_map(REASONING_JSONL)

    with open(TRAIN_CSV, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)
        fieldnames = reader.fieldnames + ["Augmented_Utterance"]

    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matched = 0
    with open(out_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            key = (int(row["Dialogue_ID"]), int(row["Utterance_ID"]))
            reasoning = reasoning_map.get(key, "").strip()

            if reasoning:
                row["Augmented_Utterance"] = f'{row["Utterance"]} [SEP] {reasoning}'
                matched += 1
            else:
                row["Augmented_Utterance"] = row["Utterance"]

            writer.writerow(row)

    print(f"✅ Wrote augmented CSV to {OUT_CSV}")
    print(f"✅ Matched reasoning for {matched}/{len(rows)} rows")

if __name__ == "__main__":
    main()