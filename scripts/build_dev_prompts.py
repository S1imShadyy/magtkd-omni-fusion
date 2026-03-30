import json
import pandas as pd
from pathlib import Path

K = 3
IN_CSV = "datasets/MELD/dev_meld_emo.csv"
OUT_JSONL = "outputs/prompts/meld_dev_prompts_k3.jsonl"

def main():
    df = pd.read_csv(IN_CSV)

    # 必要列
    need_cols = ["Dialogue_ID", "Utterance_ID", "Speaker", "Utterance", "Emotion"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}. Got: {list(df.columns)}")

    # 排序：同一对话内按顺序
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)

    out_path = Path(OUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for did, g in df.groupby("Dialogue_ID", sort=False):
            g = g.reset_index(drop=True)
            for idx in range(len(g)):
                # 取最近 K 条历史（不含当前）
                start = max(0, idx - K)
                hist = g.loc[start:idx-1, ["Speaker", "Utterance"]]

                # 格式化 history
                history_lines = []
                for _, row in hist.iterrows():
                    spk = str(row["Speaker"]).strip()
                    utt = str(row["Utterance"]).strip()
                    history_lines.append(f"{spk}: {utt}")

                history_text = "\n".join(history_lines) if history_lines else "(no previous context)"

                cur_spk = str(g.loc[idx, "Speaker"]).strip()
                cur_utt = str(g.loc[idx, "Utterance"]).strip()

                # 这是你后面喂给 LLM 的输入（prompt）
                prompt = (
                    "You are an expert in conversational understanding.\n"
                    "Task: Given the dialogue history and the current utterance, write 1-2 sentences of reasoning context "
                    "explaining the likely intent/tone shift and key semantic cues.\n"
                    "Constraints:\n"
                    "- Do NOT guess or mention the emotion label name.\n"
                    "- Keep it concise (max 40 words).\n"
                    "- Output only the reasoning text.\n\n"
                    f"Dialogue history (last {K} utterances):\n{history_text}\n\n"
                    f"Current utterance:\n{cur_spk}: {cur_utt}\n"
                )

                record = {
                    "dataset": "MELD",
                    "split": "dev",
                    "dialogue_id": int(did),
                    "utterance_id": int(g.loc[idx, "Utterance_ID"]),
                    "speaker": cur_spk,
                    "utterance": cur_utt,
                    "label": str(g.loc[idx, "Emotion"]),
                    "history_k": K,
                    "history_text": history_text,
                    "prompt": prompt,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"✅ Wrote {n_written} prompts to {OUT_JSONL}")

if __name__ == "__main__":
    main()