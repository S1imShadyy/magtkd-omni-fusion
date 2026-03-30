#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from preprocessing import preprocessing


# ==============================
# 一、基础工具函数
# ==============================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def normalize_text(text: Optional[str]) -> str:
 
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text



def make_record_key(split: str, dialogue_id: str, utterance_id: str) -> str:
    return f"{split}::{dialogue_id}::{utterance_id}"



def prepare_out_dir(out_root: Path, overwrite: bool = False) -> None:
    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)
    ensure_dir(out_root)
    ensure_dir(out_root / "train")
    ensure_dir(out_root / "dev")
    ensure_dir(out_root / "test")



def append_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")



def load_existing_status_map(index_path: Path) -> Dict[str, Dict]:
    status_map: Dict[str, Dict] = {}
    if not index_path.exists():
        return status_map

    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rk = obj.get("record_key")
                if rk:
                    status_map[rk] = obj
            except Exception:
                continue
    return status_map


# ==============================
# 二、数据结构定义
# ==============================


@dataclass
class FlattenedSample:
    global_index: int
    item: Dict
    history: List[Dict]


# ==============================
# 三、媒体路径解析
# ==============================


def resolve_media_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()



def infer_wav_candidates(video_path: Path, raw_wav_path: Path, split_name: str) -> List[Path]:
    candidates: List[Path] = []

    candidates.append(video_path.with_suffix('.wav'))

    raw_str = str(raw_wav_path)
    if split_name == 'train':
        candidates.append(raw_wav_path)
    else:
        if 'train_splits' not in raw_str:
            candidates.append(raw_wav_path)

    deduped: List[Path] = []
    seen = set()
    for p in candidates:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp not in seen:
            deduped.append(p)
            seen.add(rp)
    return deduped



def validate_media(item: Dict, project_root: Path, split_name: str) -> Tuple[Path, Path]:
    video_path = resolve_media_path(item["video_path"], project_root)
    raw_wav_path = resolve_media_path(item["wav_path"], project_root)

    if not video_path.exists():
        raise FileNotFoundError(f"Missing video file: {video_path}")

    for candidate in infer_wav_candidates(video_path, raw_wav_path, split_name):
        if candidate.exists():
            return candidate, video_path

    raise FileNotFoundError(
        "Missing wav file candidates: "
        + ", ".join(str(p) for p in infer_wav_candidates(video_path, raw_wav_path, split_name))
    )


# ==============================
# 四、读取并展开 MELD split
# ==============================


def load_split_dialogues(data_path: str, split_type: str) -> List[List[Dict]]:
    session_dataset = preprocessing(data_path, split_type)

    grouped: Dict[str, List[Dict]] = {}

    seen_keys = set()

    for dialogue_prefix in session_dataset:
        for item in dialogue_prefix:
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, split_name, dialogue_id, utterance_id = item

            dialogue_id = str(dialogue_id)
            utterance_id = str(utterance_id)
            dedup_key = (dialogue_id, utterance_id)

            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            utt_dict = {
                "speaker": speaker,
                "utterance_text": utt,
                "wav_path": wav_path,
                "video_path": video_path,
                "start_time": start_time,
                "end_time": end_time,
                "emotion": emotion,
                "split": split_name,
                "dialogue_id": dialogue_id,
                "utterance_id": utterance_id,
            }

            grouped.setdefault(dialogue_id, []).append(utt_dict)

    dialogues: List[List[Dict]] = []
    for dialogue_id in sorted(grouped.keys(), key=lambda x: int(x)):
        utterances = sorted(grouped[dialogue_id], key=lambda x: int(x["utterance_id"]))
        dialogues.append(utterances)

    return dialogues



def flatten_dialogues(dialogues: List[List[Dict]], context_window: int) -> List[FlattenedSample]:
    flattened: List[FlattenedSample] = []
    global_idx = 0

    for dialogue in dialogues:
        for i, item in enumerate(dialogue):
            history = dialogue[max(0, i - context_window): i]
            flattened.append(
                FlattenedSample(
                    global_index=global_idx,
                    item=item,
                    history=history,
                )
            )
            global_idx += 1

    return flattened


# ==============================
# 五、构造 prompt
# ==============================


def build_multimodal_prompt(item: Dict, history: List[Dict]) -> Dict[str, str]:
    
    if history:
        history_lines = [
            f"[Speaker: {h['speaker']}] {normalize_text(h['utterance_text'])}"
            for h in history
        ]
        history_text = "\n".join(history_lines)
    else:
        history_text = "(No previous context)"

    current_text = normalize_text(item["utterance_text"])

    system_prompt = (
        "You are a multimodal dialogue understanding assistant. "
        "You will analyze one utterance using text, audio, and video evidence jointly. "
        "Do not output the gold emotion label. "
        "Focus on observable evidence and cross-modal interpretation. "
        "Return only the requested structured fields."
    )

    user_prompt = f"""
Dialogue history:
{history_text}

Current utterance:
[Speaker: {item['speaker']}] {current_text}

You will receive the current utterance's audio and video together with this text.
Please analyze the current utterance based on:
1. text content,
2. audio cues (tone, energy, speaking style, pauses, emphasis, etc.),
3. visual/video cues (facial expression, gesture, movement, visible reaction, etc.).

Return your answer exactly in this format:
text_evidence: ...
audio_evidence: ...
video_evidence: ...
cross_modal_reasoning: ...
multimodal_summary: ...
""".strip()

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "history_text": history_text,
    }


# ==============================
# 六、多模态响应规范化
# ==============================


def default_mm_fields() -> Dict[str, str]:
    return {
        "text_evidence": "unavailable",
        "audio_evidence": "unavailable",
        "video_evidence": "unavailable",
        "cross_modal_reasoning": "unavailable",
        "multimodal_summary": "unavailable",
    }



def normalize_multimodal_response(raw_response: str) -> Dict[str, str]:
    
    fields = default_mm_fields()

    text = (raw_response or "").strip()
    if not text:
        return fields

    pattern = re.compile(
        r"(?is)(text_evidence|audio_evidence|video_evidence|cross_modal_reasoning|multimodal_summary)\s*:\s*(.*?)"
        r"(?=(?:text_evidence|audio_evidence|video_evidence|cross_modal_reasoning|multimodal_summary)\s*:|$)"
    )

    matched_any = False
    for name, value in pattern.findall(text):
        matched_any = True
        key = name.strip().lower()
        value = normalize_text(value)
        if value:
            fields[key] = value

    if not matched_any:
        fields["multimodal_summary"] = normalize_text(text)
        fields["cross_modal_reasoning"] = "Response did not follow the requested structure; raw response stored as summary."

    return fields



def pack_summary_text(mm_fields: Dict[str, str]) -> str:
    return (
        f"text_evidence: {mm_fields['text_evidence']}\n"
        f"audio_evidence: {mm_fields['audio_evidence']}\n"
        f"video_evidence: {mm_fields['video_evidence']}\n"
        f"cross_modal_reasoning: {mm_fields['cross_modal_reasoning']}\n"
        f"multimodal_summary: {mm_fields['multimodal_summary']}"
    )


# ==============================
# 七、模型后端：Mock 与 Qwen Omni
# ==============================


class MockMultimodalGenerator:
    
    def __init__(self) -> None:
        pass

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        wav_path: Path,
        video_path: Path,
    ) -> str:
        return (
            "text_evidence: The current utterance contains semantically meaningful content.\n"
            "audio_evidence: Mock mode does not inspect real audio; this field validates the pipeline only.\n"
            "video_evidence: Mock mode does not inspect real video; this field validates the pipeline only.\n"
            "cross_modal_reasoning: Mock mode is used only to verify data flow, caching, and fusion compatibility.\n"
            "multimodal_summary: Placeholder multimodal semantic summary generated in mock mode."
        )


class QwenOmniGenerator:
    
    def __init__(
        self,
        model_name: str,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 256,
        fps: float = 1.0,
        max_pixels: Optional[int] = None,
        use_audio_in_video: bool = False,
        attn_implementation: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.fps = fps
        self.max_pixels = max_pixels
        self.use_audio_in_video = use_audio_in_video

        try:
            from transformers import (
                Qwen2_5OmniProcessor,
                Qwen2_5OmniThinkerForConditionalGeneration,
            )
        except Exception as e:
            raise ImportError(
                "无法导入 Qwen2.5-Omni 所需的 transformers 类。\n"
                "请确认当前环境安装了支持 Qwen2.5-Omni 的 transformers 版本。"
            ) from e

        # 统一 dtype 解析。
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "auto": "auto",
        }
        dtype_value = dtype_map.get(torch_dtype.lower(), torch.bfloat16)

        model_kwargs = {
            "device_map": device_map,
        }
        if dtype_value != "auto":
            model_kwargs["dtype"] = dtype_value
        else:
            model_kwargs["dtype"] = "auto"

        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        print(f"[QwenOmniGenerator] loading processor: {model_name}")
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

        if max_pixels is not None:
            try:
                self.processor.max_pixels = max_pixels
            except Exception:
                pass

        print(f"[QwenOmniGenerator] loading thinker model: {model_name}")
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
        )
        self.model.eval()

    def _build_conversation(
        self,
        system_prompt: str,
        user_prompt: str,
        wav_path: Path,
        video_path: Path,
    ) -> List[Dict]:
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {"type": "audio", "audio": str(wav_path)},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

    @torch.inference_mode()
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        wav_path: Path,
        video_path: Path,
    ) -> str:
        conversations = self._build_conversation(system_prompt, user_prompt, wav_path, video_path)

        processor_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
            "fps": self.fps,
            "load_audio_from_video": self.use_audio_in_video,
            "use_audio_in_video": self.use_audio_in_video,
        }

        inputs = self.processor.apply_chat_template(
            conversations,
            **processor_kwargs,
        )

        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_audio_in_video=self.use_audio_in_video,
        )

        input_ids = inputs.get("input_ids", None)
        if input_ids is not None:
            prompt_len = input_ids.shape[1]
            generated_ids = generated_ids[:, prompt_len:]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return response



def build_generator(args):
    if args.backend == "mock_mm":
        return MockMultimodalGenerator()

    if args.backend == "qwen_omni_hf":
        return QwenOmniGenerator(
            model_name=args.omni_model_name,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            max_new_tokens=args.max_new_tokens,
            fps=args.video_fps,
            max_pixels=args.max_pixels,
            use_audio_in_video=args.use_audio_in_video,
            attn_implementation=args.attn_implementation,
        )

    raise ValueError(f"Unsupported backend: {args.backend}")


# ==============================
# 八、summary -> z_m 编码器
# ==============================


class SemanticSummaryEncoder:
    def __init__(self, model_name: str, target_dim: int = 768, device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.target_dim = target_dim
        self.device = device

        # sentence-transformers 内部会自行处理 device；这里不强依赖 GPU。
        self.model = SentenceTransformer(model_name, device=device)

    def _fit_dim(self, emb: np.ndarray) -> np.ndarray:
        emb = emb.astype(np.float32)
        dim = emb.shape[0]
        if dim == self.target_dim:
            return emb
        if dim > self.target_dim:
            return emb[: self.target_dim].astype(np.float32)

        # dim < target_dim 时补零
        out = np.zeros((self.target_dim,), dtype=np.float32)
        out[:dim] = emb
        return out

    def encode(self, summary_text: str) -> np.ndarray:
        emb = self.model.encode(
            [summary_text],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )[0]
        emb = self._fit_dim(emb)
        return emb.astype(np.float32)


# ==============================
# 九、文件命名与保存
# ==============================


def make_vector_relpath(split_name: str, dialogue_id: str, utterance_id: str) -> Path:
    return Path(split_name) / f"{split_name}_{dialogue_id}_{utterance_id}_zm.npy"



def save_vector(out_root: Path, relpath: Path, vector: np.ndarray) -> None:
    abs_path = out_root / relpath
    ensure_dir(abs_path.parent)
    np.save(abs_path, vector.astype(np.float32))


# ==============================
# 十、split 主处理函数
# ==============================


def process_split(
    split_name: str,
    data_path: str,
    out_root: Path,
    project_root: Path,
    generator,
    encoder: SemanticSummaryEncoder,
    context_window: int,
    limit_per_split: Optional[int],
    start_index: int,
    skip_existing: bool,
    sleep_per_sample: float,
) -> None:
    print(f"\n========== Processing split: {split_name} ==========")
    index_path = out_root / f"index_{split_name}.jsonl"
    existing_status_map = load_existing_status_map(index_path) if skip_existing else {}

    dialogues = load_split_dialogues(data_path, split_name)
    samples = flatten_dialogues(dialogues, context_window=context_window)

    samples = [s for s in samples if s.global_index >= start_index]
    if limit_per_split is not None:
        samples = samples[:limit_per_split]

    print(f"[{split_name}] total selected samples: {len(samples)}")

    for idx, sample in enumerate(samples, start=1):
        item = sample.item
        record_key = make_record_key(split_name, item["dialogue_id"], item["utterance_id"])
        rel_vector_path = make_vector_relpath(split_name, item["dialogue_id"], item["utterance_id"])
        abs_vector_path = out_root / rel_vector_path

        if skip_existing:
            existing_obj = existing_status_map.get(record_key)
            if existing_obj is not None:
                existing_status = existing_obj.get("generation_status", "")
                existing_vector_file = existing_obj.get("vector_file")
                existing_abs_vector = out_root / existing_vector_file if existing_vector_file else abs_vector_path

                if existing_status == "ok" and existing_abs_vector.exists():
                    print(f"[{split_name}] skip existing ok ({idx}/{len(samples)}): {record_key}")
                    continue

        print(f"[{split_name}] processing ({idx}/{len(samples)}): {record_key}")

        raw_response = ""
        mm_fields = default_mm_fields()
        summary_text = pack_summary_text(mm_fields)
        generation_status = "ok"

        try:
            wav_path, video_path = validate_media(item, project_root, split_name)
            prompt_bundle = build_multimodal_prompt(item, sample.history)

            raw_response = generator.generate(
                system_prompt=prompt_bundle["system_prompt"],
                user_prompt=prompt_bundle["user_prompt"],
                wav_path=wav_path,
                video_path=video_path,
            )

            mm_fields = normalize_multimodal_response(raw_response)
            summary_text = pack_summary_text(mm_fields)
            z_m = encoder.encode(summary_text)

        except Exception as e:
            generation_status = f"failed: {type(e).__name__}: {str(e)}"
            mm_fields = default_mm_fields()
            mm_fields["cross_modal_reasoning"] = "generation_failed"
            mm_fields["multimodal_summary"] = "generation_failed"
            summary_text = pack_summary_text(mm_fields)
            z_m = np.zeros((encoder.target_dim,), dtype=np.float32)
            wav_path = resolve_media_path(item["wav_path"], project_root)
            video_path = resolve_media_path(item["video_path"], project_root)
            print(f"[{split_name}] WARNING: {record_key} -> {generation_status}")

        save_vector(out_root, rel_vector_path, z_m)

        record = {
            "split": split_name,
            "dialogue_id": item["dialogue_id"],
            "utterance_id": item["utterance_id"],
            "record_key": record_key,
            "global_index": sample.global_index,
            "speaker": item["speaker"],
            "emotion": item["emotion"],
            "utterance_text": normalize_text(item["utterance_text"]),
            "history_text": normalize_text("\n".join([f"[Speaker: {h['speaker']}] {normalize_text(h['utterance_text'])}" for h in sample.history])),
            "wav_path": str(wav_path),
            "video_path": str(video_path),
            "raw_response": raw_response,
            "text_evidence": mm_fields["text_evidence"],
            "audio_evidence": mm_fields["audio_evidence"],
            "video_evidence": mm_fields["video_evidence"],
            "cross_modal_reasoning": mm_fields["cross_modal_reasoning"],
            "multimodal_summary": mm_fields["multimodal_summary"],
            "summary_text": summary_text,
            "vector_file": str(rel_vector_path),
            "vector_dim": int(z_m.shape[0]),
            "generation_status": generation_status,
        }
        append_jsonl(index_path, record)

        if sleep_per_sample > 0:
            time.sleep(sleep_per_sample)

    print(f"[{split_name}] done.")


# ==============================
# 十一、命令行参数
# ==============================


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    参数分组说明：
        A. 数据路径相关
        B. 输出目录相关
        C. 多模态模型后端相关
        D. summary encoder 相关
        E. 运行控制相关

    这样分组是为了后面你写新的 slurm 脚本时更清晰。
    """
    parser = argparse.ArgumentParser(description="Generate multimodal z_m vectors for MELD using Qwen2.5-Omni-7B.")

    parser.add_argument("--train_path", type=str, required=True, help="Path to train_meld_emo.csv")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to dev_meld_emo.csv")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test_meld_emo.csv")
    parser.add_argument("--project_root", type=str, default="..", help="Project root used to resolve relative wav/video paths")

    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for generated z_m files")
    parser.add_argument("--overwrite", action="store_true", help="If set, remove old out_dir content before running")
    parser.add_argument("--skip_existing", action="store_true", help="Skip records already present in index_{split}.jsonl and .npy")

    parser.add_argument("--backend", type=str, default="mock_mm", choices=["mock_mm", "qwen_omni_hf"], help="Multimodal generation backend")
    parser.add_argument("--omni_model_name", type=str, default="Qwen/Qwen2.5-Omni-7B", help="HF model name for Qwen2.5-Omni")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"], help="Model dtype")
    parser.add_argument("--device_map", type=str, default="auto", help="HF device_map")
    parser.add_argument("--attn_implementation", type=str, default=None, help="Optional attention implementation, e.g. flash_attention_2")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of generated tokens")
    parser.add_argument("--video_fps", type=float, default=1.0, help="FPS sampling for video input to Qwen Omni processor")
    parser.add_argument("--max_pixels", type=int, default=None, help="Optional max_pixels for processor to reduce video memory usage")
    parser.add_argument("--use_audio_in_video", action="store_true", help="If set, also load audio track from video")

    parser.add_argument("--summary_encoder_name", type=str, default="sentence-transformers/all-mpnet-base-v2", help="SentenceTransformer model used to encode multimodal summary text")
    parser.add_argument("--summary_encoder_device", type=str, default=None, help="Optional device for summary encoder, e.g. cuda / cpu")
    parser.add_argument("--target_dim", type=int, default=768, help="Final z_m dimension")

    parser.add_argument("--context_window", type=int, default=3, help="Number of previous utterances used as dialogue history")
    parser.add_argument("--limit_per_split", type=int, default=None, help="Optional cap on number of samples per split (useful for smoke test)")
    parser.add_argument("--start_index", type=int, default=0, help="Start processing from this flattened sample index within each split")
    parser.add_argument("--sleep_per_sample", type=float, default=0.0, help="Optional sleep time after each sample, mainly for debug / API-like pacing")
    parser.add_argument("--splits", type=str, default="train,dev,test", help="Comma-separated list of splits to run, e.g. train,dev or dev,test")

    args = parser.parse_args()
    return args


# ==============================
# 十二、主函数
# ==============================


def main() -> None:
    args = parse_args()

    print("========== generate_multimodal_zm.py ==========")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    out_root = Path(args.out_dir).resolve()
    project_root = Path(args.project_root).resolve()

    prepare_out_dir(out_root, overwrite=args.overwrite)

    generator = build_generator(args)

    encoder = SemanticSummaryEncoder(
        model_name=args.summary_encoder_name,
        target_dim=args.target_dim,
        device=args.summary_encoder_device,
    )

    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]
    valid_splits = {"train", "dev", "test"}
    for s in split_list:
        if s not in valid_splits:
            raise ValueError(f"Invalid split name: {s}. Expected subset of {sorted(valid_splits)}")

    for split_name in split_list:
        if split_name == "train":
            process_split(
                split_name="train",
                data_path=args.train_path,
                out_root=out_root,
                project_root=project_root,
                generator=generator,
                encoder=encoder,
                context_window=args.context_window,
                limit_per_split=args.limit_per_split,
                start_index=args.start_index,
                skip_existing=args.skip_existing,
                sleep_per_sample=args.sleep_per_sample,
            )
        elif split_name == "dev":
            process_split(
                split_name="dev",
                data_path=args.dev_path,
                out_root=out_root,
                project_root=project_root,
                generator=generator,
                encoder=encoder,
                context_window=args.context_window,
                limit_per_split=args.limit_per_split,
                start_index=args.start_index,
                skip_existing=args.skip_existing,
                sleep_per_sample=args.sleep_per_sample,
            )
        elif split_name == "test":
            process_split(
                split_name="test",
                data_path=args.test_path,
                out_root=out_root,
                project_root=project_root,
                generator=generator,
                encoder=encoder,
                context_window=args.context_window,
                limit_per_split=args.limit_per_split,
                start_index=args.start_index,
                skip_existing=args.skip_existing,
                sleep_per_sample=args.sleep_per_sample,
            )

    print("\nAll requested splits finished.")


if __name__ == "__main__":
    main()