#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_multimodal_zm.py

用途：
    为 MAGTKD / MELD 项目生成“真正多模态来源”的 z_m 向量。

核心思路：
    1. 读取 MELD 的 train / dev / test split。
    2. 对每条 utterance 收集：
       - 文本（utterance text）
       - 音频（wav）
       - 视频（video）
       - 对话历史（history）
    3. 调用 Qwen2.5-Omni-7B 生成结构化多模态语义结果：
       - text_evidence
       - audio_evidence
       - video_evidence
       - cross_modal_reasoning
       - multimodal_summary
    4. 将这些结构化语义字段拼接成 summary_text。
    5. 再用 sentence-transformer 把 summary_text 编码为固定维度向量 z_m。
    6. 保存为：
       - 每条样本一个 .npy 向量文件
       - 每个 split 一个 index_{split}.jsonl 索引文件

为什么这样设计：
    - 你的 thesis 新方向是“MLLM 参与多模态 fusion”，而不是 text-only augmentation。
    - 因此 z_m 的信息来源必须来自 text + audio + video。
    - 但为了工程稳定、便于检查和论文书写，我们不让模型直接吐 latent vector，
      而是先吐结构化语义，再转成固定维 embedding。

兼容性：
    - 输出格式故意尽量兼容你现在 multimodal_fusion.py 的读取方式：
      仍然是逐样本 .npy + index_{split}.jsonl。
    - 这样 fusion 侧只需要小改，或者在已有 zm_root 读取逻辑下直接复用。

官方实现依据：
    - 本脚本的 Qwen2.5-Omni 输入组织方式，参考了 Hugging Face 官方文档：
      Qwen2.5-Omni 支持 text / audio / video 输入；processor.apply_chat_template()
      可以组织带多模态内容的对话输入。
"""

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

# 这里沿用你原项目中的 preprocessing，保证读取 split 的逻辑与现有项目一致。
from preprocessing import preprocessing


# ==============================
# 一、基础工具函数
# ==============================


def ensure_dir(path: Path) -> None:
    """
    确保目录存在。

    说明：
        很多输出目录（例如 out_dir/train、out_dir/dev）在第一次运行前并不存在。
        这个函数统一负责创建目录，避免每次写文件前都重复判断。
    """
    path.mkdir(parents=True, exist_ok=True)



def normalize_text(text: Optional[str]) -> str:
    """
    对文本做轻量清洗，便于写入 prompt / jsonl。

    处理内容：
        - None -> 空字符串
        - 去掉首尾空白
        - 将连续空白字符压缩成单个空格
        - 将换行改为空格，避免 prompt 被破坏
    """
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text



def make_record_key(split: str, dialogue_id: str, utterance_id: str) -> str:
    """
    为每条 utterance 构造唯一键。

    设计原因：
        fusion 阶段、debug 阶段、断点续跑阶段，都需要一个稳定的唯一 key。
        这里采用：split::dialogue_id::utterance_id

    例如：
        train::23::5
    """
    return f"{split}::{dialogue_id}::{utterance_id}"



def prepare_out_dir(out_root: Path, overwrite: bool = False) -> None:
    """
    准备输出目录。

    参数：
        out_root: 输出根目录
        overwrite: 是否允许清空旧结果

    设计说明：
        旧脚本里是“输出目录必须完全干净”，这样安全但不利于断点续跑。
        新脚本改成：
        - overwrite=False: 保留已有结果，支持续跑
        - overwrite=True : 删除旧目录后重建
    """
    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)
    ensure_dir(out_root)
    ensure_dir(out_root / "train")
    ensure_dir(out_root / "dev")
    ensure_dir(out_root / "test")



def append_jsonl(path: Path, record: Dict) -> None:
    """
    将一条记录追加写入 jsonl 文件。

    为什么必须逐条 append：
        你后面跑全量 split 时，多模态模型推理可能中途失败或作业被中断。
        如果最后才一次性写文件，那么前面已经完成的结果会全部丢失。
        逐条 append 可以最大程度保护已生成结果。
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")



def load_existing_record_keys(index_path: Path) -> set[str]:
    """
    从已有 jsonl 索引文件中读取已完成样本的 key。

    用途：
        - 支持 --skip_existing
        - 避免重复生成已完成样本
    """
    keys: set[str] = set()
    if not index_path.exists():
        return keys

    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "record_key" in obj:
                    keys.add(obj["record_key"])
            except Exception:
                # 某一行损坏时，不让整个流程崩掉。
                continue
    return keys


# ==============================
# 二、数据结构定义
# ==============================


@dataclass
class FlattenedSample:
    """
    将原始 dialogue 结构展开后的单条样本。

    字段说明：
        global_index:
            当前 split 内部的扁平索引，便于 start_index / limit_per_split 控制。
        item:
            当前 utterance 的原始信息字典。
        history:
            当前 utterance 之前的若干条历史上下文。
    """

    global_index: int
    item: Dict
    history: List[Dict]


# ==============================
# 三、媒体路径解析
# ==============================


def resolve_media_path(path_str: str, project_root: Path) -> Path:
    """
    将媒体路径解析为绝对路径。

    设计原因：
        你的 preprocessing 输出里，有些路径可能是相对路径，有些可能已经是绝对路径。
        为了避免后面模型加载媒体时混乱，这里统一解析成绝对路径。

    参数：
        path_str: csv / preprocessing 中给出的 wav_path 或 video_path
        project_root: 项目根目录，用于解析相对路径
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()



def validate_media(item: Dict, project_root: Path) -> Tuple[Path, Path]:
    """
    检查当前样本的 wav 和 video 路径是否存在。

    返回：
        (wav_abs_path, video_abs_path)

    为什么要单独写这个函数：
        媒体文件缺失是多模态生成阶段最常见的报错来源之一。
        提前做显式检查，能让错误更早、更清楚地暴露出来。
    """
    wav_path = resolve_media_path(item["wav_path"], project_root)
    video_path = resolve_media_path(item["video_path"], project_root)

    if not wav_path.exists():
        raise FileNotFoundError(f"Missing wav file: {wav_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video file: {video_path}")

    return wav_path, video_path


# ==============================
# 四、读取并展开 MELD split
# ==============================


def load_split_dialogues(data_path: str, split_type: str) -> List[List[Dict]]:
    """
    读取某个 split 的对话，并保持原始 utterance 顺序。

    返回格式：
        List[List[Dict]]
        外层 list = 一个个 dialogue
        内层 list = 该 dialogue 中按顺序排列的 utterance

    这里沿用你旧脚本的结构，方便平滑迁移。
    """
    session_dataset = preprocessing(data_path, split_type)
    dialogues: List[List[Dict]] = []

    for dialogue in session_dataset:
        utterances: List[Dict] = []
        for item in dialogue:
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, split_name, dialogue_id, utterance_id = item
            utterances.append(
                {
                    "speaker": speaker,
                    "utterance_text": utt,
                    "wav_path": wav_path,
                    "video_path": video_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "emotion": emotion,
                    "split": split_name,
                    "dialogue_id": str(dialogue_id),
                    "utterance_id": str(utterance_id),
                }
            )
        dialogues.append(utterances)

    return dialogues



def flatten_dialogues(dialogues: List[List[Dict]], context_window: int) -> List[FlattenedSample]:
    """
    将 dialogues 展平为逐样本列表，同时为每条样本附带历史上下文。

    参数：
        dialogues: load_split_dialogues() 的结果
        context_window: 取当前 utterance 前多少条历史 utterances

    为什么要 flatten：
        原始 dialogue 嵌套结构不方便做：
        - start_index
        - limit_per_split
        - 进度条式处理
        - 断点续跑
        所以这里统一展平。
    """
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
    """
    为当前样本构造多模态 prompt。

    返回：
        {
            "system_prompt": ...,
            "user_prompt": ...,
            "history_text": ...,
        }

    设计思路：
        这里的 prompt 不要求模型直接预测 emotion label。
        我们要的是“可观察证据驱动”的多模态语义描述，方便后续作为 fusion 的语义指导信号。

    输出字段固定为：
        text_evidence
        audio_evidence
        video_evidence
        cross_modal_reasoning
        multimodal_summary

    为什么这样比直接输出标签更好：
        - 更符合 thesis 的 multimodal semantic alignment 方向
        - 更容易检查模型有没有真的用到音频和视频
        - 更利于论文方法部分写清楚
    """
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
    """
    返回默认的多模态结构化字段。

    用途：
        - 模型输出不规范时兜底
        - 媒体文件缺失时兜底
        - 推理异常时兜底
    """
    return {
        "text_evidence": "unavailable",
        "audio_evidence": "unavailable",
        "video_evidence": "unavailable",
        "cross_modal_reasoning": "unavailable",
        "multimodal_summary": "unavailable",
    }



def normalize_multimodal_response(raw_response: str) -> Dict[str, str]:
    """
    将模型原始输出解析为结构化字段。

    目标格式：
        text_evidence: ...
        audio_evidence: ...
        video_evidence: ...
        cross_modal_reasoning: ...
        multimodal_summary: ...

    为什么要单独做解析函数：
        实际大模型输出可能并不总是完全服从格式要求。
        这里集中处理各种边界情况，保证后续编码阶段总能拿到完整字段。
    """
    fields = default_mm_fields()

    text = (raw_response or "").strip()
    if not text:
        return fields

    # 用“字段名:”作为切分标志。
    # DOTALL 允许字段内容跨行；IGNORECASE 提高兼容性。
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

    # 如果完全没匹配到结构化字段，则退化为把整个文本放进 multimodal_summary。
    # 这样至少不会丢掉模型给出的内容。
    if not matched_any:
        fields["multimodal_summary"] = normalize_text(text)
        fields["cross_modal_reasoning"] = "Response did not follow the requested structure; raw response stored as summary."

    return fields



def pack_summary_text(mm_fields: Dict[str, str]) -> str:
    """
    将结构化字段重新打包成统一文本。

    作用：
        SentenceTransformer 编码时需要一个连续字符串。
        这里把 5 个字段拼成一个稳定、可重复的 summary_text。

    注意：
        这一步虽然最终用的是 text encoder，
        但 summary_text 的信息来源已经是“看过 text + audio + video 后生成的结构化结果”，
        因此它仍然是 multimodal 来源的 z_m。
    """
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
    """
    一个假的多模态生成器，用于 smoke test。

    用途：
        当你还没有真正把 Qwen2.5-Omni 跑起来时，先用 mock 模式验证整个工程链路：
        - 读 split
        - 构造 prompt
        - 生成 summary_text
        - 编码 z_m
        - 写 index/jsonl
        - 写 .npy

    这样可以先把“工程骨架”跑通，再上真正的多模态模型。
    """

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
    """
    Qwen2.5-Omni-7B 本地 Hugging Face 推理后端。

    设计重点：
        - 使用 Qwen2_5OmniThinkerForConditionalGeneration 只生成文本结果，
          不开启语音输出，从而减少显存和复杂度。
        - 使用 Qwen2_5OmniProcessor 组织 text + audio + video 输入。
        - 支持控制视频 fps、最大分辨率、是否使用视频内音频。

    为什么选 Thinker 而不是完整 Talker：
        你当前任务只需要模型输出结构化文本，然后再编码成 z_m。
        因此没有必要开启音频生成部分，省显存、省时间，也更适合集群部署。
    """

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

        # 延迟导入：
        # 只有在真正使用 qwen_omni_hf backend 时才要求 transformers 版本足够新。
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

        # 官方文档提示 max_pixels 过大可能导致视频 OOM。
        if max_pixels is not None:
            # processor.max_pixels 在部分版本下可直接设置。
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
        """
        构造多模态对话输入。

        输入内容组织原则：
            - system 中放任务角色与约束
            - user 中同时放：
              1) 当前视频
              2) 当前音频
              3) 文本 prompt

        为什么既传 video 又传 audio：
            你的 MELD 任务里音频和视频都来自同一 utterance，但它们在项目数据流中是独立文件。
            为了让模型明确使用两种模态，这里显式同时提供。

        关于 use_audio_in_video：
            - 若视频本身已经有音轨，并且你希望直接从视频中提取音频，可设为 True。
            - 但你的项目已经有单独 wav，因此默认 False，避免把同一段音频重复喂两次。
        """
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
        """
        执行一次真正的 Qwen2.5-Omni 推理，并返回模型生成的文本结果。

        关键步骤：
            1. 构造带 text/audio/video 的 conversation。
            2. 交给 processor.apply_chat_template() 组织成模型输入。
            3. 调用 thinker model.generate()。
            4. 仅解码新增生成的 token，避免把 prompt 一起解码回来。
        """
        conversations = self._build_conversation(system_prompt, user_prompt, wav_path, video_path)

        processor_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
            "fps": self.fps,
            # 由于我们已经单独提供 wav，这里默认不再从视频中抽音频，避免重复。
            "load_audio_from_video": self.use_audio_in_video,
            "use_audio_in_video": self.use_audio_in_video,
        }

        inputs = self.processor.apply_chat_template(
            conversations,
            **processor_kwargs,
        )

        # 将输入移动到模型所在设备。
        # device_map="auto" 时，model.device 仍可作为主设备入口使用。
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_audio_in_video=self.use_audio_in_video,
        )

        # 只截取“新生成”的部分。
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
    """
    根据命令行参数构建实际使用的多模态生成器。

    当前支持：
        - mock_mm
        - qwen_omni_hf

    设计成工厂函数的原因：
        后续如果你还想增加其他后端（例如 API 版、量化版、本地服务版），
        只需要在这里扩展即可。
    """
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
    """
    将结构化 multimodal summary 编码为固定维度 z_m 的编码器。

    工作流程：
        1. 使用 sentence-transformer 得到一个原始向量。
        2. 若原始维度 == target_dim，直接返回。
        3. 若原始维度 > target_dim，直接截断。
        4. 若原始维度 < target_dim，末尾补零。

    为什么不在这里再训练一个额外投影层：
        你当前优先目标是把“多模态 z_m 生成-接入-fusion”这条链跑通。
        先用简单、稳定、可复现的方式固定维度，后面如果效果好，再考虑更复杂投影。
    """

    def __init__(self, model_name: str, target_dim: int = 768, device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.target_dim = target_dim
        self.device = device

        # sentence-transformers 内部会自行处理 device；这里不强依赖 GPU。
        self.model = SentenceTransformer(model_name, device=device)

    def _fit_dim(self, emb: np.ndarray) -> np.ndarray:
        """
        将原始 embedding 调整为固定维度。
        """
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
        """
        将 summary_text 编码为固定维度 np.ndarray。
        """
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
    """
    生成某条样本的向量相对路径。

    例如：
        train/train_12_4_zm.npy
    """
    return Path(split_name) / f"{split_name}_{dialogue_id}_{utterance_id}_zm.npy"



def save_vector(out_root: Path, relpath: Path, vector: np.ndarray) -> None:
    """
    保存 z_m 向量到 .npy 文件。

    使用 np.save 的原因：
        - 与你现有 fusion 读取习惯一致
        - 简单直观
        - 调试方便
    """
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
    """
    处理某一个 split（train / dev / test）。

    这是整个脚本最核心的函数。

    完整流程：
        1. 读取 split 对话
        2. 展平为逐样本列表
        3. 根据 start_index / limit_per_split 截取处理范围
        4. 对每条样本：
           - 检查 wav/video 路径
           - 构造 prompt
           - 调用多模态模型生成结构化结果
           - 解析字段
           - 打包 summary_text
           - 编码成 z_m
           - 保存 .npy
           - 记录到 index_{split}.jsonl
        5. 任意单条样本失败时，使用零向量 fallback，不让整个 split 报废

    为什么这个函数设计得这么“啰嗦”：
        因为你后面跑的不是 5 条 smoke，而是作者原实验同量级 split。
        一旦某条媒体有问题、模型偶发失败、显存有瞬时波动，都不能让整批作业直接作废。
    """
    print(f"\n========== Processing split: {split_name} ==========")
    index_path = out_root / f"index_{split_name}.jsonl"
    existing_keys = load_existing_record_keys(index_path) if skip_existing else set()

    dialogues = load_split_dialogues(data_path, split_name)
    samples = flatten_dialogues(dialogues, context_window=context_window)

    # 先按 start_index 截断，再按 limit_per_split 限制样本数。
    samples = [s for s in samples if s.global_index >= start_index]
    if limit_per_split is not None:
        samples = samples[:limit_per_split]

    print(f"[{split_name}] total selected samples: {len(samples)}")

    for idx, sample in enumerate(samples, start=1):
        item = sample.item
        record_key = make_record_key(split_name, item["dialogue_id"], item["utterance_id"])
        rel_vector_path = make_vector_relpath(split_name, item["dialogue_id"], item["utterance_id"])
        abs_vector_path = out_root / rel_vector_path

        if skip_existing and record_key in existing_keys and abs_vector_path.exists():
            print(f"[{split_name}] skip existing ({idx}/{len(samples)}): {record_key}")
            continue

        print(f"[{split_name}] processing ({idx}/{len(samples)}): {record_key}")

        raw_response = ""
        mm_fields = default_mm_fields()
        summary_text = pack_summary_text(mm_fields)
        generation_status = "ok"

        try:
            wav_path, video_path = validate_media(item, project_root)
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
            # 任意单条失败，都不让整个 split 崩掉。
            # 这里统一：
            #   - 结构化字段用默认值
            #   - z_m 用零向量
            #   - generation_status 记录错误信息
            generation_status = f"failed: {type(e).__name__}: {str(e)}"
            mm_fields = default_mm_fields()
            mm_fields["cross_modal_reasoning"] = "generation_failed"
            mm_fields["multimodal_summary"] = "generation_failed"
            summary_text = pack_summary_text(mm_fields)
            z_m = np.zeros((encoder.target_dim,), dtype=np.float32)
            wav_path = resolve_media_path(item["wav_path"], project_root)
            video_path = resolve_media_path(item["video_path"], project_root)
            print(f"[{split_name}] WARNING: {record_key} -> {generation_status}")

        # 保存向量文件
        save_vector(out_root, rel_vector_path, z_m)

        # 逐条写索引，确保作业中途停止时已完成样本不会丢失。
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

    # ===== A. 数据路径 =====
    parser.add_argument("--train_path", type=str, required=True, help="Path to train_meld_emo.csv")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to dev_meld_emo.csv")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test_meld_emo.csv")
    parser.add_argument("--project_root", type=str, default="..", help="Project root used to resolve relative wav/video paths")

    # ===== B. 输出目录 =====
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for generated z_m files")
    parser.add_argument("--overwrite", action="store_true", help="If set, remove old out_dir content before running")
    parser.add_argument("--skip_existing", action="store_true", help="Skip records already present in index_{split}.jsonl and .npy")

    # ===== C. 后端相关 =====
    parser.add_argument("--backend", type=str, default="mock_mm", choices=["mock_mm", "qwen_omni_hf"], help="Multimodal generation backend")
    parser.add_argument("--omni_model_name", type=str, default="Qwen/Qwen2.5-Omni-7B", help="HF model name for Qwen2.5-Omni")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"], help="Model dtype")
    parser.add_argument("--device_map", type=str, default="auto", help="HF device_map")
    parser.add_argument("--attn_implementation", type=str, default=None, help="Optional attention implementation, e.g. flash_attention_2")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of generated tokens")
    parser.add_argument("--video_fps", type=float, default=1.0, help="FPS sampling for video input to Qwen Omni processor")
    parser.add_argument("--max_pixels", type=int, default=None, help="Optional max_pixels for processor to reduce video memory usage")
    parser.add_argument("--use_audio_in_video", action="store_true", help="If set, also load audio track from video")

    # ===== D. summary 编码器 =====
    parser.add_argument("--summary_encoder_name", type=str, default="sentence-transformers/all-mpnet-base-v2", help="SentenceTransformer model used to encode multimodal summary text")
    parser.add_argument("--summary_encoder_device", type=str, default=None, help="Optional device for summary encoder, e.g. cuda / cpu")
    parser.add_argument("--target_dim", type=int, default=768, help="Final z_m dimension")

    # ===== E. 运行控制 =====
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
    """
    主函数：组织整个 z_m 生成流程。

    main 的职责：
        1. 解析命令行参数
        2. 准备输出目录
        3. 构建多模态生成器（mock 或 Qwen Omni）
        4. 构建 summary encoder
        5. 依次处理指定 split

    为什么 main 不直接写太多逻辑：
        为了后续你自己读代码时更清晰：
        - main 只做“总调度”
        - 具体细节都拆到函数 / 类里
        这样你以后想改 prompt、换模型、改输出格式，都更容易定位。
    """
    args = parse_args()

    print("========== generate_multimodal_zm.py ==========")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    out_root = Path(args.out_dir).resolve()
    project_root = Path(args.project_root).resolve()

    prepare_out_dir(out_root, overwrite=args.overwrite)

    # 1) 构建多模态生成器
    generator = build_generator(args)

    # 2) 构建 summary -> z_m 编码器
    encoder = SemanticSummaryEncoder(
        model_name=args.summary_encoder_name,
        target_dim=args.target_dim,
        device=args.summary_encoder_device,
    )

    # 3) 决定本次运行哪些 split
    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]
    valid_splits = {"train", "dev", "test"}
    for s in split_list:
        if s not in valid_splits:
            raise ValueError(f"Invalid split name: {s}. Expected subset of {sorted(valid_splits)}")

    # 4) 依次处理每个 split
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
