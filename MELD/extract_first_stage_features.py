import gc
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from preprocessing import *
from dataset import *
from utils import *
from model import *


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_omni_zm(split_type: str, dialogue_id: int, utterance_id: int, omni_root: str = "./omni_zm") -> np.ndarray:
    zm_path = Path(omni_root) / split_type / f"{dialogue_id}_{utterance_id}.npy"
    if not zm_path.exists():
        raise FileNotFoundError(f"Missing omni z_m file: {zm_path}")
    zm = np.load(zm_path)
    return zm.astype(np.float32)


def extract_features(model_t, audio_s, video_s, dataloader, save_path):
    model_t.eval()
    audio_s.eval()
    video_s.eval()

    all_text_hidden = []
    all_audio_hidden = []
    all_video_hidden = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens = batch_input_tokens.cuda()
            attention_masks = attention_masks.cuda()
            audio_inputs = audio_inputs.cuda()
            video_inputs = video_inputs.cuda()
            batch_labels = batch_labels.cuda()

            text_hidden, _ = model_t(batch_input_tokens, attention_masks)
            audio_hidden, _ = audio_s(audio_inputs)
            video_hidden, _ = video_s(video_inputs)

            all_text_hidden.append(text_hidden.cpu())
            all_audio_hidden.append(audio_hidden.cpu())
            all_video_hidden.append(video_hidden.cpu())
            all_labels.append(batch_labels.cpu())

    all_text_hidden = torch.cat(all_text_hidden, dim=0)
    all_audio_hidden = torch.cat(all_audio_hidden, dim=0)
    all_video_hidden = torch.cat(all_video_hidden, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    features_dict = {
        'text_hidden': all_text_hidden,
        'audio_hidden': all_audio_hidden,
        'video_hidden': all_video_hidden,
        'labels': all_labels
    }

    with open(save_path, 'wb') as f:
        pickle.dump(features_dict, f)

    print(f"Features saved to {save_path}")


def extract_all_features(
    model_t,
    model_s_a,
    model_s_v,
    model_s_a_KD,
    model_s_v_KD,
    data_loader,
    save_path,
    split_type,
    omni_root="./omni_zm",
):
    model_t.eval()
    model_s_a.eval()
    model_s_v.eval()
    model_s_a_KD.eval()
    model_s_v_KD.eval()

    vids = []
    dia2utt = {}
    text = {}
    audio = {}
    video = {}
    audio_kd = {}
    video_kd = {}
    z_m = {}
    speakers = {}
    labels = {}

    with torch.no_grad():
        for batch in tqdm(data_loader):
            (
                batch_input_tokens,
                attention_masks,
                audio_inputs,
                video_inputs,
                batch_labels,
                batch_speakers,
                batch_dia_ids,
                batch_utt_ids,
            ) = batch

            batch_input_tokens = batch_input_tokens.cuda()
            attention_masks = attention_masks.cuda()
            audio_inputs = audio_inputs.cuda()
            video_inputs = video_inputs.cuda()
            batch_labels = batch_labels.cuda()
            batch_speakers = batch_speakers.cuda()

            text_hidden, _ = model_t(batch_input_tokens, attention_masks)
            audio_hidden, _ = model_s_a(audio_inputs)
            video_hidden, _ = model_s_v(video_inputs)
            audio_kd_hidden, _ = model_s_a_KD(audio_inputs)
            video_kd_hidden, _ = model_s_v_KD(video_inputs)

            batch_size = batch_input_tokens.shape[0]

            for i in range(batch_size):
                vid = int(batch_dia_ids[i].item())
                uttid = int(batch_utt_ids[i].item())

                if vid not in vids:
                    vids.append(vid)
                    dia2utt[vid] = []
                    text[vid] = []
                    audio[vid] = []
                    video[vid] = []
                    audio_kd[vid] = []
                    video_kd[vid] = []
                    z_m[vid] = []
                    speakers[vid] = []
                    labels[vid] = []

                dia2utt[vid].append(uttid)
                text[vid].append(text_hidden[i].detach().cpu().numpy())
                audio[vid].append(audio_hidden[i].detach().cpu().numpy())
                video[vid].append(video_hidden[i].detach().cpu().numpy())
                audio_kd[vid].append(audio_kd_hidden[i].detach().cpu().numpy())
                video_kd[vid].append(video_kd_hidden[i].detach().cpu().numpy())
                z_m[vid].append(load_omni_zm(split_type, vid, uttid, omni_root=omni_root))
                speakers[vid].append(batch_speakers[i].item())
                labels[vid].append(batch_labels[i].item())

    features_dict = {
        'text': text,
        'audio': audio,
        'video': video,
        'audio_kd': audio_kd,
        'video_kd': video_kd,
        'z_m': z_m,
        'speakers': speakers,
        'labels': labels,
        'vids': vids,
        'dia2utt': dia2utt
    }

    with open(save_path, 'wb') as f:
        pickle.dump(features_dict, f)

    print(f"Features saved to {save_path}")


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--epochs', type=int, default=10, help='epoch for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate for training.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training.')
    parser.add_argument('--seed', type=int, default=2024, help='random seed for training.')
    parser.add_argument('--train', type=bool, default=True, help='whether to train the model.')
    parser.add_argument('--teacher', type=str, default='text', help='who is teacher?')
    parser.add_argument('--student', type=str, default='text', help='who is student?')
    parser.add_argument('--advanced', type=bool, default=False, help='whether to use advanced loss.')
    parser.add_argument('--omni_root', type=str, default='./omni_zm', help='directory containing generated omni z_m files.')
    args = parser.parse_args()

    seed_everything(args.seed)

    @dataclass
    class Config():
        mask_time_length: int = 3

    text_model = "roberta-large"
    audio_model = "data2vec-audio-base-960h"
    video_model = "timesformer-base-finetuned-k400"

    data_path = "../datasets/MELD"
    train_path = os.path.join(data_path, "train_meld_emo.csv")
    dev_path = os.path.join(data_path, "dev_meld_emo.csv")
    test_path = os.path.join(data_path, "test_meld_emo.csv")

    train_dataset = MELD_Dataset(preprocessing(train_path, split_type='train'))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=SequentialSampler(train_dataset), num_workers=16, collate_fn=all_features_batchs)

    dev_dataset = MELD_Dataset(preprocessing(dev_path, split_type='dev'))
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, sampler=SequentialSampler(dev_dataset), num_workers=16, collate_fn=all_features_batchs)

    test_dataset = MELD_Dataset(preprocessing(test_path, split_type='test'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=SequentialSampler(test_dataset), num_workers=16, collate_fn=all_features_batchs)

    save_model_path = './save_model'
    print("###Save Path### ", save_model_path)

    clsNum = len(train_dataset.emoList)
    init_config = Config()

    model_t = Text_model(text_model, clsNum)
    model_t.load_state_dict(torch.load(os.path.join(save_model_path, 'text.bin')))

    model_s_a = Audio_model(audio_model, clsNum, init_config)
    _ckpt = os.path.join(save_model_path, 'audio.bin')
    if os.path.exists(_ckpt):
        print(f"[RESUME] loading {_ckpt}")
        model_s_a.load_state_dict(torch.load(_ckpt))
    else:
        print(f"[RESUME] skip: {_ckpt} not found, start fresh")

    model_s_v = Video_model(video_model, clsNum)
    model_s_v.load_state_dict(torch.load(os.path.join(save_model_path, 'video.bin')))

    model_s_a_KD = Audio_model(audio_model, clsNum, init_config)
    model_s_a_KD.load_state_dict(torch.load(os.path.join(save_model_path, 'text_KD_audio.bin')))

    model_s_v_KD = Video_model(video_model, clsNum)
    model_s_v_KD.load_state_dict(torch.load(os.path.join(save_model_path, 'text_KD_video.bin')))

    for para in model_t.parameters():
        para.requires_grad = False
    for para in model_s_a.parameters():
        para.requires_grad = False
    for para in model_s_v.parameters():
        para.requires_grad = False
    for para in model_s_a_KD.parameters():
        para.requires_grad = False
    for para in model_s_v_KD.parameters():
        para.requires_grad = False

    model_t = model_t.cuda().eval()
    model_s_a = model_s_a.cuda().eval()
    model_s_v = model_s_v.cuda().eval()
    model_s_a_KD = model_s_a_KD.cuda().eval()
    model_s_v_KD = model_s_v_KD.cuda().eval()

    save_path = "feature/first_stage_train_features.pkl"
    extract_all_features(model_t, model_s_a, model_s_v, model_s_a_KD, model_s_v_KD, train_loader, save_path, split_type='train', omni_root=args.omni_root)

    save_path = "feature/first_stage_dev_features.pkl"
    extract_all_features(model_t, model_s_a, model_s_v, model_s_a_KD, model_s_v_KD, dev_loader, save_path, split_type='dev', omni_root=args.omni_root)

    save_path = "feature/first_stage_test_features.pkl"
    extract_all_features(model_t, model_s_a, model_s_v, model_s_a_KD, model_s_v_KD, test_loader, save_path, split_type='test', omni_root=args.omni_root)

    print("---------------Done--------------")
