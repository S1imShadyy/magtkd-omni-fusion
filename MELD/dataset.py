import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pickle


class MELD_Dataset(Dataset):
    def __init__(self, data):
        self.emoList = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        self.session_dataset = data

    def __len__(self):
        return len(self.session_dataset)

    def __getitem__(self, idx):
        return self.session_dataset[idx]


class AudioDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
        self.dialogue_ids = self.data['dialogue_id']
        self.utterance_ids = self.data['utterance_id']
        self.features = self.data['features']
        self.labels = self.data['label']
        self.emoList = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
        self.label_to_index = {label: idx for idx, label in enumerate(self.emoList)}

    def load_data(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        label_index = self.label_to_index[label]
        return [feature, label_index]


class IEMOCAP_Dataset(Dataset):
    def __init__(self, data_path):
        self.emoList = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
        data = pickle.load(open(data_path, 'rb'))
        self.text = data['text']
        self.audio = data['audio']
        self.video = data['video']
        self.audio_kd = data['audio_kd']
        self.video_kd = data['video_kd']
        self.speakers = data['speakers']
        self.labels = data['labels']
        self.vids = data['vids']
        self.dia2utt = data['dia2utt']
        self.len = len(self.vids)

    def __len__(self):
        return self.len

class MELD_MM_Dataset(Dataset):
    def __init__(self, data_path, zm_root=None, zm_dim=768):
        self.emoList = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        data = pickle.load(open(data_path, 'rb'))

        self.text = data['text']
        self.audio = data['audio']
        self.video = data['video']
        self.audio_kd = data['audio_kd']
        self.video_kd = data['video_kd']
        self.speakers = data['speakers']
        self.labels = data['labels']
        self.vids = data['vids']
        self.dia2utt = data['dia2utt']
        self.len = len(self.vids)

        self.z_m = data['z_m'] if 'z_m' in data else None
        self.zm_root = zm_root
        self.zm_dim = zm_dim

        lower_path = data_path.lower()
        if 'train' in lower_path:
            self.split = 'train'
        elif 'dev' in lower_path or 'valid' in lower_path:
            self.split = 'dev'
        elif 'test' in lower_path:
            self.split = 'test'
        else:
            self.split = None

    def __len__(self):
        return self.len

    def _load_external_zm_for_vid(self, vid):
        if self.zm_root is None or self.split is None:
            return None

        if vid not in self.dia2utt:
            return None

        utt_list = self.dia2utt[vid]
        seq_zm = []

        dialogue_id = str(vid)

        for idx, utt in enumerate(utt_list):
            utterance_id = str(utt)

            candidates = [
                os.path.join(self.zm_root, self.split, f"{self.split}_{dialogue_id}_{utterance_id}_zm.npy"),
                os.path.join(self.zm_root, self.split, f"{dialogue_id}_{utterance_id}.npy"),
            ]

            found = None
            for path in candidates:
                if os.path.exists(path):
                    z = np.load(path).astype(np.float32)
                    if z.ndim != 1 or z.shape[0] != self.zm_dim:
                        raise ValueError(
                            f"Bad z_m shape at {path}: got {z.shape}, expected ({self.zm_dim},)"
                        )
                    found = z
                    break

            if found is not None:
                seq_zm.append(found)
            else:
                if self.z_m is not None:
                    old_z = np.array(self.z_m[vid][idx], dtype=np.float32)
                    seq_zm.append(old_z)
                else:
                    seq_zm.append(np.zeros(self.zm_dim, dtype=np.float32))
                    print(f"[warn] fallback to zero z_m for vid={vid}, utt={utt}")

        return np.stack(seq_zm, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        vid = self.vids[idx]

        z_m_arr = self._load_external_zm_for_vid(vid)

        if z_m_arr is None:
            if self.z_m is not None:
                z_m_arr = np.array(self.z_m[vid], dtype=np.float32)
            else:
                seq_len = len(self.labels[vid])
                z_m_arr = np.zeros((seq_len, self.zm_dim), dtype=np.float32)
                print(f"[warn] fallback to zero z_m for vid={vid}")

        return (
            torch.FloatTensor(np.array(self.text[vid])),
            torch.FloatTensor(np.array(self.audio[vid])),
            torch.FloatTensor(np.array(self.video[vid])),
            torch.FloatTensor(np.array(self.audio_kd[vid])),
            torch.FloatTensor(np.array(self.video_kd[vid])),
            torch.FloatTensor(z_m_arr),
            torch.FloatTensor(np.array(self.speakers[vid])),
            torch.FloatTensor([1] * len(self.labels[vid])),
            torch.LongTensor(self.labels[vid]),
            vid
        )

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [
            pad_sequence(dat[i]) if i < 6
            else pad_sequence(dat[i], True) if i < 9
            else dat[i].tolist()
            for i in dat
        ]

    def __len__(self):
        return self.len

def _load_external_zm_for_vid(self, vid):
    if self.zm_root is None or self.split is None:
        return None

    if vid not in self.dia2utt:
        return None

    utt_list = self.dia2utt[vid]
    seq_zm = []

    # 现在 vid 本身更可能就是 dialogue_id
    dialogue_id = str(vid)

    for utt in utt_list:
        # 当前数据结构里 utt 是 utterance_id（int）
        utterance_id = str(utt)

        candidates = [
            os.path.join(self.zm_root, self.split, f"{self.split}_{dialogue_id}_{utterance_id}_zm.npy"),
            os.path.join(self.zm_root, self.split, f"{dialogue_id}_{utterance_id}.npy"),
        ]

        found = None
        for path in candidates:
            if os.path.exists(path):
                z = np.load(path).astype(np.float32)
                if z.ndim != 1 or z.shape[0] != self.zm_dim:
                    raise ValueError(
                        f"Bad z_m shape at {path}: got {z.shape}, expected ({self.zm_dim},)"
                    )
                found = z
                break

        if found is None:
            print(f"[warn] missing external z_m for vid={vid}, utt={utt}")
            return None

        seq_zm.append(found)

    return np.stack(seq_zm, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        vid = self.vids[idx]

        # 优先读取新的外部 z_m
        z_m_arr = self._load_external_zm_for_vid(vid)

        # 如果新 z_m 不完整或没找到，则回退到旧 pickle 内的 z_m
        if z_m_arr is None:
            if self.z_m is not None:
                z_m_arr = np.array(self.z_m[vid], dtype=np.float32)
            else:
                # 再兜底：全零
                seq_len = len(self.labels[vid])
                z_m_arr = np.zeros((seq_len, self.zm_dim), dtype=np.float32)
                print(f"[warn] fallback to zero z_m for vid={vid}")

        return torch.FloatTensor(np.array(self.text[vid])), \
               torch.FloatTensor(np.array(self.audio[vid])), \
               torch.FloatTensor(np.array(self.video[vid])), \
               torch.FloatTensor(np.array(self.audio_kd[vid])), \
               torch.FloatTensor(np.array(self.video_kd[vid])), \
               torch.FloatTensor(z_m_arr), \
               torch.FloatTensor(np.array(self.speakers[vid])), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               vid

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 6 else pad_sequence(dat[i], True) if i < 9 else dat[i].tolist() for i in dat]