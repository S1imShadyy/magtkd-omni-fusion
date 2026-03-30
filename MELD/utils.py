import torch
from transformers import RobertaTokenizer, RobertaModel, AutoFeatureExtractor, AutoImageProcessor
import librosa
import cv2
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")

audio_processor = AutoFeatureExtractor.from_pretrained("/data/asun661/models/facebook/data2vec-audio-base-960h")
roberta_tokenizer = RobertaTokenizer.from_pretrained('/data/asun661/models/roberta-large')
speaker_list = ['<s1>', '<s2>', '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']


def generate_silence(duration_seconds, sample_rate=44100):
    """
    生成指定时长的空白音频片段。
    """
    num_samples = int(duration_seconds * sample_rate)
    silence = np.ones(num_samples)
    return silence, sample_rate

def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]
    ids = tokenizer.convert_tokens_to_ids(truncated)
    return ids + [tokenizer.mask_token_id]

def padding(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
    pad_ids = []
    attention_masks = []
    for ids in ids_list:
        pad_len = max_len - len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        attention_mask = [1 for _ in range(len(ids))]
        add_attention = [0 for _ in range(len(add_ids))]
        pad_ids.append(add_ids + ids)
        attention_masks.append(add_attention + attention_mask)
    return torch.tensor(pad_ids), torch.tensor(attention_masks)

def padding_audio_or_video(batch):
    max_len = 0
    for ids in batch:
        if len(ids) > max_len:
            max_len = len(ids)
    pad_ids = []
    for ids in batch:
        pad_len = max_len-len(ids)
        add_ids = [ 0 for _ in range(pad_len)]
        pad_ids.append(add_ids+ids.tolist())
    return torch.tensor(pad_ids)

def get_audio(processor, path):
    try:
        audio, rate = librosa.load(path)
    except Exception as e:
        audio, rate = generate_silence(5)
        print(path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    return inputs["input_values"][0]

def find_video(dataset_split_type, sess, uttid, base_dir="./feature/video"):
    dataset_dir = os.path.join(base_dir, dataset_split_type)
    file_name = f"{sess}_{uttid}.npy"
    file_path = os.path.join(dataset_dir, file_name)
    if os.path.exists(file_path):
        return file_path
    else:
        print(f"File not found: {file_path}")
        return None

def text_batchs(sessions):
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_text, batch_audio, batch_video, batch_labels = [], [], [], []
    for session in sessions:
        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, split_type, sess, uttid = line
            inputString += '<s' + str(speaker+1) + '> ' + utt + ' '
            now_speaker = speaker
        prompt = "Now" + '<s' + str(now_speaker+1) + '> '+"feels"
        concat_string = inputString.strip() + " " + "</s>" + " " + prompt
        batch_text.append(encode_right_truncated(concat_string, roberta_tokenizer))
        label_index = label_list.index(emotion)
        batch_labels.append(label_index)
    batch_text_tokens, batch_attention_masks = padding(batch_text, roberta_tokenizer)
    batch_labels = torch.tensor(batch_labels)
    return batch_text_tokens, batch_attention_masks, batch_labels

def audio_batchs(sessions):
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_text, batch_audio, batch_video, batch_labels = [], [], [], []
    max_length = 400000
    for session in sessions:
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, split_type, sess, uttid = line
        audio = get_audio(audio_processor, wav_path)
        audio = audio[:max_length]
        batch_audio.append(audio)
        label_index = label_list.index(emotion)
        batch_labels.append(label_index)
    batch_audio = padding_audio_or_video(batch_audio)
    batch_labels = torch.tensor(batch_labels)
    return batch_audio, batch_labels

def video_batchs(sessions):
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_text, batch_audio, batch_video, batch_labels = [], [], [], []
    max_length = 400000
    for session in sessions:
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, split_type, sess, uttid = line
        video_path = find_video(split_type, sess, uttid)
        video = np.load(video_path)
        batch_video.append(video)
        label_index = label_list.index(emotion)
        batch_labels.append(label_index)
    batch_video = padding_audio_or_video(batch_video)
    batch_labels = torch.tensor(batch_labels)
    return batch_video, batch_labels

def all_batchs(sessions):
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_text, batch_audio, batch_video, batch_labels = [], [], [], []
    max_length = 400000
    for session in sessions:
        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, split_type, sess, uttid = line
            inputString += '<s' + str(speaker+1) + '> ' + utt + ' '
            now_speaker = speaker
        prompt = "Now" + '<s' + str(now_speaker+1) + '> '+"feels"
        concat_string = inputString.strip() + " " + "</s>" + " " + prompt
        batch_text.append(encode_right_truncated(concat_string, roberta_tokenizer))
        audio = get_audio(audio_processor, wav_path)
        audio = audio[:max_length]
        batch_audio.append(audio)
        video_path = find_video(split_type, sess, uttid)
        video = np.load(video_path)
        batch_video.append(video)
        label_index = label_list.index(emotion)
        batch_labels.append(label_index)
    batch_text_tokens, batch_attention_masks = padding(batch_text, roberta_tokenizer)
    batch_audio = padding_audio_or_video(batch_audio)
    batch_video = padding_audio_or_video(batch_video)
    batch_labels = torch.tensor(batch_labels)
    return batch_text_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels

def all_features_batchs(sessions):
    label_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    batch_text, batch_audio, batch_video, batch_labels, batch_speakers, batch_dia_ids, batch_utt_ids = [], [], [], [], [], [], []
    max_length = 400000
    for session in sessions:
        inputString = ""
        now_speaker = None
        for turn, line in enumerate(session):
            speaker, utt, wav_path, video_path, start_time, end_time, emotion, split_type, sess, uttid = line
            inputString += '<s' + str(speaker+1) + '> ' + utt + ' '
            now_speaker = speaker
        prompt = "Now" + '<s' + str(now_speaker+1) + '> '+"feels"
        concat_string = inputString.strip() + " " + "</s>" + " " + prompt
        batch_text.append(encode_right_truncated(concat_string, roberta_tokenizer))
        audio = get_audio(audio_processor, wav_path)
        audio = audio[:max_length]
        batch_audio.append(audio)
        video_path = find_video(split_type, sess, uttid)
        video = np.load(video_path)
        batch_video.append(video)
        label_index = label_list.index(emotion)
        batch_labels.append(label_index)
        batch_speakers.append(speaker)
        batch_dia_ids.append(int(sess))
        batch_utt_ids.append(int(uttid))
    batch_text_tokens, batch_attention_masks = padding(batch_text, roberta_tokenizer)
    batch_audio = padding_audio_or_video(batch_audio)
    batch_video = padding_audio_or_video(batch_video)
    batch_labels = torch.tensor(batch_labels)
    batch_speakers = torch.tensor(batch_speakers)
    batch_dia_ids = torch.tensor(batch_dia_ids)
    batch_utt_ids = torch.tensor(batch_utt_ids)
    return batch_text_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels, batch_speakers, batch_dia_ids, batch_utt_ids
