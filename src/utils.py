from models.osu_model import OsuModel
from pydub import AudioSegment
import os
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import json
import constants


# Deprecated, used for old model
def labels_from_csv(csv_file):
    data = np.genfromtxt(csv_file, delimiter=",", dtype=str, encoding='utf-8')
    file_paths = data[:, 0]
    labels = data[:, 1:].astype(float)

    return file_paths, labels


def labels_from_json(json_file):
    with open(json_file) as json_data:
        max_len = 0
        file_paths = []
        attributes_list = []
        empty_token = {
            constants.exists_key: 2,
            **{k: 0 for k in constants.predictions_keys if k != constants.exists_key}
        }

        data = json.load(json_data)
        for splice in data:
            attributes = splice[constants.json_attributes_key]
            file_paths.append(splice[constants.json_file_path_key])
            for i in range(len(attributes)):
                attributes[i][constants.exists_key] = 1 if i < len(attributes) - 1 else 0
            max_len = max(max_len, len(attributes))
            attributes_list.append(attributes)

        padded_attributes_list = []
        for attributes in attributes_list:
            # Exists set to 2 to denote padded
            padded_attributes = attributes + [empty_token] * (max_len - len(attributes))
            padded_attributes_list.append(padded_attributes)

        final_attributes_list = np.array([
            [[
                attr[key] for key in constants.predictions_keys
            ]
             for attr in attrs]
            for attrs in padded_attributes_list])
        return file_paths, final_attributes_list


def get_model():
    print("Retrieving model...")
    model = OsuModel(
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )
    if os.path.exists(constants.trained_model_path):
        print("Trained model exists. Loading saved state")
        checkpoint = torch.load(constants.trained_model_path)
        model.load_state_dict(checkpoint)

    return model

def get_model_infer():
    print("Retrieving model...")
    model = OsuModel(
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )
    if os.path.exists(constants.best_model_path):
        print("Loading best model")
        checkpoint = torch.load(constants.best_model_path)
        model.load_state_dict(checkpoint)

    return model


def causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    return mask


def audio_to_spectrogram_tensor(audio_file):
    y, sr = librosa.load(audio_file, sr=None, duration=10.24)
    n_fft = 256
    hop_length = 441
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S))

    spectrogram = S_db[:128, :1024].T

    spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32)
    padded_tensor = pad_spectrogram_tensor(spectrogram_tensor)
    normalized_tensor = normalize_tensor(padded_tensor)

    return normalized_tensor


def pad_spectrogram_tensor(tensor, target_shape=(1024, 128)):
    current_shape = tensor.shape

    if current_shape == target_shape:
        return tensor

    padding_height = target_shape[0] - current_shape[0]

    if padding_height > 0:
        tensor = F.pad(tensor, (0, 0, 0, padding_height), mode='constant', value=0)

    return tensor


# Z-score Normalization
def normalize_tensor(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std * 2)


# 10.24s intervals
def splice_audio(file_path, beatmap_id, interval_ms=constants.seq_length * 10):
    _, file_name = os.path.split(file_path)
    prefix, file_extension = os.path.splitext(file_name)
    file_extension = file_extension[1:]  # Remove leading dot
    new_directory = constants.splice_directory

    if not os.path.exists(new_directory):
        os.mkdir(new_directory)

    audio = AudioSegment.from_file(file_path)
    audio_duration = len(audio)
    audio_splices = []

    for i in range(0, audio_duration, interval_ms):

        chunk = audio[i:i + interval_ms]
        chunk_name = os.path.join(new_directory, f"{beatmap_id}-{prefix}_{i // interval_ms}.{file_extension}")
        chunk.export(chunk_name, format=file_extension)
        audio_splices.append(chunk_name)

    return audio_splices
