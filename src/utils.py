from models.osu_model import OsuModel
from pydub import AudioSegment
import os
import torch
import librosa
import json
import constants


def labels_from_json(json_file):
    with open(json_file) as json_data:
        file_paths = []
        attributes_list = []

        data = json.load(json_data)

        for splice in data:
            attributes = splice[constants.json_attributes_key]
            file_paths.append(splice[constants.json_file_path_key])
            for i in range(len(attributes)):
                attributes[i][constants.exists_key] = 1 if i < len(attributes) - 1 else 0
            attributes_list.append(attributes)

        final_attributes_list = [
            [[
                attr[key] for key in constants.predictions_keys
            ]
             for attr in attrs]
            for attrs in attributes_list]
        return file_paths, final_attributes_list


def get_model(path=constants.trained_model_path):
    print("Retrieving model...")
    model = OsuModel()
    if os.path.exists(path):
        print("Model exists. Loading saved state")
        checkpoint = torch.load(path, weights_only=True)
        model.load_state_dict(checkpoint)

    return model


def get_model_infer():
    return get_model(constants.best_model_path)


def causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    return mask


def audio_to_spectrogram_tensor(audio_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y, sr = librosa.load(audio_file, sr=22050)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32).to(device)

    return spectrogram


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


def collate_fn(batch):
    spectrograms, targets = zip(*batch)

    padded_spectrograms = torch.stack([
        torch.nn.functional.pad(spectrogram, (0, max(0, 512 - spectrogram.shape[1])), value=0)[:, :512]
        for spectrogram in spectrograms
    ])
    targets = torch.stack([
        torch.nn.functional.pad(target, (0, 0, 0, max(0, 120 - target.shape[0])), value=0)[:120]
        for target in targets
    ])

    return padded_spectrograms, targets
