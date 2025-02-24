import time
import os
import torch

import constants
from utils import audio_to_spectrogram_tensor, get_model_infer, causal_mask, splice_audio

def infer(src):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_infer()
    model.eval()
    model.to(device)

    src.to(device, dtype=torch.float32)
    tgt = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], device=device)
    batch_size = src.size(0)

    tgt = tgt.expand(batch_size, 1, -1)
    for _ in range(constants.seq_length):
        _, target_len, _ = tgt.shape
        tgt_mask = causal_mask(target_len)
        normalize_bounds = torch.tensor(constants.predictions_normalize, dtype=torch.float32)

        output = model(src, tgt, tgt_mask=tgt_mask)
        output[:, -1, 0] = (output[:, -1, 0] > 0.5).float()
        output[:, -1, 1:] = output[:, -1, 1:] * normalize_bounds
        tgt = torch.cat((tgt, output[:, -1:].detach()), dim=1)

        if torch.all(output[:, -1:, 0] == 0):
            break

    tgt = tgt[:, 1:, :] # Remove the start token
    return tgt

def infer_all():
    dir = constants.audio_directory
    audio_files = [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]

    if not audio_files:
        print(f"No audio files found in {dir}")
        return

    for file in audio_files:
        audio_directory = dir + "/" + file
        timestamp = round(time.time() * 1000)
        spliced_audio_paths = splice_audio(audio_directory, timestamp)
        spectrograms = torch.stack([audio_to_spectrogram_tensor(splice) for splice in spliced_audio_paths])

        notes = infer(spectrograms)
        for i in range(notes.size(0)):
            note_count = 0
            for note in notes[i]:
                print(f"{note[1].int()},{note[2].int()},{((note[3] + i * constants.seq_length) * 10).int()},1,{note[5].int()},0:0:0:0:")
                if note[0] == 0 or note_count == 15:
                    break

            print(f"=== End of splice {i} ===")


infer_all()
