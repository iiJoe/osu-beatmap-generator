import time
import os
import torch

import constants
from utils import audio_to_spectrogram_tensor, get_model_infer, splice_audio


def infer(src):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_infer().to(device)
    model.eval()
    with torch.no_grad():
        src = src.to(device, dtype=torch.float32)

        batch_size = src.size(0)
        tgt = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
        tgt = tgt.expand(batch_size, 1, -1)

        for _ in range(120):
            spec_pad_mask = (src.abs().sum(dim=1) == 0)
            hit_pad_mask = (tgt == 0).all(dim=2)
            hit_pad_mask[:, 0] = False

            output = model(src, tgt, spec_pad_mask=spec_pad_mask, hit_pad_mask=hit_pad_mask)
            output[:, -1, 0] = (output[:, -1, 0] > 0.5).float()
            tgt = torch.cat((tgt, output[:, -1, :].unsqueeze(1)), dim=1)
            if torch.all(output[:, -1:, 0] == 0):
                break

        normalize_bounds = torch.tensor(constants.predictions_normalize, dtype=torch.float32).to(device)
        tgt = tgt[:, 1:, :]  # Remove the start token
        tgt[:, :, 1:] = tgt[:, :, 1:] * normalize_bounds

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
        spectrograms = torch.stack([
            torch.nn.functional.pad(
                spec,
                (0, max(0, 512 - spec.shape[1])),
                value=0
            )[:, :512]
            for spec in (audio_to_spectrogram_tensor(splice) for splice in spliced_audio_paths)
        ])

        notes = infer(spectrograms)
        for i in range(notes.size(0)):
            for note in notes[i]:
                print(f"{note[1].int()},{note[2].int()},{((note[3] + i * constants.seq_length) * 10).int()},1,{note[5].int()},0:0:0:0:")
                if note[0] == 0:
                    break

            print(f"=== End of splice {i} ===")


infer_all()
