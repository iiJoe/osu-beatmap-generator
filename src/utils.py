import numpy as np

from models.fast_transformers import fast_transformer
from models.fine_tuned_model import FineTunedModel
from models.pos_embed import PositionalEncoding
import constants

def labels_from_csv(csv_file):
    data = np.genfromtxt(csv_file, delimiter=",", dtype=str, encoding='utf-8')
    file_paths = data[:, 0]
    labels = data[:, 1:].astype(float)

    return file_paths, labels

def get_model():
    # TODO check if state exists, load if so
    pos_embed = PositionalEncoding(constants.seq_length, 128)
    fine_tuned_model = FineTunedModel(pos_embed, fast_transformer)
    return fine_tuned_model
