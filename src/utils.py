import numpy as np
from models.ast_model import ast_mdl
from models.fast_transformers import fast_transformer
from models.fine_tuned_model import FineTunedModel

def labels_from_csv(csv_file):
    data = np.genfromtxt(csv_file, delimiter=",", dtype=str, encoding='utf-8')
    file_paths = data[:, 0]
    labels = data[:, 1:].astype(float)

    return file_paths, labels

def get_model():
    # TODO check if state exists, load if so
    fine_tuned_model = FineTunedModel(ast_mdl, fast_transformer)
    return fine_tuned_model
