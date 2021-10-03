import pickle

import torch


def save_model(model, models_path):
    torch.save(model.state_dict(), models_path)


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))


def save_pickle_file(file_path: str, object_to_save):
    with open(file_path, 'wb') as output:
        pickle.dump(object_to_save, output, pickle.HIGHEST_PROTOCOL)


def load_pickle_file(file_path: str):
    with open(file_path, 'rb') as config_dictionary_file:
        return pickle.load(config_dictionary_file)