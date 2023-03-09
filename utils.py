import os
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error


def to_path(path):
    main_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(main_path, path)


def create_directory(path, path_is_directory=False):
    p = Path(to_path(path))
    if not path_is_directory:
        dirname = p.parent
    else:
        dirname = p
    if not dirname.exists():
        os.makedirs(dirname)


def save_pickle(array, filename):
    main_dir = to_path('data')
    dir = os.path.join(main_dir, filename)
    with open(dir, 'wb') as f:
        pickle.dump(array, f)


def load_pickle(filename):
    main_dir = to_path('data')
    dir = os.path.join(main_dir, filename)
    with open(dir, 'rb') as f:
        data = pickle.load(f)

    return data


def std_scaling(array, mean, std):
    scaled_array = (array - mean) / std

    return scaled_array


def root_mean_squared_error(target, pred):
    rmse = np.sqrt(mean_squared_error(target, pred))

    return rmse