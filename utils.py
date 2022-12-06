import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def save_pickle(array, filename):
    main_dir = '/home/ekdms1228/Documents/ESNet_221205/data'
    dir = os.path.join(main_dir, filename)
    with open(dir, 'wb') as f:
        pickle.dump(array, f)


def load_pickle(filename):
    main_dir = '/home/ekdms1228/Documents/ESNet_221205/data'
    dir = os.path.join(main_dir, filename)
    with open(dir, 'rb') as f:
        data = pickle.load(f)

    return data


def count_comp(data, pair=True, type='train', df=False):
    comp_list = []
    for d in data:
        if type == 'train':
            comp_list.extend(d['elements'])
        else:
            if pair:
                components = '_'.join(d['species'])
                comp_list.append(components)
            else:
                components = d['species']
                comp_list.extend(components)
    comp = list(set(comp_list))

    comp_count = dict()
    for c in comp:
        comp_count[c] = comp_list.count(c)

    if df:
        count_df = pd.DataFrame(list(comp_count.items()), columns=['components', 'count'])
        return comp_count, count_df
    else:
        return comp_count


def std_scaling(array, mean, std):
    scaled_array = (array - mean) / std

    return scaled_array


def root_mean_squared_error(target, pred):
    rmse = np.sqrt(mean_squared_error(target, pred))

    return rmse