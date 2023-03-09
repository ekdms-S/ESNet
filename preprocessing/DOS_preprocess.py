import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess DOS data and generate input files')

    # train data
    parser.add_argument('--train_filename', type=str, default='MP_relax_data.pkl')
    parser.add_argument('--train_del_idx_filename', type=str, default='MP_relax_deleted_idx.pkl')
    parser.add_argument('--train_input_filename', type=str, default='MP_relax_dos.pkl')
    parser.add_argument('--train_target_filename', type=str, default='MP_relax_target.pkl')
    parser.add_argument('--train_state_dict_filename', type=str, default='MP_relax_dos_info.pkl')

    # validation data: 2eORR RS2RE
    parser.add_argument('--val_relax_filename', type=str, default='2eORR_relax_data.pkl')
    parser.add_argument('--val_relax_del_idx_filename', type=str, default='2eORR_relax_deleted_idx.pkl')
    parser.add_argument('--val_relax_input_filename', type=str, default='2eORR_relax_dos.pkl')
    parser.add_argument('--val_relax_target_filename', type=str, default='2eORR_relax_target.pkl')

    # validation data: 2eORR IS2RE
    parser.add_argument('--val_init_filename', type=str, default='2eORR_init_data.pkl')
    parser.add_argument('--val_init_del_idx_filename', type=str, default='2eORR_init_deleted_idx.pkl')
    parser.add_argument('--val_init_input_filename', type=str, default='2eORR_init_dos.pkl')
    parser.add_argument('--val_init_target_filename', type=str, default='2eORR_init_target.pkl')

    # application data: 2eORR IS2RE
    parser.add_argument('--is_appl_val', type=bool, default=False)
    parser.add_argument('--appl_filename', type=str, default='2eORR_appl_init_data.pkl')
    parser.add_argument('--appl_del_idx_filename', type=str, default='2eORR_appl_init_deleted_idx.pkl')
    parser.add_argument('--appl_input_filename', type=str, default='2eORR_appl_init_dos.pkl')

    args = parser.parse_args()

    return args


class DOSOperation:
    def __init__(self, args):
        print('Start data processing ...')

        # 1. train data: MP relax DOS
        train_data = load_pickle(args.train_filename)
        train_del_idx = load_pickle(args.train_del_idx_filename)

        # exclude data which have density of states over 50 and formation energy larger than 5 eV/atoms
        train_x = np.array([train_data[i]['dos'] for i in range(len(train_data))])
        train_x = np.delete(train_x, train_del_idx, 0)
        train_y = np.array([train_data[i]['target'] for i in range(len(train_data))])
        train_y = np.delete(train_y, train_del_idx, 0)

        # 2. validation data1: 2eORR relax DOS (2eORR RS2RE)
        relax_test_data = load_pickle(args.val_relax_filename)
        relax_test_del_idx = load_pickle(args.val_relax_del_idx_filename)

        # exclude data which have density of states over 50 and formation energy larger than 5 eV/atoms
        relax_test_x = np.array([relax_test_data[i]['input'] for i in range(len(relax_test_data))])
        relax_test_x = np.delete(relax_test_x, relax_test_del_idx, 0)
        relax_test_y = np.array([relax_test_data[i]['target'] for i in range(len(relax_test_data))])
        relax_test_y = np.delete(relax_test_y, relax_test_del_idx, 0)

        # 3. validation data2: 2eORR init DOS (2eORR IS2RE)
        init_test_data = load_pickle(args.val_init_filename)
        init_test_del_idx = load_pickle(args.val_init_del_idx_filename)

        # exclude data which have density of states over 50 and formation energy larger than 5 eV/atoms
        init_test_x = np.array([init_test_data[i]['input'] for i in range(len(init_test_data))])
        init_test_x = np.delete(init_test_x, init_test_del_idx, 0)
        init_test_y = np.array([init_test_data[i]['target'] for i in range(len(init_test_data))])
        init_test_y = np.delete(init_test_y, init_test_del_idx, 0)

        # save data
        save_pickle(train_y, args.train_target_filename)
        save_pickle(relax_test_y, args.val_relax_target_filename)
        save_pickle(init_test_y, args.val_init_target_filename)

        self.train_x = train_x
        self.relax_test_x = relax_test_x
        self.init_test_x = init_test_x

        self.preprocess(args)

    def preprocess(self, args):
        train_x = np.transpose(self.train_x, (0, 2, 1))
        relax_test_x = np.transpose(self.relax_test_x, (0, 2, 1))
        init_test_x = np.transpose(self.init_test_x, (0, 2, 1))

        scaler = StandardScaler()
        sc_train_x = scaler.fit_transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        mean, std = scaler.mean_, np.sqrt(scaler.var_)
        train_info = {'mean': mean, 'std': std}
        save_pickle(train_info, args.train_state_dict_filename)

        sc_relax_test_x = scaler.transform(relax_test_x.reshape(-1, relax_test_x.shape[-1])).reshape(relax_test_x.shape)
        sc_init_test_x = scaler.transform(init_test_x.reshape(-1, init_test_x.shape[-1])).reshape(init_test_x.shape)

        save_pickle(sc_train_x, args.train_input_filename)
        save_pickle(sc_relax_test_x, args.val_relax_input_filename)
        save_pickle(sc_init_test_x, args.val_init_input_filename)


class ApplDOSOperation:
    def __init__(self, args):
        print('Start application data processing ... ')

        appl_data = load_pickle(args.appl_filename)
        appl_del_idx = load_pickle(args.appl_del_idx_filename)

        # exclude data which have density of states over 50 and formation energy larger than 5 eV/atoms
        appl_x = np.array([appl_data[i]['input'] for i in range(len(appl_data))])
        appl_x = np.delete(appl_x, appl_del_idx, 0)

        self.appl_x = appl_x

        self.preprocess(args)

    def preprocess(self, args):
        train_info = load_pickle(args.train_state_dict_filename)
        mean, std = train_info['mean'], train_info['std']

        appl_x = np.transpose(self.appl_x, (0, 2, 1))
        sc_appl_x = std_scaling(appl_x.reshape(-1, appl_x.shape[-1]), mean, std).reshape(appl_x.shape)

        save_pickle(sc_appl_x, args.appl_input_filename)


def get_inputs(type, opt=None):
    assert type in ['train', 'val', 'appl']

    if type == 'appl':
        input = load_pickle(f'2eORR_appl_init_dos.pkl')
    else:
        if type == 'train':
            dataset, opt = 'MP', 'relax'
        else:  # 'val'
            assert opt is not None
            dataset = '2eORR'

        input = load_pickle(f'{dataset}_{opt}_dos.pkl')
        target = load_pickle(f'{dataset}_{opt}_target.pkl')

    for i in range(9):
        if i == 0:
            input_up = np.expand_dims(input[:, :, 2 * i], axis=-1)
            input_down = np.expand_dims(input[:, :, 2 * i + 1], axis=-1)
        else:
            input_up = np.concatenate((input_up, np.expand_dims(input[:, :, 2 * i], axis=-1)), axis=-1)
            input_down = np.concatenate((input_down, np.expand_dims(input[:, :, 2 * i + 1], axis=-1)), axis=-1)
    input_comp = input[:, :111, -1]

    if type == 'appl':
        return input_up, input_down, input_comp
    else:
        return input_up, input_down, input_comp, target


if __name__ == '__main__':
    args = parse_args()
    DOSOperation(args)

    if args.is_appl_val:
        ApplDOSOperation(args)