from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse

from preprocessing.DOS_preprocess import get_inputs
from utils import to_path, create_directory, load_pickle, root_mean_squared_error


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate ESNet')

    parser.add_argument('--saved_model_filename', type=str, default='best_ESNet_230309.h5')
    parser.add_argument('--evaluate_version', type=str, default='IS2RE')
    parser.add_argument('--save_pred', type=bool, default=True)
    parser.add_argument('--parity_plot', type=bool, default=True)
    parser.add_argument('--save_fig', type=bool, default=True)

    args = parser.parse_args()

    return args


class Evaluate:
    def __init__(self, args):
        model_dir = to_path('saved_model')
        model_dir = os.path.join(model_dir, args.saved_model_filename)
        self.model = load_model(model_dir)

        self.opt = 'init' if args.evaluate_version == 'IS2RE' else 'relax'
        self.pred, self.actual = self.evaluate(args)

        if args.save_pred:
            create_directory(f'results/{args.evaluate_version}', path_is_directory=True)
            main_dir = to_path(f'results/{args.evaluate_version}')

            test_data = load_pickle(f'2eORR_{self.opt}_data.pkl')
            del_idx = load_pickle(f'2eORR_{self.opt}_deleted_idx.pkl')
            for idx in sorted(del_idx, reverse=True):
                del test_data[idx]

            name = [test_data[i]['name'] for i in range(len(test_data))]
            results_df = pd.DataFrame({'name': name, 'target': self.actual, 'pred': self.pred})
            results_df.to_csv(os.path.join(main_dir, f'ESNet_{args.evaluate_version}_results.csv'))

        if args.parity_plot:
            self.parity_plot(args)

    def evaluate(self, args):
        test_inputs = list(get_inputs(type='val', opt=self.opt))
        test_y = test_inputs.pop(-1)
        pred = self.model.predict(test_inputs)

        mae = mean_absolute_error(test_y, pred)
        rmse = root_mean_squared_error(test_y, pred)

        print(f'<ESNet_{args.evaluate_version}_performance> =========')
        print(f'MAE: {mae}')
        print(f'RMSE: {rmse}')

        return np.squeeze(pred), test_y

    def parity_plot(self, args):
        min_val, max_val = -5, 5
        y_range = [min_val, max_val]
        plt.figure(figsize=(5, 5))
        plt.scatter(x=self.actual, y=self.pred, s=30, color='darkorange', alpha=0.5)
        plt.plot(y_range, y_range, color='black', linewidth=1.1)
        plt.plot([0, 0], y_range, color='black', linestyle='dashed', linewidth=0.9)
        plt.plot(y_range, [0, 0], color='black', linestyle='dashed', linewidth=0.9)
        plt.xlim(y_range)
        plt.ylim(y_range)
        plt.xlabel('True', fontsize=15)
        plt.ylabel('Pred', fontsize=15)

        fig_name = f'ESNet_{args.evaluate_version}_parity_plot'
        plt.title(fig_name)

        if args.save_fig:
            create_directory(f'results/{args.evaluate_version}', path_is_directory=True)
            main_dir = to_path(f'results/{args.evaluate_version}')
            plt.savefig(os.path.join(main_dir, f'{fig_name}.png'))

        # plt.show()
        plt.close()


if __name__ == '__main__':
    args = parse_args()
    Evaluate(args)