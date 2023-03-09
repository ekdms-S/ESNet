import tensorflow as tf
from tensorflow import keras
import time
from sklearn.metrics import mean_absolute_error
import argparse

from model.ESNet import ESNet
from preprocessing.DOS_preprocess import get_inputs
from utils import root_mean_squared_error


def parse_args():
    parser = argparse.ArgumentParser(description='train ESNet')

    # training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--init_lr', type=float, default=0.0015)
    parser.add_argument('--early_stp_patience', type=int, default=35)
    parser.add_argument('--lr_sch_factor', type=float, default=0.5)
    parser.add_argument('--lr_sch_patience', type=int, default=15)
    parser.add_argument('--loss', type=str, default='logcosh')
    parser.add_argument('--epochs', type=int, default=300)

    # evaluate
    parser.add_argument('--is_evaluate', type=bool, default=True)
    parser.add_argument('--evaluate_version', type=str, default='IS2RE')

    args = parser.parse_args()

    return args


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.ti = time.time()

    def on_epoch_end(self, epoch, logs=None):
        train_mae = logs['mae']
        val_mae = logs['val_mae']

        tt = time.time() - self.ti

        print('Epoch{:2d}      TrainMAE: {:6.5f}      ValMAE: {:6.5f}      {:.2f}sec'
            .format(epoch, train_mae, val_mae, tt))


def train(args):
    model = ESNet()
    model.summary()

    file_path = f'saved_model/best_ESNet.h5'
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                       save_best_only=True, verbose=True)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                   patience=args.early_stp_patience, verbose=True)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.lr_sch_factor,
                                                     patience=args.lr_sch_patience, verbose=True)
    epoch_log = CustomCallback()

    optimizer = tf.optimizers.Adam(learning_rate=args.init_lr)
    model.compile(loss=args.loss, optimizer=optimizer, metrics=['mae'])

    train_inputs = list(get_inputs(type='train'))
    val_inputs = list(get_inputs(type='val', opt='init'))

    train_y = train_inputs.pop(-1)
    val_y = val_inputs.pop(-1)

    model.fit(train_inputs, train_y,
              batch_size=args.batch_size, epochs=args.epochs,
              callbacks=[model_checkpoint, lr_scheduler, epoch_log, early_stopping],
              validation_data=(val_inputs, val_y),
              verbose=0)

    if args.is_evaluate:
        if args.evaluate_version == 'IS2RE':
            test_inputs = list(get_inputs(type='val', opt='init'))
        elif args.evaluate_version == 'RS2RE':
            test_inputs = list(get_inputs(type='val', opt='relax'))
        else:
            print(f'There is no evaluate version of "{args.evaluate_version}"')

        test_y = test_inputs.pop(-1)
        pred = model.predict(test_inputs)

        mae = mean_absolute_error(test_y, pred)
        rmse = root_mean_squared_error(test_y, pred)

        print(f'<ESNet_{args.evaluate_version}_performance> =========')
        print(f'MAE: {mae}')
        print(f'RMSE: {rmse}')


if __name__ == '__main__':
    args = parse_args()
    train(args)