from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Reshape, Conv1D, MaxPooling1D, \
                                    GlobalAveragePooling1D, Multiply, Add, Lambda, Concatenate, GlobalMaxPooling1D, Flatten, \
                                    UpSampling1D, Conv1DTranspose, BatchNormalization, AveragePooling1D

from model.model_utils import SpinDOSFeaturizer

def ESNet():
    input_up = Input(shape=(1500, 9)) # partial DOS_up
    input_down = Input(shape=(1500, 9)) # partial DOS_down
    input_comp = Input(shape=(111)) # component

    x_up = SpinDOSFeaturizer(input_up, [64, 128, 256, 512])
    x_down = SpinDOSFeaturizer(input_down, [64, 128, 256, 512])

    # Channel concatenate
    x = Concatenate(axis=-1)([x_up, x_down])

    # TotalDOSFeaturizer
    x = Conv1D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_dos = GlobalAveragePooling1D()(x)

    # Embedding
    x_comp = Dense(64)(input_comp)

    # Spatial concatenate
    x = Concatenate(axis=1)([x_dos, x_comp])

    # FCNN
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    y = Dense(1, activation='linear')(x)

    return Model(inputs=[input_up, input_down, input_comp], outputs=y)
