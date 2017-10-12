from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed


def naive_LSTM(input_shape, num_syllable_classes):
    """
    simplest LSTM model we could think of.
    """

    model = Sequential()

    model.add(LSTM(256, input_shape=(input_shape), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(num_syllable_classes)))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model
