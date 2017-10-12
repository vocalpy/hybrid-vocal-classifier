from keras.models import Sequential
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


def conv_out_size(w, f, p, s): return (w-f+2*p) / s + 1


def pool_out_size(w, f, s): return (w - f) / s + 1


def flatwindow(input_shape, num_label_classes, local_window_timebins=96):
    """DCNN model but flatten output of sliding window, pass to fully connected layer.
    """

    model = Sequential()

    model.add(Conv2D(16, (5, 5), activation='relu', name='conv1_1', input_shape=input_shape))
    model.add(Conv2D(16, (1, 1), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))

    model.add(Conv2D(16, (5, 5), activation='relu', name='conv2_1'))
    model.add(Conv2D(16, (1, 1), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))

    model.add(Conv2D(16, (4, 4), activation='relu', name='conv3_1'))
    model.add(Conv2D(16, (1, 1), activation='relu', name='conv3_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))

    # calculate shape that local window would have after passing through
    # convolution + pooling layers.
    # For y axis of window, the output shape should already be correct.
    local_window_freqbins = model.layers[-1].output_shape[2]  # i.e. rows
    # But for x axis we need to calculate it

    for layer in model.layers:
        if type(layer) == Conv2D:  # if this is a convolution layer
            f = layer.kernel_size[-1]
            local_window_timebins = conv_out_size(local_window_timebins, f, 0, 1)
        elif type(layer) == MaxPooling2D:
            f = layer.pool_size[0]
            s = layer.strides[0]
            local_window_timebins = pool_out_size(local_window_timebins, f, s)

    if not local_window_timebins.is_integer():
        raise ValueError('computing timebins for "local" window layer in '
                         'flatwindow model did not result in a whole number, '
                         'instead resulted in {}.\nPlease check kernel sizes'
                         .format(local_window_timebins))
    else:
        local_window_timebins = int(local_window_timebins)

    # note that with keras 2.0 API, specify kernel dimensions as
    # **width** first then **height**, so below for our "local window"
    # layer we want timebins first (window width) and then
    # freqbins (window height)
    model.add(Conv2D(num_label_classes * 5,
                     (local_window_timebins,
                      local_window_freqbins),
                     activation='relu',
                     name='full'))
    # pretty sure last softmax layer in model consists of yet another convolution
    # where the number of filters equals the number of syllable classes. And he
    # uses identity activation

    model.add(Flatten())
    output_ns = model.layers[-1].output_shape[-1]
    model.add(Dense(output_ns))
    model.add(Activation('relu'))
    model.add(Dense(num_label_classes))
    model.add(Activation('softmax'))

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
