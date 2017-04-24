import numpy as np

from keras.models import Sequential, Model
from keras.layers.core import Activation, Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       Convolution3D, ZeroPadding2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam

#import theano.tensor as T

def conv_out_size(w,f,p,s): return (w-f+2*p) / s + 1
def pool_out_size(w,f,s): return (w - f) / s + 1

# def Koumura_crossentropy(silent_gap_label_onehot):
#     """
#     conditional crossentropy that ignores "silent gap" label
#     """
#
#     #closure so I can have value for silent_gap_label_onehot inside loss function
#     def calc(y_true,y_pred):
#         y_true_not_silent_gap_inds = np.where((y_true!=silent_gap_label_onehot)).any(axis=1)
#         y_true_no_silent_gaps = y_true[y_true_not_silent_gap_inds,:]
#         y_pred_no_silent_gaps = y_true[y_true_not_silent_gap_inds,:]
#         return T.nnet.categorical_crossentropy(y_pred,y_true)
#
#     return calc

def DCNN(input_shape,num_syllable_classes,local_window_timebins=96):
    """
    Keras implementation of the DCNN model in Koumura Okanoya 2016.
    The model is built with three sets of convolutional layers, where each convolutional
    layer contains:
        - a 2d convolution layer with 16 5x5 filters (4x4 in the third layer)
        - a cascaded cross-channel parametric layer, implemented as a 2d conv. layer with
            16 1x1 filters
        - a max pooling layer with a filter size and stride size of 2x2
    
    The final two layers are both 2D convolutions but are supposed to act as a window that
    slides along the input spectrogram. Hence the filter size of the second-to-last layer is 
    equivalent to the size of this window **after** passing through the convolution and pooling layers.
    The size after passing through those layers is computed by a helper function.
    The height of the window is taken to be the same as freq_bins,
    i.e. the y-axis of the spectrogram. The length of the window is decided by the user;
    in Koumura Okanoya 2016 it is set to 96 time bins, which given the default FFT parameters
    should be equal to 96 ms. The layer is assigned 240 filters of the same size as the window.
    
    For the last layer, the number of filters equals the number of classes, and the filter size
    is 1x1. The output of this layer has dimensions (number of classes, number of windows, 1).
    The 3rd dimension is an artifact of the 1x1 2-D convolution, and is discarded by reshaping.

    The output than goes through a softmax activation layer producing an output with shape:
        (number of classes, number of windows)
    
    Inputs:
        input_shape -- tuple, shape of inputs to first Convolution2D layer. Expects a shape (1,freq_bins,time_bins)
                       where there freq_bins and time_bins are output from hvc.utils.make_spect_from_song.
        num_syllable_classes -- number of classes of syllables, i.e. labels. This value determines
                                the number of values in the output after running through the softmax layer.
        local_window_timebins -- number of timebins to use for the window that slides along the entire spectrogram,
                                 i.e., the size of the window along the X axis. Default is 96.
    
    Returns:
        model -- instance of Keras model object that represents the DCNN model
    
    """
    
    model = Sequential()
   
    model.add(Convolution2D(16,5, 5, activation='relu', name='conv1_1',input_shape=input_shape))
    model.add(Convolution2D(16,1, 1, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='pool1'))
    
    model.add(Convolution2D(16,5, 5, activation='relu', name='conv2_1'))
    model.add(Convolution2D(16,1, 1, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='pool2'))
    
    model.add(Convolution2D(16,4, 4, activation='relu', name='conv3_1'))
    model.add(Convolution2D(16,1, 1, activation='relu', name='conv3_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='pool3'))
    
    #calculate shape that local window would have after passing through convolution + pooling layers.
    #For y axis of window, the output shape should already be correct.
    local_window_freqbins = model.layers[-1].output_shape[2] #i.e. rows
    #But for x axis we need to calculate it

    for layer in model.layers:
        if layer.name.find('conv') > -1: # if this is a convolution layer
            f = layer.nb_row
            local_window_timebins = conv_out_size(local_window_timebins,f,0,1)
        elif layer.name.find('pool') > -1:
            f = layer.pool_size[0]
            s = layer.strides[0]
            local_window_timebins = pool_out_size(local_window_timebins,f,s)
                
    model.add(Convolution2D(240,local_window_freqbins,local_window_timebins,activation='relu', name='full'))
    #pretty sure last softmax layer in model consists of yet another convolution where the
    #number of filters equals the number of syllable classes. And he uses identity activation

    model.add(Convolution2D(num_syllable_classes,1,1,activation='linear'))
    #This yields a 3-D output (neglecting the 1st batch size dimension):
    #(number of syllable classes, number of time bins from sliding window, 1).
    #The third dimension is just an artifact of using a convolution so we reshape.
    #Reshaping allows us to apply a softmax.
    outshape = model.layers[-1].output_shape
    reshapeshape = (outshape[3],outshape[1])
    
    model.add(Reshape(reshapeshape))
    model.add(Activation("softmax"))

    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)               
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def DCNN2(input_shape,num_syllable_classes,layers_dict,silent_gap_label,local_window_timebins=96):
    """
    Same as DCNN but trying with TimeDistributed wrapper
    New argument:
        Layers_dict -- dictionary with layer names as keys and either filter size
                       or stride (for pooling) as value. Workaround since I can't
                       get names to work with TimeDistrbuted wrapper, but need
                       to loop through layers and programatically determine
                       number of time bins in 'sliding window' layer.
    """
    
    model = Sequential()
   
    model.add(TimeDistributed(
        (Convolution2D(16,5, 5, activation='relu', name='conv1_1')),
        input_shape=input_shape))
    model.add(TimeDistributed(
        (Convolution2D(16,1, 1, activation='relu', name='conv1_2'))))
    model.add(TimeDistributed(
        (MaxPooling2D((2,2), strides=(2,2),name='pool1'))))
    
    model.add(TimeDistributed(
        (Convolution2D(16,5, 5, activation='relu', name='conv2_1'))))
    model.add(TimeDistributed(
        (Convolution2D(16,1, 1, activation='relu', name='conv2_2'))))
    model.add(TimeDistributed(
        (MaxPooling2D((2,2), strides=(2,2),name='pool2'))))

    model.add(TimeDistributed(
        (Convolution2D(16,4, 4, activation='relu', name='conv3_1'))))
    model.add(TimeDistributed(
        (Convolution2D(16,1, 1, activation='relu', name='conv3_2'))))
    model.add(TimeDistributed(
        (MaxPooling2D((2,2), strides=(2,2),name='pool3'))))
    
    #calculate shape that local window would have after passing through
    #convolution + pooling layers.
    #For y axis of window, the output shape should already be correct.
    local_window_freqbins = model.layers[-1].output_shape[3] #i.e. rows
    #But for x axis we need to calculate it
    for name, val in layers_dict.items():
        if name.find('conv') > -1: # if this is a convolution layer
            f = val
            local_window_timebins = conv_out_size(local_window_timebins,f,0,1)
        elif name.find('pool') > -1:
            f = val #filter size and stride are same (2)
            s = val
            local_window_timebins = pool_out_size(local_window_timebins,f,s)
    
    model.add(TimeDistributed(
        (Convolution2D(240,
                       local_window_freqbins,
                       local_window_timebins,
                       activation='relu',
                       name='full'))))
    #pretty sure last softmax layer in model consists of yet another convolution
    # where the number of filters equals the number of syllable classes. And he
    # uses identity activation
    model.add(TimeDistributed(
        (Convolution2D(num_syllable_classes,1,1,activation='linear'))))
    #This yields a 3-D output (neglecting the 1st batch size dimension):
    #(number of syllable classes, number of time bins from sliding window, 1).
    #The third dimension is just an artifact of using a convolution so we reshape.
    #Reshaping also allows us to apply a softmax. The softmax activation is applied
    #to the last dimension of the output_shape, so we switch dims[2] and [4]

    outshape = model.layers[-1].output_shape
    reshapeshape = (outshape[1],outshape[4],outshape[2])
    
    model.add(Reshape(reshapeshape))
    model.add(TimeDistributed(Activation("softmax")))

#    k_ce = Koumura_crossentropy(silent_gap_label)
       
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def DCNN_flatwindow(input_shape,num_syllable_classes,local_window_timebins=96):
    """
    DCNN model but flatten output of sliding window, pass to fully connected layer.
    """
    
    model = Sequential()
   
    model.add(Convolution2D(16,5, 5, activation='relu', name='conv1_1',input_shape=input_shape))
    model.add(Convolution2D(16,1, 1, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='pool1'))
    
    model.add(Convolution2D(16,5, 5, activation='relu', name='conv2_1'))
    model.add(Convolution2D(16,1, 1, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='pool2'))
    
    model.add(Convolution2D(16,4, 4, activation='relu', name='conv3_1'))
    model.add(Convolution2D(16,1, 1, activation='relu', name='conv3_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='pool3'))
    
    #calculate shape that local window would have after passing through
    #convolution + pooling layers.
    #For y axis of window, the output shape should already be correct.
    local_window_freqbins = model.layers[-1].output_shape[2] #i.e. rows
    #But for x axis we need to calculate it

    for layer in model.layers:
        if layer.name.find('conv') > -1: # if this is a convolution layer
            f = layer.nb_row
            local_window_timebins = conv_out_size(local_window_timebins,f,0,1)
        elif layer.name.find('pool') > -1:
            f = layer.pool_size[0]
            s = layer.strides[0]
            local_window_timebins = pool_out_size(local_window_timebins,f,s)
                
    model.add(Convolution2D(num_syllable_classes*5,
                            local_window_freqbins,
                            local_window_timebins,
                            activation='relu',
                            name='full'))
    #pretty sure last softmax layer in model consists of yet another convolution
    #where the number of filters equals the number of syllable classes. And he
    #uses identity activation

    model.add(Flatten())
    output_ns = model.layers[-1].output_shape[-1]
    model.add(Dense(output_ns))
    model.add(Activation('relu'))
    model.add(Dense(num_syllable_classes))
    model.add(Activation('softmax'))

    #adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model    

def naive_LSTM(input_shape,num_syllable_classes):
    """
    simplest LSTM model we could think of.
    """
    
    model = Sequential()
    
    model.add(LSTM(256, input_shape=(input_shape),return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(TimeDistributed(Dense(num_syllable_classes)))
    model.add(Activation("softmax"))
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    return model

def VGG_16(input_shape,num_syllable_classes,weights_path=None):
    """
    VGG_16 convnet
    based on https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    """
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_syllable_classes, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
