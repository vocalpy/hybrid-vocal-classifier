from keras.models import Sequential
from keras.layers import AtrousConvolution2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, Permute, Reshape


def dilated(input_width, input_height):
    """
    Dilated convolutional network for segmentation [1]_.
    Based on the Keras implementation of VGG16, and the implementation of [1]_
    by Nicolò Valigi (https://github.com/nicolov/segmentation_keras), itself
    based on the structure of DilatedNet originally written in Caffe
    (https://github.com/fyu/dilation/blob/master/models/dilation8_pascal_voc_deploy.prototxt).

    Parameters
    ----------
    input_width : integer
    input_height : integer

    Returns
    -------
    model : Keras model object

    References
    ----------
    ..[1] Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by
    dilated convolutions." arXiv preprint arXiv:1511.07122 (2015).
    
    """
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same',
                            name='conv1_1', input_shape=(3,input_width,
                                                        input_height)))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', 
                            name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same',
                            name='conv2_1'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', 
                            name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', 
                            name='conv3_1'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same',
                            name='conv3_2'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same',
                            name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same',
                            name='conv4_1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same',
                            name='conv4_2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same',
                            name='conv4_3'))

    # "Ablate" the 2 MaxPool layers from VGG16,
    # Begin dilated convolutional layers
    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2),
                                  activation='relu', name='conv5_1'))
    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2),
                                  activation='relu', name='conv5_2'))
    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2),
                                  activation='relu', name='conv5_3'))
    import pdb;pdb.set_trace()

    # Replace the FC layer from VGG16 with a convolution
    model.add(AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4),
                                  activation='relu', name='fc6'))
    #TODO: add dropout here
    model.add(Convolution2D(4096, 1, 1, activation='relu', name='fc7'))
    #TODO: add dropout here
    # Note: this layer has linear activations, not ReLU
    model.add(Convolution2D(21, 1, 1, name='fc-final'))

    # Context module
    model.add(ZeroPadding2D(padding=(33, 33)))
    #note these layers have different learning rates in original model
    model.add(Convolution2D(42, 3, 3, activation='relu', name='ct_conv1_2'))
    model.add(AtrousConvolution2D(84, 3, 3, atrous_rate=(2, 2),
                                  activation='relu', name='ct_conv2_1'))
    model.add(AtrousConvolution2D(168, 3, 3, atrous_rate=(4, 4),
                                  activation='relu', name='ct_conv3_1'))
    model.add(AtrousConvolution2D(336, 3, 3, atrous_rate=(8, 8),
                                  activation='relu', name='ct_conv4_1'))
    model.add(AtrousConvolution2D(672, 3, 3, atrous_rate=(16, 16),
                                  activation='relu', name='ct_conv5_1'))
    model.add(Convolution2D(672, 3, 3, activation='relu', name='ct_fc1'))
    model.add(Convolution2D(21, 1, 1, name='ct_final'))

    # The softmax layer doesn't work on the (width, height, channel)
    # shape, so we reshape to (width*height, channel) first.
    # https://github.com/fchollet/keras/issues/1169
    curr_width, curr_height, curr_channels = model.layers[-1].output_shape[1:]
    model.add(Reshape((curr_width*curr_height, curr_channels)))
    model.add(Activation('softmax'))
    model.add(Reshape((curr_width, curr_height, curr_channels)))

    #initialize context model layers with identity

    return model
