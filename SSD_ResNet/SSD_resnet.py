# coding=utf-8
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import add, Flatten
from ssd_layers import Normalize
from ssd_layers import PriorBox
import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import merge
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model


NB_CLASS = 20
IM_WIDTH = 300
IM_HEIGHT = 224

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def bottleneck_Block(inpt,nb_filters,strides=(1,1),with_conv_shortcut=False):
    k1, k2, k3 = nb_filters
    x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def resnet_34(width,height,channel,classes):
    inpt = Input(shape=(width, height, channel))
    x = ZeroPadding2D((3, 3))(inpt)

    #conv1
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    #conv2_x
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))

    #conv3_x
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))

    #conv4_x
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))

    #conv5_x
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model

def resnet_50(input_shape, num_classes=21):

    input_tensor = Input(shape=input_shape)
    
    x = ZeroPadding2D((3, 3))(input_tensor)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    #conv2_x
    x = bottleneck_Block(x, nb_filters=[64,64,256],strides=(1,1),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[64,64,256])
    x = bottleneck_Block(x, nb_filters=[64,64,256])

    #conv3_x
    x = bottleneck_Block(x, nb_filters=[128, 128, 512],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])

    #conv4_x
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])

    #conv5_x
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])

    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model

def resnet_68(input_shape, num_classes=21):
    
    net = {}
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])
    net['input'] = input_tensor
    x = Conv2d_BN(input_tensor, nb_filter=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)#150*150

    #conv2_x
    x = bottleneck_Block(x, nb_filters=[64,64,256],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[64,64,256])
    x = bottleneck_Block(x, nb_filters=[64,64,256])#75*75

    #conv3_x
    x = bottleneck_Block(x, nb_filters=[128, 128, 512],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])#38*38
    net['conv3_x'] = x

    #conv4_x
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])#19*19
    net['conv4_x'] = x

    #conv5_x
    x = bottleneck_Block(x, nb_filters=[512, 512, 512], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[512, 512, 512])
    x = bottleneck_Block(x, nb_filters=[512, 512, 512])#10*10
    net['conv5_x'] = x

    # conv6_x
    x = bottleneck_Block(x, nb_filters=[512, 512, 256], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[512, 512, 256])
    x = bottleneck_Block(x, nb_filters=[512, 512, 256])  # 5*5
    net['conv6_x'] = x

    # conv7_x
    x = bottleneck_Block(x, nb_filters=[512, 512, 256], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[512, 512, 256])
    x = bottleneck_Block(x, nb_filters=[512, 512, 256])  # 3*3
    net['conv7_x'] = x

    # conv8_x
    x = GlobalAveragePooling2D(name='pool6')(x)
    net['conv8_x'] = x

    # Prediction from conv3_x
    net['conv3_x_norm'] = Normalize(20, name='conv3_x_norm')(net['conv3_x'])
    num_priors = 3
    x = Conv2D(12, (3, 3), name="conv3_x_norm_mbox_loc", padding="same")(net['conv3_x_norm'])

    net['conv3_x_norm_mbox_loc'] = x
    flatten = Flatten(name='conv3_x_norm_mbox_loc_flat')
    net['conv3_x_norm_mbox_loc_flat'] = flatten(net['conv3_x_norm_mbox_loc'])
    name = 'conv3_x_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(63, (3, 3), name="conv3_x_norm_mbox_conf", padding="same")(net['conv3_x_norm'])

    net['conv3_x_norm_mbox_conf'] = x
    flatten = Flatten(name='conv3_x_norm_mbox_conf_flat')
    net['conv3_x_norm_mbox_conf_flat'] = flatten(net['conv3_x_norm_mbox_conf'])
    priorbox = PriorBox(img_size, 30.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv3_x_norm_mbox_priorbox')

    net['conv3_x_norm_mbox_priorbox'] = priorbox(net['conv3_x_norm'])

    # Prediction from conv4_x
    num_priors = 6
    net['conv4_x_mbox_loc'] = Conv2D(24, (3, 3), name="conv4_x_mbox_loc", padding="same")(net['conv4_x'])
    flatten = Flatten(name='conv4_x_mbox_loc_flat')
    net['conv4_x_mbox_loc_flat'] = flatten(net['conv4_x_mbox_loc'])

    name = 'conv4_x_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    net['conv4_x_mbox_conf'] = Conv2D(126, (3, 3), name="conv4_x_mbox_conf", padding="same")(net['conv4_x'])
    flatten = Flatten(name='conv4_x_mbox_conf_flat')
    net['conv4_x_mbox_conf_flat'] = flatten(net['conv4_x_mbox_conf'])
    priorbox = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_x_mbox_priorbox')
    net['conv4_x_mbox_priorbox'] = priorbox(net['conv4_x'])

    # Prediction from conv5_x
    num_priors = 6
    x = Conv2D(24, (3, 3), name="conv5_x_mbox_loc", padding="same")(net['conv5_x'])

    net['conv5_x_mbox_loc'] = x
    flatten = Flatten(name='conv5_x_mbox_loc_flat')
    net['conv5_x_mbox_loc_flat'] = flatten(net['conv5_x_mbox_loc'])
    name = 'conv5_x_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(126, (3, 3), name="conv5_x_mbox_conf", padding="same")(net['conv5_x'])

    net['conv5_x_mbox_conf'] = x
    flatten = Flatten(name='conv5_x_mbox_conf_flat')
    net['conv5_x_mbox_conf_flat'] = flatten(net['conv5_x_mbox_conf'])
    priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv5_x_mbox_priorbox')
    net['conv5_x_mbox_priorbox'] = priorbox(net['conv5_x'])

    # Prediction from conv6_x
    num_priors = 6
    x = Conv2D(24, (3, 3), name="conv6_x_mbox_loc", padding="same")(net['conv6_x'])

    net['conv6_x_mbox_loc'] = x
    flatten = Flatten(name='conv6_x_mbox_loc_flat')
    net['conv6_x_mbox_loc_flat'] = flatten(net['conv6_x_mbox_loc'])

    name = 'conv6_x_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(126, (3, 3), name="conv6_x_mbox_conf", padding="same")(net['conv6_x'])
    net['conv6_x_mbox_conf'] = x
    flatten = Flatten(name='conv6_x_mbox_conf_flat')
    net['conv6_x_mbox_conf_flat'] = flatten(net['conv6_x_mbox_conf'])
    priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_x_mbox_priorbox')
    net['conv6_x_mbox_priorbox'] = priorbox(net['conv6_x'])

    # Prediction from conv7_x
    num_priors = 6
    x = Conv2D(24, (3, 3), name="conv7_x_mbox_loc", padding="same")(net['conv7_x'])

    net['conv7_x_mbox_loc'] = x
    flatten = Flatten(name='conv7_x_mbox_loc_flat')
    net['conv7_x_mbox_loc_flat'] = flatten(net['conv7_x_mbox_loc'])

    name = 'conv7_x_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(126, (3, 3), name="conv7_x_mbox_conf", padding="same")(net['conv7_x'])
    net['conv7_x_mbox_conf'] = x
    flatten = Flatten(name='conv7_x_mbox_conf_flat')
    net['conv7_x_mbox_conf_flat'] = flatten(net['conv7_x_mbox_conf'])
    priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_x_mbox_priorbox')
    net['conv7_x_mbox_priorbox'] = priorbox(net['conv7_x'])

    # Prediction from conv8_x
    num_priors = 6
    x = Dense(num_priors * 4, name='conv8_x_mbox_loc_flat')(net['conv8_x'])
    net['conv8_x_mbox_loc_flat'] = x

    name = 'conv8_x_mbox_conf_flat'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Dense(num_priors * num_classes, name=name)(net['conv8_x'])
    net['conv8_x_mbox_conf_flat'] = x
    priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_x_mbox_priorbox')

    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    #target_shape = (1, 1, 256)
    net['conv8_x_reshaped'] = Reshape(target_shape, name='conv8_x_reshaped')(net['conv8_x'])
    net['conv8_x_mbox_priorbox'] = priorbox(net['conv8_x_reshaped'])

    # Gather all predictions
    net['mbox_loc'] = merge([net['conv3_x_norm_mbox_loc_flat'],
                             net['conv4_x_mbox_loc_flat'],
                             net['conv5_x_mbox_loc_flat'],
                             net['conv6_x_mbox_loc_flat'],
                             net['conv7_x_mbox_loc_flat'],
                             net['conv8_x_mbox_loc_flat']],
                            mode='concat',
                            concat_axis=1,
                            name='mbox_loc')

    net['mbox_conf'] = merge([net['conv3_x_norm_mbox_conf_flat'],
                              net['conv4_x_mbox_conf_flat'],
                              net['conv5_x_mbox_conf_flat'],
                              net['conv6_x_mbox_conf_flat'],
                              net['conv7_x_mbox_conf_flat'],
                              net['conv8_x_mbox_conf_flat']],
                             mode='concat',
                             concat_axis=1,
                             name='mbox_conf')

    net['mbox_priorbox'] = merge([net['conv3_x_norm_mbox_priorbox'],
                                  net['conv4_x_mbox_priorbox'],
                                  net['conv5_x_mbox_priorbox'],
                                  net['conv6_x_mbox_priorbox'],
                                  net['conv7_x_mbox_priorbox'],
                                  net['conv8_x_mbox_priorbox']],
                                 mode='concat',
                                 concat_axis=1,
                                 name='mbox_priorbox')

    if hasattr(net['mbox_loc'], '_keras_shape'):
        num_boxes = net['mbox_loc']._keras_shape[-1] // 4
    elif hasattr(net['mbox_loc'], 'int_shape'):
        num_boxes = K.int_shape(net['mbox_loc'])[-1] // 4
    net['mbox_loc'] = Reshape((num_boxes, 4), name='mbox_loc_final')(net['mbox_loc'])
    net['mbox_conf'] = Reshape((num_boxes, num_classes), name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])
    net['predictions'] = merge([net['mbox_loc'],
                                net['mbox_conf'],
                                net['mbox_priorbox']],
                               mode='concat',
                               concat_axis=2,
                               name='predictions')

    model = Model(net['input'], net['predictions'])
    return model