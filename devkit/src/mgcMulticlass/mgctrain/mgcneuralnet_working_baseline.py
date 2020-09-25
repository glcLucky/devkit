###----------------------------------------------------------------------------
#
# (c) Ericsson 2018 - All Rights Reserved
#
# No part of this material may be reproduced in any form
# without the written permission of the copyright owner.
# The contents are subject to revision without notice due 
# to continued progress in methodology, design and manufacturing. 
# Ericsson shall have no liability for any error or damage of any
# kind resulting from the use of these documents.
#
# Any unauthorized review, use, disclosure or distribution is 
# expressly prohibited, and may result in severe civil and 
# criminal penalties.
#
# Ericsson is the trademark or registered trademark of
# Telefonaktiebolaget LM Ericsson. All other trademarks mentioned
# herein are the property of their respective owners. 
# 
#------------------------------------------------------------------------------

# import module
import numpy as np
import pandas as pd

from keras import layers
from keras.models import Model, load_model
from keras import regularizers
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import Callback, ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from keras import optimizers
from keras.layers.merge import concatenate

from keras.utils import np_utils

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
       
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D,  GlobalAveragePooling2D
from keras.layers.core import Activation

from IPython.display import SVG
import datetime, time
import numpy as np
import os
import sys
import re
from io import StringIO

import mgctrain.resnet as resnet
#import mgctrain.densenet

from keras import backend as K
K.clear_session()


## Network in network   
def mgcNetArchNin(outLayer, l2_val, **kwargs):
    
    """
    CNN architecture - Network in network
    Network architecture summary and plot
    The output layers, either multiple layer perceptron network or maximum pooling
    Return end-to-end network architecture to be compiled and trained
    
    Argumnents:
        input_img_rows: horizontal dimension in pixel of input image
        input_img_cols:vertical dimension in pixel of input image
        channels: number of colour channel
        nb_classes: number of unique classification class exist in the dataset target
    """

    def_vals = {"input_img_rows" : 72,
                "input_img_cols" : 72,
                "channels" : 1,
                "nb_classes" : 13
               } # default parameters value

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

    input_img_rows = kwargs['input_img_rows']
    input_img_cols = kwargs['input_img_cols']
    channels = kwargs['channels']
    nb_classes = kwargs['nb_classes']

    
    # Input: 72 x 72 x 1
    img_shape = layers.Input(shape = (input_img_rows, input_img_cols, channels))

    # Layer 1
    #------------------------
    conv1 = layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu')(img_shape)
    conv1 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(conv1)
    conv1 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(conv1)
    conv1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = layers.Dropout(0.4)(conv1)

    # Layer 2
    #------------------------
    conv2 = layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(conv1)
    conv2 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv2)
    conv2 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv2)
    conv2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = layers.Dropout(0.4)(conv2)

    # Layer 3
    #------------------------
    conv3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    conv3 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv3)
    conv3 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv3)
    conv3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3 = layers.Dropout(0.4)(conv3)

    # Layer 4
    #------------------------
    #conv4 = layers.Conv2D(filters=128, kernel_size=(2, 2), padding='same', activation='relu')(conv3)
    #conv4 = layers.Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(conv4)
    #conv4 = layers.Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(conv4)
    #conv4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    #conv4 = layers.Dropout(0.4)(conv4)

    # Layer 5
    #------------------------
    output = layers.Conv2D(filters=128, kernel_size=(2, 2), padding='same', activation='relu')(conv3) # skip layer 4
    output = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(output)
    output = layers.Conv2D(filters=32, kernel_size=(1, 1))(output)
    output = layers.MaxPooling2D(pool_size=(2, 2))(output)
    output = layers.Dropout(0.4)(output)

            
    # FC Layer
    #------------------------
    outputmlp = layers.Flatten()(output)
    outputmlp = layers.Dense(64, activation = 'relu')(outputmlp)
    outputmlp = layers.Dropout(0.5)(outputmlp)

    predictionsMlp = layers.Dense(nb_classes, activation='softmax')(outputmlp)
    
    
    # global averaging
    weight_decay=1E-4
    concat_axis = 1
    
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=regularizers.l2(weight_decay),
                           beta_regularizer=regularizers.l2(weight_decay))(output)
    x = Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    
    predictionsGloAvg = layers.Dense(nb_classes,
                        activation='softmax',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        bias_regularizer=regularizers.l2(weight_decay))(x)
    
    if outLayer == "gloAvg":
        predictions = predictionsGloAvg
    elif outLayer == "mlp":
        predictions = predictionsMlp
        
    # prediction model
    model = Model(img_shape, predictions, name = 'net_in_net')


    return model

## Basic CNN with max pooling
def mgcNetArchMax(outLayer, l2_val, **kwargs):
    
    """
    CNN architecture - with maximum pooling
    Network architecture summary and plot
    The output layers, either multiple layer perceptron network or maximum pooling
    Return end-to-end network architecture to be compiled and trained
    
    Argumnents:
        input_img_rows: horizontal dimension in pixel of input image
        input_img_cols:vertical dimension in pixel of input image
        channels: number of colour channel
        nb_classes: number of unique classification class exist in the dataset target
    """

    def_vals = {"input_img_rows" : 72,
                "input_img_cols" : 72,
                "channels" : 1,
                "nb_classes" : 13
               } # default parameters value

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

    input_img_rows = kwargs['input_img_rows']
    input_img_cols = kwargs['input_img_cols']
    channels = kwargs['channels']
    nb_classes = kwargs['nb_classes']

    
    # Input: 72 x 72 x 1
    img_shape = layers.Input(shape = (input_img_rows, input_img_cols, channels))

    # Layer 1
    #------------------------
    conv1 = layers.Conv2D(filters=32, kernel_size=(2, 2), padding='same', kernel_regularizer=regularizers.l2(l2_val))(img_shape)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = layers.Dropout(0.4)(conv1)

    # Layer 2
    #------------------------
    conv2 = layers.Conv2D(filters=64, kernel_size=(2,2), padding='same', kernel_regularizer=regularizers.l2(l2_val))(conv1)
    conv2 = layers.Activation('relu')(conv2)   
    conv2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = layers.Dropout(0.4)(conv2)

    # Layer 3
    #------------------------
    conv3 = layers.Conv2D(filters=128, kernel_size=(2,2), padding='same', kernel_regularizer=regularizers.l2(l2_val))(conv2)
    conv3 = layers.Activation('relu')(conv3)   
    conv3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3 = layers.Dropout(0.4)(conv3)

    # Layer 4
    #------------------------
    conv4 = layers.Conv2D(filters=256, kernel_size=(2,2), padding='same', dilation_rate = (2, 2), kernel_regularizer=regularizers.l2(l2_val))(conv3)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4 = layers.Dropout(0.4)(conv4)

    # Layer 5
    #------------------------
    output = layers.Conv2D(filters=128, kernel_size=(2,2), padding='same', kernel_regularizer=regularizers.l2(l2_val))(conv3) # skip layer 4
    output = layers.Activation('relu')(output) 
    output = layers.MaxPooling2D(pool_size=(2, 2))(output)
    output = layers.Dropout(0.4)(output)


            
    # FC Layer
    #------------------------
    outputmlp = layers.Flatten()(output)
    outputmlp = layers.Dense(64, activation = 'relu')(outputmlp)
    outputmlp = layers.Dropout(0.5)(outputmlp)

    predictionsMlp = layers.Dense(nb_classes, activation='softmax')(outputmlp)
    
    
    # global averaging
    weight_decay=1E-4
    concat_axis = 1
    
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=regularizers.l2(weight_decay),
                           beta_regularizer=regularizers.l2(weight_decay))(output)
    x = Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    
    predictionsGloAvg = layers.Dense(nb_classes,
                        activation='softmax',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        bias_regularizer=regularizers.l2(weight_decay))(x)
    
    if outLayer == "gloAvg":
        predictions = predictionsGloAvg
    elif outLayer == "mlp":
        predictions = predictionsMlp
        
    # prediction model
    model = Model(img_shape, predictions, name = 'cnn_max')

    return model

## Basic CNN with stride of 2 instead of max pooling    
def mgcNetArchStride2(outLayer, l2_val, **kwargs):
    
    """
    CNN architecture - without maximum pooling (replaced by convolutional layer of stride 2)
    Network architecture summary and plot
    The output layers, either multiple layer perceptron network or maximum pooling
    Return end-to-end network architecture to be compiled and trained
    
    Argumnents:
        input_img_rows: horizontal dimension in pixel of input image
        input_img_cols:vertical dimension in pixel of input image
        channels: number of colour channel
        nb_classes: number of unique classification class exist in the dataset target
    """

    def_vals = {"input_img_rows" : 72,
                "input_img_cols" : 72,
                "channels" : 1,
                "nb_classes" : 13
               } # default parameters value

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

    input_img_rows = kwargs['input_img_rows']
    input_img_cols = kwargs['input_img_cols']
    channels = kwargs['channels']
    nb_classes = kwargs['nb_classes']

    
    # Input: 72 x 72 x 1
    img_shape = layers.Input(shape = (input_img_rows, input_img_cols, channels))

    # Layer 1
    #------------------------
    conv1 = layers.Conv2D(filters=32, kernel_size=(2, 2), padding='same', kernel_regularizer=regularizers.l2(l2_val))(img_shape)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = layers.Dropout(0.4)(conv1)

    # Layer 2
    #------------------------
    conv2 = layers.Conv2D(filters=64, kernel_size=(2,2), padding='same', kernel_regularizer=regularizers.l2(l2_val))(conv1)
    conv2 = layers.Activation('relu')(conv2)   
    conv2 = layers.Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu', strides = 2)(conv2)
    conv2 = layers.Dropout(0.4)(conv2)

    # Layer 3
    #------------------------
    conv3 = layers.Conv2D(filters=128, kernel_size=(2,2), padding='same', kernel_regularizer=regularizers.l2(l2_val))(conv2)
    conv3 = layers.Activation('relu')(conv3)   
    conv3 = layers.Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu', strides = 2)(conv3)
    conv3 = layers.Dropout(0.4)(conv3)

    # Layer 4
    #------------------------
    conv4 = layers.Conv2D(filters=256, kernel_size=(2,2), padding='same', dilation_rate = (2, 2), kernel_regularizer=regularizers.l2(l2_val))(conv3)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = layers.Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu', strides = 2)(conv4)
    conv4 = layers.Dropout(0.4)(conv4)

    # Layer 5
    #------------------------
    output = layers.Conv2D(filters=128, kernel_size=(2,2), padding='same', kernel_regularizer=regularizers.l2(l2_val))(conv3) # skip layer 4
    output = layers.Activation('relu')(output)
    output = layers.Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu', strides = 2)(output)    
    output = layers.Dropout(0.4)(output)


            
    # FC Layer
    #------------------------
    outputmlp = layers.Flatten()(output)
    outputmlp = layers.Dense(64, activation = 'relu')(outputmlp)
    outputmlp = layers.Dropout(0.5)(outputmlp)

    predictionsMlp = layers.Dense(nb_classes, activation='softmax')(outputmlp)
    
    
    # global averaging
    weight_decay=1E-4
    concat_axis = 1
    
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=regularizers.l2(weight_decay),
                           beta_regularizer=regularizers.l2(weight_decay))(output)
    x = Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    
    predictionsGloAvg = layers.Dense(nb_classes,
                        activation='softmax',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        bias_regularizer=regularizers.l2(weight_decay))(x)
    
    if outLayer == "gloAvg":
        predictions = predictionsGloAvg
    elif outLayer == "mlp":
        predictions = predictionsMlp
        
    # prediction model
    model = Model(img_shape, predictions, name = 'cnn_stride2')

    return model

def mgcNetArchSkip(outLayer, l2_val, **kwargs):
    
    """
    CNN architecture - without maximum pooling (replaced by convolutional layer of stride 2)
    Network architecture summary and plot
    The output layers, either multiple layer perceptron network or maximum pooling
    Return end-to-end network architecture to be compiled and trained
    
    Argumnents:
        input_img_rows: horizontal dimension in pixel of input image
        input_img_cols:vertical dimension in pixel of input image
        channels: number of colour channel
        nb_classes: number of unique classification class exist in the dataset target
    """

    def_vals = {"input_img_rows" : 72,
                "input_img_cols" : 72,
                "channels" : 1,
                "nb_classes" : 13
               } # default parameters value

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

    input_img_rows = kwargs['input_img_rows']
    input_img_cols = kwargs['input_img_cols']
    channels = kwargs['channels']
    nb_classes = kwargs['nb_classes']

    
    # Input: 72 x 72 x 1
    img_shape = layers.Input(shape = (input_img_rows, input_img_cols, channels))

    # Layer 1
    #------------------------
    conv1 = layers.Conv2D(filters=32, kernel_size=(2, 2), padding='same', kernel_regularizer=regularizers.l2(l2_val))(img_shape)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = layers.Dropout(0.4)(conv1)

    # Layer 2
    #------------------------
    conv2 = layers.Conv2D(filters=64, kernel_size=(2,2), padding='same', kernel_regularizer=regularizers.l2(l2_val))(conv1)
    conv2 = layers.Activation('relu')(conv2)   
    conv2 = layers.Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu', strides = 2)(conv2)
    conv2 = layers.Dropout(0.4)(conv2)

    # Layer 3
    #------------------------
    conv3 = layers.Conv2D(filters=128, kernel_size=(2,2), padding='same', kernel_regularizer=regularizers.l2(l2_val))(conv2)
    conv3 = layers.Activation('relu')(conv3)   
    conv3 = layers.Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu', strides = 2)(conv3)
    conv3 = layers.Dropout(0.4)(conv3)
    
    # skip connect 1
    #shortcut_layer = layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu', strides = 4)(conv1)
    shortcut_layer = layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu', strides = 8)(img_shape)
        
    conv3 = layers.add([shortcut_layer, conv3])
    #conv3 = layers.Concatenate()([shortcut_layer,conv3])    

    # Layer 4
    #------------------------
    conv4 = layers.Conv2D(filters=256, kernel_size=(2,2), padding='same', dilation_rate = (2, 2), kernel_regularizer=regularizers.l2(l2_val))(conv3)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = layers.Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu', strides = 2)(conv4)
    conv4 = layers.Dropout(0.4)(conv4)

    # Layer 5
    #------------------------
    output = layers.Conv2D(filters=128, kernel_size=(2,2), padding='same', kernel_regularizer=regularizers.l2(l2_val))(conv3) # skip layer 4
    output = layers.Activation('relu')(output)
    output = layers.Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu', strides = 2)(output)    
    output = layers.Dropout(0.4)(output)
    
    # skip connect 2
    shortcut_layer2 = layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu', strides = 2)(conv3)
    output = layers.add([shortcut_layer2, output])
    
    # FC Layer
    #------------------------
    outputmlp = layers.Flatten()(output)
    outputmlp = layers.Dense(64, activation = 'relu')(outputmlp)
    outputmlp = layers.Dropout(0.5)(outputmlp)

    predictionsMlp = layers.Dense(nb_classes, activation='softmax')(outputmlp)
    
    
    # global averaging
    weight_decay=1E-4
    concat_axis = 1
    
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=regularizers.l2(weight_decay),
                           beta_regularizer=regularizers.l2(weight_decay))(output)
    x = Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    
    predictionsGloAvg = layers.Dense(nb_classes,
                        activation='softmax',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        bias_regularizer=regularizers.l2(weight_decay))(x)
    
    if outLayer == "gloAvg":
        predictions = predictionsGloAvg
    elif outLayer == "mlp":
        predictions = predictionsMlp
        
    # prediction model
    model = Model(img_shape, predictions, name = 'skipconnect')

    return model


def mgcResnet(block_type, block_repeat, **kwargs):
    
    """
    CNN architecture - without maximum pooling (replaced by convolutional layer of stride 2)
    Network architecture summary and plot
    The output layers, either multiple layer perceptron network or maximum pooling
    Return end-to-end network architecture to be compiled and trained
    
    Argumnents:
        input_img_rows: horizontal dimension in pixel of input image
        input_img_cols:vertical dimension in pixel of input image
        channels: number of colour channel
        nb_classes: number of unique classification class exist in the dataset target
    """
    
    
    def_vals = {"input_img_rows" : 72,
                "input_img_cols" : 72,
                "channels" : 1,
                "nb_classes" : 13
               } # default parameters value

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

    input_img_rows = kwargs['input_img_rows']
    input_img_cols = kwargs['input_img_cols']
    channels = kwargs['channels']
    nb_classes = kwargs['nb_classes']
    
    # block_type residual block option
    if block_type == 'basic':
        block_fn = resnet.basic_block
    elif block_type == 'bottleneck':
        block_fn = resnet.bottleneck

    model = resnet.ResnetBuilder.build((channels, input_img_rows, input_img_cols), nb_classes, block_fn, block_repeat)
    return model

##################################################################
# Residual block
##################################################################
def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y

## Basic CNN with stride of 2 instead of max pooling    
def mgcNetArchRes(outLayer, l2_val, **kwargs):
    
    """
    CNN architecture - without maximum pooling (replaced by convolutional layer of stride 2)
    Network architecture summary and plot
    The output layers, either multiple layer perceptron network or maximum pooling
    Return end-to-end network architecture to be compiled and trained
    
    Argumnents:
        input_img_rows: horizontal dimension in pixel of input image
        input_img_cols:vertical dimension in pixel of input image
        channels: number of colour channel
        nb_classes: number of unique classification class exist in the dataset target
    """

    def_vals = {"input_img_rows" : 72,
                "input_img_cols" : 72,
                "channels" : 1,
                "nb_classes" : 13
               } # default parameters value

    for k, v in def_vals.items():
        kwargs.setdefault(k, v)

    input_img_rows = kwargs['input_img_rows']
    input_img_cols = kwargs['input_img_cols']
    channels = kwargs['channels']
    nb_classes = kwargs['nb_classes']

    
    # Input: 72 x 72 x 1
    img_shape = layers.Input(shape = (input_img_rows, input_img_cols, channels))

    # Layer 1
    #------------------------
    conv1 = layers.Conv2D(filters=32, kernel_size=(2, 2), padding='same', kernel_regularizer=regularizers.l2(l2_val))(img_shape)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = layers.Dropout(0.4)(conv1)
    
    conv1 = residual_block(img_shape, 32, _strides=(1, 1), _project_shortcut=False)

    # Layer 2
    #------------------------
    conv2 = residual_block(conv1, 32, _strides=(1, 1), _project_shortcut=False)

    # Layer 3
    #------------------------
    conv3 = residual_block(conv2, 32, _strides=(1, 1), _project_shortcut=False)
    
    # Layer 4
    # -----------------------
    #residual = residual_block(conv3, 64, _strides=(1, 1), _project_shortcut=False)

    # Layer 5
    #------------------------
    output = layers.Conv2D(filters=128, kernel_size=(2,2), padding='same', kernel_regularizer=regularizers.l2(l2_val))(conv3) # skip layer 4
    output = layers.Activation('relu')(output)
    output = layers.Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu', strides = 2)(output)    
    output = layers.Dropout(0.4)(output)


            
    # FC Layer
    #------------------------
    outputmlp = layers.Flatten()(output)
    outputmlp = layers.Dense(64, activation = 'relu')(outputmlp)
    outputmlp = layers.Dropout(0.5)(outputmlp)

    predictionsMlp = layers.Dense(nb_classes, activation='softmax')(outputmlp)
    
    
    # global averaging
    weight_decay=1E-4
    concat_axis = 1
    
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=regularizers.l2(weight_decay),
                           beta_regularizer=regularizers.l2(weight_decay))(output)
    x = Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    
    predictionsGloAvg = layers.Dense(nb_classes,
                        activation='softmax',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        bias_regularizer=regularizers.l2(weight_decay))(x)
    
    if outLayer == "gloAvg":
        predictions = predictionsGloAvg
    elif outLayer == "mlp":
        predictions = predictionsMlp
        
    # prediction model
    model = Model(img_shape, predictions, name = 'resblock')

    return model


##################################################################
# extract validation dataset balanced for test
##################################################################


#Creating a Callback subclass that stores each epoch prediction

def report_to_df(report):
    """
    Transform classifcation report text strings to dataframe
    Arguments:
        report: classification report in texts
    """
    report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)        
    return(report_df)


class prediction_history(Callback):
    """
    Capture the training history, predict at the end of each epoch and produce model metrics such as classification report and confusion matrix
    
    Arguments:
        model: model name
        batch_size: dataset sample batch size during each epoch iteration
        y_test: target class of the test dataset
        x_test: feature of the test dataset
        nb_classes: number of unique class in the target (i.e. y_test)
    
    """
    def __init__(self, **kwargs):
        def_vals = {"model": None,
                    "batch_size": 16, 
                    "x_test": None, 
                    "y_test": None, 
                    "nb_classes": None}


        for k, v in def_vals.items():
            kwargs.setdefault(k, v)

        self.modelx = kwargs['model']
        self.batch_sizex = kwargs['batch_size']
        self.x_testx = kwargs['x_test']
        self.y_testx = kwargs['y_test']
        self.nb_classes = kwargs['nb_classes']

        self.predhis = []
        self.clas_rpt = []
        self.conf_mtrx = []

    def on_epoch_end(self, epoch, logs={}):
        """
        model metrics at the end of each epoch; classification report and confusion matrix
        
        """
        self.predhis.append(self.modelx.predict(self.x_testx))
        y_true = np.argmax(self.y_testx, 1)
        y_pred = np.argmax(self.modelx.predict(self.x_testx, batch_size=self.batch_sizex, verbose=0), axis=1)
        
        self.clas_rpt.append(report_to_df(classification_report(y_true, y_pred)))
        self.conf_mtrx.append(confusion_matrix(y_true, y_pred, labels = range(self.nb_classes)))
        #self.conf_mtrx[epoch] = confusion_matrix(y_true, y_pred, labels = range(self.nb_classes))
        return self
    ##############################################################      

        
class mgcNeuralNet(object): 
    
    """
    Method to train a classifier with user defined input based on Keras (Tensorflow as backend) API.
    2 main functions within the class:
        (1) define and initiate network architecture
        (2) fit and train the model with input data

    Option to call different network architecture defined in the above

    """
        
    def __init__(self, **kwargs):
        
        """
        Initiate the keyword arguement, assign default value if parameters are not provided
        
        Arguments:
            x_train: feature in numpy array 
            y_train: associated target in numpy array 
            x_test: feature in numpy array 
            y_test: associated target in numy array 
            channel: initial inpput date channel, grey is 1, RGB is 3
            input_img_cols: 2D input array dimension
            input_img_rows: 2D input array dimension
            nb_classes: number of unique class in target; y_train and y_test
        """

        def paraChck(**kwargs):
            """
            check and validate the keyword argument input
            """
            import sys

        
            def_val = {
                        'x_train':None,
                        'y_train':None,
                        'x_test':None,
                        'y_test':None,
                        'channel':1,
                        'input_img_cols':72,
                        'input_img_rows':72,
                        'nb_classes':13,
                        'nb_epoch': 5,
                        'batch_size' : 16,
                        'dict_label' : None} #  default parameteters value

            diff = set(kwargs.keys()) - set(def_val.keys())
            if diff:
                print("Invalid args:",tuple(diff),file=sys.stderr)
                return

            def_val.update(kwargs)
            return def_val
        
        def_val = paraChck(**kwargs)
        
        class Bunch(object):
            def __init__(self, adict):
                self.__dict__.update(adict)
         
        self.x_train = def_val['x_train']
        self.y_train = def_val['y_train']
        self.x_test = def_val['x_test']
        self.y_test = def_val['y_test']
        self.channels = def_val['channel']
        self.input_img_rows = def_val['input_img_rows']
        self.input_img_cols = def_val['input_img_cols']
        self.nb_classes = def_val['nb_classes']
        self.plot_model = None
        self.model = None
        self.nb_epoch = def_val['nb_epoch']
        self.batch_size = def_val['batch_size']
        self.dict_label = def_val['dict_label']
        
        # default label dictionary if users do not provide
        values = ['label_' + str(i).zfill(2) for i in range(0,self.nb_classes)]
        keys = range(self.nb_classes) 
            
        if self.dict_label is None:
            self.dict_label = dict(zip(keys, values))
        else:
            self.dict_label = kwargs['dict_label']
        
        self.dict_factor = {v: k for k, v in self.dict_label.items()}
        
        
    def mgcNetArch(self, **kwargs):
        
        """  
        CNN architecture variant
        return specified model, model summary and model network plot
        Arguments:
            outLayer: 'gloAvg' or 'mlp' at last layer, either use global averaging or multi layer perceptron
            l2_val: l2 regularization, default 0.002
            net_architr: cnn_max, cnn_stride or net_in_net
            
        """
        
        def_vals = {"input_img_rows" : self.input_img_rows,
                    "input_img_cols" : self.input_img_cols,
                    "channels" : self.channels,
                    "nb_classes" : self.nb_classes,
                    "outLayer" : 'gloAvg', 
                    "l2_val" : 0.00, 
                    "net_architr" : 'cnn_max', 
                    "block_typex" : 'basic', 
                    "block_repeatx" : [1, 1]
                   }


        for k, v in def_vals.items():
            kwargs.setdefault(k, v)

        _input_img_rows = kwargs['input_img_rows']
        _input_img_cols = kwargs['input_img_cols']
        _channels = kwargs['channels']
        _nb_classes = kwargs['nb_classes']
        _outLayer = kwargs['outLayer']
        _l2_val = kwargs['l2_val']
        _net_architr = kwargs['net_architr']
        _block_typex = kwargs['block_typex']
        _block_repeatx = kwargs['block_repeatx']
        
        
        params = {"input_img_rows" : _input_img_rows,
                  "input_img_cols" : _input_img_cols,
                  "channels" : _channels,
                  "nb_classes" : _nb_classes
                 }
         
        print(_net_architr)
        
        if _net_architr == 'cnn_max':
            model = mgcNetArchMax(outLayer = _outLayer, l2_val = _l2_val, **params)
            
        elif _net_architr == 'cnn_stride':
            model = mgcNetArchStride2(outLayer = _outLayer, l2_val = _l2_val, **params)
            
        elif _net_architr == 'net_in_net':
            model = mgcNetArchNin(outLayer = _outLayer, l2_val = _l2_val, **params)
        
        elif _net_architr == 'resnet':
            model = mgcResnet(block_type = _block_typex, block_repeat = _block_repeatx, **params)
            
        elif _net_architr == 'resblock':
            model = mgcNetArchRes(outLayer = _outLayer, l2_val = _l2_val, **params)

        elif _net_architr == 'skipconnect':
            model = mgcNetArchSkip(outLayer = _outLayer, l2_val = _l2_val, **params)
        
        self.model = model
        self.plot_model = SVG(model_to_dot(model, show_shapes = True).create(prog='dot', format='svg'))
        #self.model_summary = model.summary() 
        
        return self


        
    def mgcTrainMod(self, **kwargs):
        
        """
        Train model.
        Arguments:
            nb_epoch: number of iteration where all sample in the dataset has been learned by the model
            batch_size: number of sample in each batchc
        
        Output:
            (1) model for when best accuracy is achieved
            (2) history log
            (3) Tensorboard files (require Tensorboard installation and initiation to monitor the hitory)
            
        """
        import datetime, time
        from keras.callbacks import Callback, ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
        from keras import optimizers
        import numpy as np
        import os
        from AdamW import AdamW


        
        # create directory if does not exist
        #os.makedirs('output/model/', exist_ok=True)
        #os.makedirs('output/graph/', exist_ok=True)
        
        def_vals = {"nb_epoch": 2,
                    "batch_size": 16,
                    "best_model": False,
                    "exp_tag": 'my_experiment_', 
                    "save_weight": False,
                    "custom_optimizer": 'AdamW', 
                    "monitor_metric" : 'val_acc'}
        
        
        for k, v in def_vals.items():
            kwargs.setdefault(k, v)

        self.nb_epoch = kwargs['nb_epoch']
        self.batch_size = kwargs['batch_size']
        self.best_model = kwargs['best_model']      
        exp_tag = kwargs['exp_tag']
        save_weight = kwargs['save_weight']
        custom_optimizer = kwargs['custom_optimizer']
        _monitor_metric = kwargs['monitor_metric']
        
        now = datetime.datetime.now()
        digitcode = exp_tag + str(now.month).zfill(2) + str(now.day).zfill(2) + str(now.hour).zfill(2)
        digitcode2 = digitcode + str(now.minute).zfill(2) + str(now.second).zfill(2)
         
        # compile model
        
        # use normal adam
        #opt = optimizers.Adam(lr=1e-3, beta_1=0.95, beta_2=0.999, epsilon=1e-08, decay=0.001)
        #self.model.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer=opt)
        
        if custom_optimizer == "AdamW":
        # use Fixing Weight Decay Regularization in Adam
            opt = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.003, batch_size=self.batch_size, samples_per_epoch=self.batch_size, epochs=self.nb_epoch)
            print("AdamW")
            
        elif custom_optimizer == "SGD":
            # Stochastic gradient descent optimizer with momentum and nesterov
            opt = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
            print("SGD")

        elif custom_optimizer == "Adam":
            # normal adam
            opt = optimizers.Adam(lr=1e-3, beta_1=0.95, beta_2=0.999, epsilon=1e-08, decay=0.001)
            print("Adam")            

        self.model.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer=opt)

        # checkpoint to save model when there is improvement
        if self.best_model is False:
            filepath = "output/model/weights-cnn_" + str(digitcode) + "-{epoch:03d}-{val_acc:.2f}.hdf5"
        else:
            filepath = "output/model/" + self.best_model
        
        if save_weight: 
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
            # Save the model architecture
            with open(filepath + "_model_architecture.json", 'w') as f:
                f.write(self.model.to_json())
        else:
            if _monitor_metric == "val_acc":
                checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=False)
            elif _monitor_metric == "val_loss":
                checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)

        # to save the training metrics into csv
        filepathcsv = "output/model/train_log_cnn_" + str(digitcode2) + ".csv"
        csv_logger = CSVLogger(filepathcsv, separator=',', append = False)  

        # to save each epoch prediction
        params = {"model": self.model, 
                  "batch_size": self.batch_size, 
                  "x_test": self.x_test, 
                  "y_test": self.y_test, 
                  "nb_classes" : self.nb_classes}
        
        pred_hist = prediction_history(**params)
        prediction_epoch = pred_hist.on_epoch_end(self)
        
        # monitor in Tensorboard training history
        tensorb = TensorBoard(log_dir="output/graph/" + str(digitcode2), histogram_freq=1, write_graph=True, write_images=True)

        # learning rate adjustment
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=30, min_lr=0.5e-6)
        #lrate=LearningRateScheduler(step_decay)
        #lrate = LearningRateScheduler(lambda epoch: 0.001 * 0.5 ** (epoch // 2)),

        # early stopping
        early_stopper = EarlyStopping(min_delta=0.001, patience=30)
        
        # output training history, reduce learning rate and early stopping
        #callbacks_list = [checkpoint, csv_logger, tensorb, lr_reducer, early_stopper]
        
        # output training history, reduce learning rate, early stopping and prediction at the end of each epoch
        callbacks_list = [checkpoint, csv_logger, tensorb, lr_reducer, early_stopper, prediction_epoch]
        #callbacks_list = [checkpoint, csv_logger, tensorb, lr_reducer]
        
            
        seed = 7
        np.random.seed(seed)
        
        start_t = time.time()
        self.history = self.model.fit(self.x_train, self.y_train, epochs=self.nb_epoch, batch_size=self.batch_size,
                            validation_data=(self.x_test, self.y_test), verbose=2, shuffle=True, callbacks=callbacks_list)

        print('Testing...')
        self.metrics = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size, verbose=2)
        
        self.prediction_epoch = prediction_epoch

        end_t = time.time()
        print("Execution time [min]: " + str((end_t - start_t)/60))
        
        print('')
        print('Test score: {0:.4f}'.format(self.metrics[0]))
        print('Test accuracy: {0:.4f}'.format(self.metrics[1]))
        return self

    #def mgcCrossValidation(self, **kwargs):
        
        
        
    def mgcLoadMod(self, **kwargs):

        """
        load the best model from the output path or use the last epoch
        Arguments:
            ld_model: boolean True to load the model defined in model_name else False to load the last epoch model
            model_name: model name which the function to fetch
        """

        def_vals = {"ld_model" : False,
                    "model_name" : 'best_model.hdf5'
                   } # default parameters value

        for k, v in def_vals.items():
            kwargs.setdefault(k, v)
    
        self.ld_model = kwargs['ld_model']
        self.model_name = kwargs['model_name']
        
        import numpy as np        
        if self.ld_model == True:
            print('best epoch')
            # load model weights
            model = self.model

            # Load checkpoint best model
            filepath="output/model/" + self.model_name

            # load model weights
            model.load_weights(filepath, by_name=False) #model.load_weights
            #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

        else:
            print('last epoch')
            model = self.model

        self.y_pred_prob = model.predict(self.x_test, batch_size=self.batch_size, verbose=0)
        self.y_pred = np.argmax(self.y_pred_prob, axis = 1)
        self.y_true = np.argmax(self.y_test, 1)
        
        return model


    def mgcEnsemble (self, outLayer = "gloAvg", l2_val = 0.002):
        
        """
        Ensemble model
        """
        cnn_params = {"input_img_rows" : self.input_img_rows,
                        "input_img_cols" : self.input_img_cols,
                        "channels" : self.channels,
                        "nb_classes" : self.nb_classes
                       } # default parameters value
                       
        cnn = mgcNetArchMax(outLayer = outLayer, l2_val = l2_val, **cnn_params)
        stride2 = mgcNetArchStride2(outLayer = outLayer, l2_val = l2_val, **cnn_params)
        nin = mgcNetArchNin(outLayer = outLayer, l2_val = l2_val, **cnn_params)
        
        # load weight
        cnn.model.load_weights('output/model/script_13c_cnn_max_SGD_100305.hdf5')
        stride2.model.load_weights('output/model/script_13c_cnn_stride_SGD_100305.hdf5')
        nin.model.load_weights('output/model/script_13c_net_in_net_SGD_100305.hdf5')

        model_en = [cnn, stride2, nin]

        def ensemble(model_en):
            
            img_shape = layers.Input(shape = (input_img_rows, input_img_cols, channels))
            
            outputs = [modelx.outputs[0] for modelx in model_en]
            y = Average()(outputs)
            
            model = Model(img_shape, y, name='ensemble')
            
            return model
            
        self.model_ensemble = ensemble(model_en)
        
        return self
        
        
    def mgcEval(self):
        
        """
        evaluate model classification report
        """
        import numpy as np
        def report_to_df(report):

            """
            function to convert classification report to dataframe (for visualisation plot)
            """

            report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
            report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)        
            return(report_df)
    
        #txt report to df
        class_rpttop1 = classification_report(self.y_true, self.y_pred)
        df_report = report_to_df(class_rpttop1)

        df_report = df_report.iloc[:self.nb_classes, :].copy()
        df_report.index = df_report.index.astype(int)
        

        # classifier prediction metrics
        def classMetrics(averagex):
            precision, recall, fscore, support = score(self.y_true, self.y_pred, average=averagex)
            
            return(
            print(''),    
            print('-------------{0:}--------------------'.format(averagex)),  
            print('precision: {0:.4f}'.format(precision)),
            print('recall: {0:.4f}'.format(recall)),
            print('fscore: {0:.4f}'.format(fscore)),
            print(''),
            print('kappa score: {0:.4f}'.format(cohen_kappa_score(self.y_true, self.y_pred))),
            print('accuracy score: {0:.4f}'.format(accuracy_score(self.y_true, self.y_pred))))
        
        def predSamp():

            correct = np.nonzero(self.y_pred==self.y_true)[0]
            incorrect = np.nonzero(self.y_pred!=self.y_true)[0]

            # quick check of the number of correct prediction from validation set
            print("")
            print("correct/total = {0: .4f}".format(len(correct)/(len(correct)+len(incorrect))))
            print("total correct sample = {0: .0f}".format(len(correct)))
            print('------------------------------------------------------------------')
            
        def classReport():
            print('----------------------------- Classfication Report -------------------------------')
            print(classification_report(pd.Series(self.y_true).map(self.dict_label), pd.Series(self.y_pred).map(self.dict_label)))
            
        self.class_rpt = pd.concat([pd.DataFrame(pd.Series(df_report.index.tolist()).map(self.dict_label), columns = ['label']), df_report], axis = 1)
        
        self.classMetricsMac = classMetrics("macro")
        self.classMetricsMic = classMetrics("micro")
        self.predSample = predSamp()
        self.class_rptTop1 = classReport()
        
        return self
    
