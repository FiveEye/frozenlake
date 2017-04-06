import numpy as np
import keras as ks
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling3D
from keras.layers.merge import concatenate
import keras.backend as K


def VIN(sz, k, ch_i, ch_h, ch_q, ch_a):
    map_in = Input(shape=(sz,sz))
    s1 = Input(shape=(1,), dtype='int32')
    s2 = Input(shape=(1,), dtype='int32')
    h = Conv2D(filters=ch_h, 
               kernel_size=(3,3), 
               padding='same', 
               activation='relu')(map_in)
    r = Conv2D(filters=1, 
               kernel_size=(3,3), 
               padding='same',
               bias=False,
               activation=None,
               )(h)
    conv3 = Conv2D(filters=l_q, 
                   kernel_size=(3, 3), 
                   padding='same',
                   bias=False)

    conv3b = Conv2D(filters=l_q, 
                   kernel_size=(3, 3), 
                   padding='same',
                   bias=False)
    
    q = conv3(r)

    for _ in range(k):
        #v = Lambda(lambda x: K.max(x, axis=CHANNEL_AXIS, keepdims=True)),
        #           output_shape=(sz,sz,1))(q)
        v = MaxPooling3D(pool_size=(1,1,ch_q))(q)
        rv = concatenate([r, v], axis=3)
        q = conv3b(rv)

    q_out = attention(q,s1,s2) 

    out = Dense(ch_a, activation='softmax', bias=False)(q_out)
