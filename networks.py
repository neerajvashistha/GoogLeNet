from keras import optimizers
from keras import backend as K
from keras import layers
from keras.datasets import mnist
from keras.datasets import cifar10
import numpy as np


def inception(x, filters):
	# 1x1
	conv1 = layers.Conv2D(filters=filters[0], 
						kernel_size=(1,1), 
						strides=1, 
						padding='same', 
						activation='relu')(x)

	# 1x1->3x3
	conv2 = layers.Conv2D(filters=filters[1][0], 
						kernel_size=(1,1), 
						strides=1, 
						padding='same', 
						activation='relu')(x)
	conv2 = layers.Conv2D(filters=filters[1][1], 
						kernel_size=(3,3), 
						strides=1, 
						padding='same', 
						activation='relu')(conv2)
	
	# 1x1->5x5
	conv3 = layers.Conv2D(filters=filters[2][0], 
						kernel_size=(1,1), 
						strides=1, 
						padding='same', 
						activation='relu')(x)
	conv3 = layers.Conv2D(filters=filters[2][1], 
						kernel_size=(5,5), 
						strides=1, 
						padding='same', 
						activation='relu')(conv3)

	# 3x3->1x1
	conv4 = layers.MaxPooling2D(pool_size=(3,3), 
						strides=1, 
						padding='same')(x)
	conv4 = layers.Conv2D(filters=filters[3], 
						kernel_size=(1,1), 
						strides=1, 
						padding='same', 
						activation='relu')(conv4)

	return layers.Concatenate(axis=-1)([conv1,conv2,conv3,conv4])


def auxiliary(x, num_classes,name=None):
	x = layers.AveragePooling2D(pool_size=(5,5), 
						strides=3, 
						padding='valid')(x)
	x = layers.Conv2D(filters=128, 
						kernel_size=(1,1), 
						strides=1, 
						padding='same', 
						activation='relu')(x)
	x = layers.Flatten()(x)
	x = layers.Dense(units=256, 
						activation='relu')(x)
	x = layers.Dropout(0.4)(x)
	x = layers.Dense(units=num_classes, 
						activation='softmax', name=name)(x)
	return x


def googlenet_layers(layer_in,num_classes,dropout_rate=0.4,activation_ch='softmax'):

	# stage-1
	x = layers.Conv2D(filters=64,
						kernel_size=(7,7),
						strides=2,
						padding='same',
						activation='relu')(layer_in)
	x = layers.MaxPooling2D(pool_size=(3,3),
						strides=2,
						padding='same')(x)
	x = layers.BatchNormalization()(x)

	# stage-2
	x = layers.Conv2D(filters=64,
						kernel_size=(1,1),
						strides=1,
						padding='same',
						activation='relu')(x)
	x = layers.Conv2D(filters=192,
						kernel_size=(3,3),
						strides=1,
						padding='same',
						activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling2D(pool_size=(3,3),
						strides=2,
						padding='same')(x)

	# stage-3
	x = inception(x, [ 64,  (96,128), (16,32), 32]) #3a
	x = inception(x, [128, (128,192), (32,96), 64]) #3b
	x = layers.MaxPooling2D(pool_size=(3,3), 
						strides=2, 
						padding='same')(x)
	
	# stage-4
	x = inception(x, [192,  (96,208),  (16,48),  64]) #4a
	aux1  = auxiliary(x, num_classes, name='aux1')
	x = inception(x, [160, (112,224),  (24,64),  64]) #4b
	x = inception(x, [128, (128,256),  (24,64),  64]) #4c
	x = inception(x, [112, (144,288),  (32,64),  64]) #4d
	aux2  = auxiliary(x, num_classes, name='aux2')
	x = inception(x, [256, (160,320), (32,128), 128]) #4e
	x = layers.MaxPooling2D(pool_size=(3,3), 
						strides=2, 
						padding='same')(x)
	
	# stage-5
	x = inception(x, [256, (160,320), (32,128), 128]) #5a
	x = inception(x, [384, (192,384), (48,128), 128]) #5b
	x = layers.AveragePooling2D(pool_size=(7,7), 
						strides=1, 
						padding='valid')(x)
	
	# stage-6
	x = layers.Flatten()(x)
	x = layers.Dropout(dropout_rate)(x)
	x = layers.Dense(units=256, 
					activation='linear')(x)
	main = layers.Dense(units=num_classes, 
					activation=activation_ch, 
					name='main')(x)
	
	return main, aux1, aux2


def select_optimiser(opt,learning_rate):
    if opt=='sgd':
        opt_type =  optimizers.SGD(lr=learning_rate, 
                                         decay=1e-6, 
                                         momentum=0.9, 
                                         nesterov=True)
    if opt=='adagrad':
        opt_type =  optimizers.Adagrad(lr=learning_rate, 
                                             epsilon=1e-08)
    if opt=='adam':
        opt_type = optimizers.Adam(lr=learning_rate, 
                                         beta_1=0.9, 
                                         beta_2=0.999, 
                                         epsilon=1e-08, 
                                         amsgrad=False)
    return opt_type

def evaluate_test(ypred,ytrue):
	true_val = []
	error_val = []
	for i in range(len(ypred)):
		if np.argmax(ypred[i]) == np.argmax(ytrue[i]):
			true_val.append(i)
		else:
			error_val.append(i)
	return true_val, error_val