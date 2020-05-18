import numpy as np
from keras import utils
from keras.datasets import mnist
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split

class ProcessData:
    def __init__(self, train, test, val, num_classes=10, original_shape=(28,28), final_shape=(32,32), rgb_channel=False):
        self.num_classes = num_classes
        self.original_shape = original_shape
        self.final_shape = final_shape
        self.rgb_channel = rgb_channel
        (x_train,y_train) = train
        (x_test,y_test) = test
        (x_val,y_val) = val
        ((self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_val, self.y_val)) = map(self.reshape_dims, [train, test, val])

    def reshape_dims(self, d):
        (x,y) = d
        x = x.astype('float32')
        if self.rgb_channel is False:
            x = x.reshape(x.shape[0], self.original_shape[0], self.original_shape[1], 1)
        pad_0 = int((self.final_shape[0]-self.original_shape[0])/2)
        pad_1 = int((self.final_shape[1]-self.original_shape[1])/2)
        x = np.pad(x, ((0,0),(pad_0,pad_1),(pad_0,pad_1),(0,0)), 'constant')
        if self.rgb_channel is False:
            x = np.stack((x,)*3, axis=-1)
            x = x[:,:,:,0,:]      
        y = utils.to_categorical(y, self.num_classes)
        return x, y

def dataset_selection(name,final_shape=(32,32)):
    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, shuffle=True ,random_state=42)
        data = ProcessData((x_train, y_train), (x_test, y_test), (x_val, y_val), num_classes=10, original_shape=(x_train.shape[1],x_train.shape[2]), final_shape=final_shape, rgb_channel=False)

    if name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, shuffle=True ,random_state=42)
        data = ProcessData((x_train, y_train), (x_test, y_test), (x_val, y_val), num_classes=10, original_shape=(x_train.shape[1],x_train.shape[2]), final_shape=final_shape, rgb_channel=True)

    return data