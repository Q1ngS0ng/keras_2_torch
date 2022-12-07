from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Input, MaxPool2D, concatenate, Conv2D
from keras.datasets import mnist

class pytorch_Net(nn.Module):
    def __init__(self):
        super(pytorch_Net, self).__init__()
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        self.conv2d = nn.Conv2d(1, 36, 5, 1)
        self.conv2d_1 = nn.Conv2d(36, 32, 5, 1)
        self.conv2d_2 = nn.Conv2d(36, 32, 5, 1)

        self.dense = nn.Linear(4 * 4 * 64, 110)
        self.dense_1 = nn.Linear(110, 110)
        self.dense_2 = nn.Linear(110, 10)


    def forward(self, x):

        x = F.relu(self.conv2d(x))
        x = F.max_pool2d(x, 2, 2)
        x1 = F.relu(self.conv2d_1(x))
        x2 = F.relu(self.conv2d_2(x))
        x = torch.cat((x1, x2), dim=1)
        x4 = F.max_pool2d(x, 2, 2)
        x = x4.permute((0, 2, 3, 1))
        x3 = x.contiguous().view(-1, 4 * 4 * 64)
        x = F.relu(self.dense(x3))
        x = F.relu(self.dense_1(x))
        x = self.dense_2(x)
        x = F.softmax(x, dim=1)
        return x

def keras_Net():
    input = Input(shape=(28, 28, 1), name='input')
    x = Conv2D(36, 5, activation="relu")(input)
    x = MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x1 = Conv2D(32, 5, activation="relu")(x)
    x2 = Conv2D(32, 5, activation="relu")(x)
    x = concatenate([x1, x2])
    x3 = MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = Flatten()(x3)
    x = Dense(110, activation="relu")(x)
    x = Dense(110, activation="relu")(x)
    output = Dense(10, activation="softmax")(x)
    model = Model(inputs=[input], outputs=[output])
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta())

    return model

def keras_to_pyt(km, pm):
    weight_dict = dict()
    for layer in km.layers:
        if type(layer) is keras.layers.convolutional.Conv2D:
            weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
            weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.Dense:
            if layer.get_config()['name'] == 'dense':
                w = layer.get_weights()[0]
                weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(w, (1, 0))
                weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
            else:
                weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (1, 0))
                weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
    pyt_state_dict = pm.state_dict()
    for key in pyt_state_dict.keys():
        pyt_state_dict[key] = torch.from_numpy(weight_dict[key])
    pm.load_state_dict(pyt_state_dict)
    return pm

def coinstance_eval(keras_network, pytorch_network):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    test_num = 10
    inp = np.expand_dims(x_train[0:test_num], axis=1)
    print('inp.shape', inp.shape)

    inp_keras = np.transpose(inp.copy(), (0, 2, 3, 1))
    inp_pyt = torch.autograd.Variable(torch.from_numpy(inp.copy()).float())
    pyt_res = pytorch_network(inp_pyt).data.numpy()
    keras_res = keras_network.predict(x=inp_keras, verbose=1)
    for i in range(test_num):
        predict1 = np.argmax(pyt_res[i])
        predict2 = np.argmax(keras_res[i])
        if predict1 != predict2:
            print("ERROR: Two model ooutput are different!")
        else:
            print("Yuwan: Two model ooutput are same!")
        if predict1 != y_train[i]:
            print("The model predict for {}th image is wrong".format(i + 1))



def k2p():
    # 定义需要的模型
    keras_network = keras_Net()
    pytorch_network = pytorch_Net()

    # 加载Keras模型
    keras_network.load_weights("../original_models/keras_model.h5")

    # 将keras模型转化为pytorch模型
    pytorch_network = keras_to_pyt(keras_network, pytorch_network)

    # 保存转换后的模型
    torch.save(pytorch_network.state_dict(), "../target_models/pyt_model.pt")

    # 对比一致性
    coinstance_eval(keras_network, pytorch_network)

if __name__ == "__main__":
    k2p()