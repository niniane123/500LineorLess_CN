import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy import matrix
from math import pow
from collections import namedtuple
import math
import random
import os
import json

"""
This class does some initial training of a neural network for predicting drawn
digits based on a data set in data_matrix and data_labels. It can then be used to
train the network further by calling train() with any array of data or to predict
what a drawn digit is by calling predict().

The weights that define the neural network can be saved to a file, NN_FILE_PATH,
to be reloaded upon initilization.
"""


class OCRNeuralNetwork:
    LEARNING_RATE = 0.1
    WIDTH_IN_PIXELS = 20
    NN_FILE_PATH = 'nn.json'

    def __init__(self, num_hidden_nodes, data_matrix, data_labels, training_indices, use_file=True):
        # 构建实例对象并且赋值给self.sigmoid_scalar，其实就是得到了一个函数，再去调用传参就行了；
        self.sigmoid = np.vectorize(self._sigmoid_scalar)
        # 构建实例对象并且赋值给self._sigmoid_prime_scalar ,其实得到了一个函数，再去调用传参就行
        self.sigmoid_prime = np.vectorize(self._sigmoid_prime_scalar)

        self._use_file = use_file
        self.data_matrix = data_matrix
        self.data_labels = data_labels
        # 如果不使用file作为原始数据的话,随即的初始化神经网络中的权重；
        if (not os.path.isfile(OCRNeuralNetwork.NN_FILE_PATH) or not use_file):
            # Step 1: Initialize weights to small numbers，我们只需要知道这些变量都是向量、矩阵即可；对应到编程
            #语言中来就是说列表，一维的列表或者是二维的列表都可以 ；

            # 随机初始化各种因子为各种形状的矩阵或者数组；
            self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
            self.theta2 = self._rand_initialize_weights(num_hidden_nodes, 10)
            self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
            self.hidden_layer_bias = self._rand_initialize_weights(1, 10)

            # Train using sample data 使用简单数据开始训练
            TrainData = namedtuple('TrainData', ['y0', 'label'])
            # 一层遍历是拿到一个矩阵中一行数据；第二层遍历才是拿到具体的某一个值，一层遍历都是拿到一个一个得分列表，列表里面就是一条数据的，但是有多个维度；
            # 构造TrainData实例对象，存放y0和训练标签的数据； 存放的本质上是tupple对象
            # 用数组存放整个训练集的数据和标签，并且开始训练；

            # 训练的时候是一行一行的数据作为一个样本，以及一个一个的标签值作为样本去训练
            self.train([TrainData(self.data_matrix[i], int(self.data_labels[i])) for i in training_indices])
            self.save()
        else:
            #从nn.json中读取数据并且存放在self对象中
            self._load()

    def _rand_initialize_weights(self, size_in, size_out):
        return [((x * 0.12) - 0.06) for x in np.random.rand(size_out, size_in)]

    # The sigmoid activation function. Operates on scalars.
    def _sigmoid_scalar(self, z):
        return 1 / (1 + math.e ** -z)

    def _sigmoid_prime_scalar(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def _draw(self, sample):
        pixelArray = [sample[j:j + self.WIDTH_IN_PIXELS] for j in xrange(0, len(sample), self.WIDTH_IN_PIXELS)]
        plt.imshow(zip(*pixelArray), cmap=cm.Greys_r, interpolation="nearest")
        plt.show()

    def train(self, training_data_array):
        for data in training_data_array:
            # Step 2: Forward propagation          Returns the transpose of the matrix.
            # data其实是一个具名的元组，可以通过属性名（类似类成员）来访问其中的任意一个元素。

            # 感觉这边的语法不太对data.y0
            y1 = np.dot(np.mat(self.theta1), np.mat(data["y0"]).T)
            sum1 = y1 + np.mat(self.input_layer_bias)  # Add the bias
            y1 = self.sigmoid(sum1)

            y2 = np.dot(np.array(self.theta2), y1)
            y2 = np.add(y2, self.hidden_layer_bias)  # Add the bias
            y2 = self.sigmoid(y2)

            # Step 3: Back propagation
            actual_vals = [0] * 10  # actual_vals is a python list for easy initialization and is later turned into an np matrix (2 lines down).
            actual_vals[data['label']] = 1
            output_errors = np.mat(actual_vals).T - np.mat(y2)
            hiddenErrors = np.multiply(np.dot(np.mat(self.theta2).T, output_errors), self.sigmoid_prime(sum1))

            # Step 4: Update weights
            self.theta1 += self.LEARNING_RATE * np.dot(np.mat(hiddenErrors), np.mat(data['y0']))
            self.theta2 += self.LEARNING_RATE * np.dot(np.mat(output_errors), np.mat(y1).T)
            self.hidden_layer_bias += self.LEARNING_RATE * output_errors
            self.input_layer_bias += self.LEARNING_RATE * hiddenErrors

    def predict(self, test):
        y1 = np.dot(np.mat(self.theta1), np.mat(test).T)
        y1 = y1 + np.mat(self.input_layer_bias)  # Add the bias
        y1 = self.sigmoid(y1)

        y2 = np.dot(np.array(self.theta2), y1)
        y2 = np.add(y2, self.hidden_layer_bias)  # Add the bias
        y2 = self.sigmoid(y2)

        results = y2.T.tolist()[0]
        return results.index(max(results))

    # 存盘
    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "theta1": [np_mat.tolist()[0] for np_mat in self.theta1],
            "theta2": [np_mat.tolist()[0] for np_mat in self.theta2],
            "b1": self.input_layer_bias[0].tolist()[0],
            "b2": self.hidden_layer_bias[0].tolist()[0]
        };
        with open(OCRNeuralNetwork.NN_FILE_PATH, 'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    # 加载数据
    def _load(self):
        if not self._use_file:
            return

        with open(OCRNeuralNetwork.NN_FILE_PATH) as nnFile:
            # 得到一个字典类型
            nn = json.load(nnFile)
        #     得到一个列表，列表里面嵌套着列表,其实里面每个元素都是一个长度为400的列表，一共长为15个元素；
        self.theta1 = [np.array(li) for li in nn['theta1']]
        self.theta2 = [np.array(li) for li in nn['theta2']]
        self.input_layer_bias = [np.array(nn['b1'][0])]
        self.hidden_layer_bias = [np.array(nn['b2'][0])]
