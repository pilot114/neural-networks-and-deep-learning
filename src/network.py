# -*- coding: utf-8 -*-

import random
import numpy as np

#  персептрон - совокупность нейронов, принимающая решения
#  нейроны с сигмоидноидной функцией активации (и с аналоговым выходом) легче подстраивать
# к тому же, значения маштабируются в [0; 1]

# Функция потерь - функция, оценивающая веса и пороги С(w,b). Чем она меньше, тем лучше.
# Т.к. она прямо зависит от весов и порогов, она лучше для оценки, чем просто
# кол-во правильных ответов.

# Градиентный спуск - метод, позволяющий искать наименьшее значение функции потерь.
# Градиент - это вектор частных производных функции потерь к каждому параметру сети.
# в совокупности с переменной "скорость обучения", он даёт направление и скорость спуска к минимуму функции потерь.
# Если скорость большая, мы можем пропустить минимум, если маленькая - обучаться будем долго.

# Стохастический градиентный спуск - вычисление спуска не для каждого объекта или всей выборки,
# а среднего для небольшой случайной выборки, что ускорит процесс (не осторожный точный спуск, а быстро спускающийся, шатающийся пьяный)

# Если скрытых слоев много - это глубокие нейронные сети

# Модуль для реализации стохастического обучения градиентным спуском
# для прямой нейронной сети. Градиенты вычисляются используя backpropagation
class Network(object):

    # задаем кол-во нейронов в слоях. пример: [2,4,5 ...]
    # смещения и веса задаются случайно с мат.ожиданием=0 и дисперсией=1
    # для первого слоя не задаются
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # смещения ("пороги") - у каждого нейрона, кроме входных
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # веса - связи между нейронами
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # обучение сети (тут передаем ГИПЕРПАРАМЕТРЫ сети). training_data - кортежи из учебных входов и желаемых выводов.
    # test_data можно передать для "самооценки" (существенно замедляет работу)
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    # обновление смещений и весов используя backpropagation для одной мини-партии
    # функция потерь для батча высчитывается как среднее арифметическое функций потерь для каждого примера
    # eta - скорость обучения
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    # самая важная часть - backpropagation - алгоритм быстрого поиска градиента
    # https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
    # (x - вход, y - ожидаемый выход)
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass -  разницу между полученым значением и ожидаемым (ф.потерь) умножаем на производную функции активации
        # http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # подставляем разницу в последний слой
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    # функция потерь
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    # возвращает кол-во тестов с правильными результатами
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # фактически, это и есть работа сети - просто прогонка от входа к выходу
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# производная функции активации
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))