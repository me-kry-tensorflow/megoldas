# logikai vagy muvelet neuralis haloval
# input1 OR input2 -> output1
# 0 OR 1 -> 1
# 1 OR 0 -> 1
# 1 OR 1 -> 1
# 0 OR 0 -> 0

import numpy
import random

learning_rate = 0.01
bias = 1
# veletlen sulyok, ketto a neuronokhoz, 1 a bias
weights = [random.random(), random.random(), random.random()]


def heaviside(calculated_output):
    return 1 if calculated_output else 0


def sigmoid(calculated_output):
    return 1 / (1 + numpy.exp(-calculated_output))


def calculate(input1, input2):
    calculated_output = input1 * weights[0] + input2 * weights[1] + bias * weights[2]
    return sigmoid(calculated_output)


def perceptron(input1, input2, output) -> None:
    calculated_output = calculate(input1, input2)

    error = output - calculated_output
    weights[0] += error * input1 * learning_rate
    weights[1] += error * input2 * learning_rate
    weights[2] += error * bias * learning_rate


def predict(input1, input2) -> int:
    return calculate(input1, input2)
