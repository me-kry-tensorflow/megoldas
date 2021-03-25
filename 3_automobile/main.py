# https://www.tensorflow.org/tutorials/keras/regression
# https://www.kaggle.com/toramky/automobile-dataset

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns

from automobile import load_data, split_data_train_test, linear_regression


def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10000])
    plt.xlabel('Epoch')
    plt.ylabel('Error [price]')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_horsepower(training_x, training_y, x, y):
    #  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.figure()
    plt.scatter(training_x, training_y, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('price')
    plt.legend()
    plt.show()


def linear_regression_one_input():
    data_frame = load_data()
    sns.heatmap(data_frame.corr(), annot=True)
    train_features, train_labels, test_features, test_labels = split_data_train_test(data_frame, train_size=0.8)

    hp_index = data_frame.columns.get_loc("horsepower")

    horsepower = train_features[:, hp_index]
    horsepower = np.asarray(horsepower).astype(np.float32)

    train_labels_1 = np.asarray(train_labels).astype(np.float32)

    horsepower_model, history = linear_regression(horsepower, train_labels_1)

    plot_loss(history)

    horsepower_model.predict(horsepower[:10])

    test_results = {}

    horsepower_test = np.asarray(test_features[:, hp_index]).astype(np.float32)
    test_labels_1 = np.asarray(test_labels).astype(np.float32)

    test_results['horsepower_model'] = horsepower_model.evaluate(
        horsepower_test,
        test_labels_1, verbose=2)

    x = tf.linspace(0.0, 250, 251)
    y = horsepower_model.predict(x)

    plot_horsepower(horsepower, train_labels_1, x, y)

    horsepower_model.save('saved_model/my_model')

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()


if __name__ == "__main__":
    linear_regression_one_input()


# e szerint is rossz: https://www.kaggle.com/vovanthuong/predict-automobile-price