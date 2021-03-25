import pandas as pd

import numpy as np
import tensorflow as tf
# import seaborn as sns
import datetime

import tensorflow.keras.layers.experimental.preprocessing as preprocessing


def replace_question_mark(data):
    df_temp = data[data['normalized-losses'] != '?']
    normalised_mean = df_temp['normalized-losses'].astype(int).mean()
    data['normalized-losses'] = data['normalized-losses'].replace('?', normalised_mean).astype(int)

    df_temp = data[data['horsepower'] != '?']
    normalised_mean = df_temp['horsepower'].astype(int).mean()
    data['horsepower'] = data['horsepower'].replace('?', normalised_mean).astype(int)

    df_temp = data[data['peak-rpm'] != '?']
    normalised_mean = df_temp['peak-rpm'].astype(int).mean()
    data['peak-rpm'] = data['peak-rpm'].replace('?', normalised_mean).astype(int)

    df_temp = data[data['bore'] != '?']
    normalised_mean = df_temp['bore'].astype(float).mean()
    data['bore'] = data['bore'].replace('?', normalised_mean).astype(float)

    df_temp = data[data['stroke'] != '?']
    normalised_mean = df_temp['stroke'].astype(float).mean()
    data['stroke'] = data['stroke'].replace('?', normalised_mean).astype(float)

    data['num-of-doors'] = data['num-of-doors'].replace('?', 'four')

    data = data.drop(data[data.price == '?'].index)

    return data


def split_data_train_test(frame, train_size):
    count = frame.shape[0]
    frame.sample(frac=1)

    data_frame_x = frame.drop('price', axis=1)
    x = np.array(data_frame_x)
    data_frame_y = frame['price']
    y = np.array(data_frame_y)

    train_count = int(count * train_size)

    train_x, test_x = np.split(x, [train_count])
    train_y, test_y = np.split(y, [train_count])

    return train_x, train_y, test_x, test_y


def linear_regression_one_input(feature, train_labels):

    normalizer = preprocessing.Normalization(input_shape=[1, ])
    normalizer.adapt(feature)

    horsepower_model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(units=1)
        # tf.keras.layers.Dense(units=1, input_shape=[1])
    ])

    horsepower_model.summary()

    horsepower_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error'
        # loss='mean_squared_error'
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = horsepower_model.fit(
        feature, train_labels,
        epochs=100,
        # suppress logging
        verbose=2,
        callbacks=tensorboard_callback,
        # Calculate validation results on 20% of the training data
        validation_split=0.2)
    return horsepower_model, history


def load_data():
    data_frame = pd.read_csv('~/.data/Automobile_data.csv')

    return replace_question_mark(data_frame)

# https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#selection




