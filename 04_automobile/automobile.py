import pandas as pd

import numpy as np
import tensorflow as tf
# import seaborn as sns
import datetime


def replace_question_mark(data):
    column_to_fix = ['normalized-losses', 'horsepower', 'peak-rpm', 'bore', 'stroke']

    for column_name in column_to_fix:
        df_temp = data[data[column_name] != '?']
        normalised_mean = df_temp[column_name].astype(np.float32).mean()
        data[column_name] = data[column_name].replace('?', normalised_mean).astype(np.float32)

    data['num-of-doors'] = data['num-of-doors'].replace('?', 'four')

    data = data.drop(data[data.price == '?'].index)

    return data


def split_data_train_test(frame, train_size):
    count = frame.shape[0]
    frame.sample(frac=1)

    data_frame_x = frame.drop('price', axis=1)
    x = np.array(data_frame_x)
    data_frame_y = frame['horsepower']
    y = np.array(data_frame_y)

    train_count = int(count * train_size)

    train_x, test_x = np.split(x, [train_count])
    train_y, test_y = np.split(y, [train_count])

    return train_x, train_y, test_x, test_y


def linear_regression(feature, train_labels):
    input_number = 1 if len(feature.shape) == 1 else feature.shape[1]

    if input_number == 1:
        horsepower_model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=input_number, input_shape=[input_number]),
        ])
    else:
        horsepower_model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=input_number, input_shape=[input_number]),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])


    horsepower_model.summary()

    horsepower_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.2),
        loss='mean_absolute_error'
        # loss='mean_squared_error'
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=(10, 15))

    history = horsepower_model.fit(
        feature, train_labels,
        epochs=200,
        # suppress logging
        verbose=2,
        callbacks=[tensorboard_callback],
        # Calculate validation results on 20% of the training data
        validation_split=0.2,
    )
    return horsepower_model, history


def load_data():
    data_frame = pd.read_csv('~/.data/Automobile_data.csv')

    return replace_question_mark(data_frame)

# https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#selection




