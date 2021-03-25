import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import seaborn as sns

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
    horsepower_normalizer = preprocessing.Normalization(input_shape=[1, ])
    horsepower_normalizer.adapt(feature)

    horsepower_model = tf.keras.Sequential([
        horsepower_normalizer,
        tf.keras.layers.Dense(units=1)
    ])

    horsepower_model.summary()

    horsepower_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

    history = horsepower_model.fit(
        feature, train_labels,
        epochs=100,
        # suppress logging
        verbose=0,
        # Calculate validation results on 20% of the training data
        validation_split=0.2)
    return horsepower_model, history


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)



# https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#selection

data_frame = pd.read_csv('~/.data/Automobile_data.csv')

data_frame = replace_question_mark(data_frame)

train_features, train_labels, test_features, test_labels = split_data_train_test(data_frame, train_size=0.8)

# normalizer = preprocessing.Normalization()

# normalizer.adapt(train_features)
# print(normalizer.mean.numpy())

hp_index = data_frame.columns.get_loc("horsepower")

horsepower = train_features[:, hp_index]
horsepower = np.asarray(horsepower).astype(np.float32)
train_labels_1 = np.asarray(train_labels).astype(np.float32)

horsepower_model, history = linear_regression_one_input(horsepower, train_labels_1)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plot_loss(history)

horsepower_model.predict(horsepower[:10])

test_results = {}

horsepower_test = np.asarray(test_features[:, hp_index]).astype(np.float32)
test_labels_1 = np.asarray(test_labels).astype(np.float32)

test_results['horsepower_model'] = horsepower_model.evaluate(
    horsepower_test,
    test_labels_1, verbose=0)

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)


def plot_horsepower(x, y):
#  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()


plot_horsepower(x, y)

data_frame.head()
data_frame.info()
plt.figure(figsize=(12, 6))
data_frame.columns
# sns.heatmap(data_frame.corr(), annot=True)
data_frame.isnull().sum()

# plt.figure(figsize=(20,6))
# sns.countplot(x='fuel-type',data=data)


