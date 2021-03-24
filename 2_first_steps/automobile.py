import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
from tensorflow.keras import preprocessing


def replace_question_with_mean(data):
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



# https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#selection

data_frame = pd.read_csv('~/.data/Automobile_data.csv')

data_frame = replace_question_with_mean(data_frame)

X_train, X_test, y_train, y_test = split_data_train_test(data_frame, train_size=0.8)

normalizer = preprocessing.Normalization()


data_frame.head()
data_frame.info()
plt.figure(figsize=(12, 6))
data_frame.columns
# sns.heatmap(data_frame.corr(), annot=True)
data_frame.isnull().sum()

# plt.figure(figsize=(20,6))
# sns.countplot(x='fuel-type',data=data)


