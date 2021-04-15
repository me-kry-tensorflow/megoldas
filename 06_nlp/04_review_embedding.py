import io

import tensorflow as tf
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

input_length = 100
word_number = 10000


def load_data():
    path_to_downloaded_file = tf.keras.utils.get_file(
        "product_review.csv",
        "https://raw.githubusercontent.com/Apress/learn-tensorflow-2/master/product_reviews_dataset.csv",
    )
    df = pd.read_csv(path_to_downloaded_file)
    df['Summary'] = df.Summary.apply(clean_data)
    X = df.Summary
    y = np.array(df.Sentiment)
    X_sequence, X_dict = get_sequence_and_dict_from(X)
    X_padded_seq = pad_sequences(X_sequence, padding='post', maxlen=input_length)
    return X_padded_seq, y, X_dict


def clean_data(text):
    text = re.sub("[^a-zA-Z]", " ", str(text))
    return re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)


def get_sequence_and_dict_from(X):
    tokenizer = Tokenizer(num_words=word_number, oov_token='xxxxxx')
    tokenizer.fit_on_texts(X)
    return tokenizer.texts_to_sequences(X), tokenizer.word_index


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_length=input_length, input_dim=word_number, output_dim=50),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def exctract_model(weights, dictionary):
    vec_file = io.open('embedding_vectors.tsv', 'w', encoding='utf-8')
    meta_file = io.open('metadata.tsv', 'w', encoding='utf-8')
    index_based_embedding = dict([(value, key) for (key, value) in X_dict.items()])
    for i in range(1, len(dictionary) - 1):
        word = index_based_embedding[i]
        embedding_weight_values = weights[i]
        meta_file.write(word + "\n")
        vec_file.write('\t'.join([str(x.numpy()) for x in embedding_weight_values]) + "\n")
        print(str(i) + ' of ' + str(len(dictionary)) + '\n')
    meta_file.close()
    vec_file.close()


if __name__ == "__main__":
    X, y, X_dict = load_data()
    model = create_model()
    model.fit(X, y, epochs=10)
    exctract_model(weights=model.layers[0].weights[0], dictionary=X_dict)
