# https://www.kaggle.com/zygmunt/goodbooks-10k
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as tf

from globals import create_model, extract_embeddings

ratings_df = pd.read_csv("~/book-data/ratings.csv")
books_df = pd.read_csv("~/book-data/books.csv")

ratings_df.head()

books_df.head()

print(ratings_df.shape)
print(ratings_df.user_id.nunique())
print(ratings_df.book_id.nunique())
ratings_df.isna().sum()


Xtrain, Xtest = train_test_split(ratings_df, test_size=0.2, random_state=1)
print(f"Shape of train data: {Xtrain.shape}")
print(f"Shape of test data: {Xtest.shape}")

nbook_id = ratings_df.book_id.nunique()
nuser_id = ratings_df.user_id.nunique()

model = create_model(nbook_id, nuser_id)

opt = tf.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='mean_squared_error')
model.summary()


hist = model.fit([Xtrain.book_id, Xtrain.user_id], Xtrain.rating,
                 batch_size=64,
                 epochs=5,
                 verbose=1,
                 validation_data=([Xtest.book_id, Xtest.user_id], Xtest.rating))

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
plt.plot(train_loss, color='r', label='Train Loss')
plt.plot(val_loss, color='b', label='Validation Loss')
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.show()

model.save('model')


extract_embeddings(model, books_df, ratings_df)

# Making recommendations for user 100
b_id = list(ratings_df.book_id.unique())
b_id.remove(10000)
book_arr = np.array(b_id)  # get all book IDs
user = np.array([100 for i in range(len(b_id))])
pred = model.predict([book_arr, user])
pred

pred = pred.reshape(-1)  # reshape to single dimension
pred_ids = (-pred).argsort()[0:5]
pred_ids

books_df.iloc[pred_ids]

web_book_data = books_df[["book_id", "title", "image_url", "authors"]]
web_book_data = web_book_data.sort_values('book_id')
web_book_data.head()

web_book_data.to_json(r'web_book_data.json', orient='records')
