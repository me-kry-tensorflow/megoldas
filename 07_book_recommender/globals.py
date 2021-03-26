import tensorflow as tf


def create_model(nbook_id, nuser_id):
    # Book input network
    input_books = tf.keras.layers.Input(shape=[1])
    embed_books = tf.keras.layers.Embedding(nbook_id + 1, 15)(input_books)
    books_out = tf.keras.layers.Flatten()(embed_books)

    # user input network
    input_users = tf.keras.layers.Input(shape=[1])
    embed_users = tf.keras.layers.Embedding(nuser_id + 1, 15)(input_users)
    users_out = tf.keras.layers.Flatten()(embed_users)

    conc_layer = tf.keras.layers.Concatenate()([books_out, users_out])
    x = tf.keras.layers.Dense(128, activation='relu')(conc_layer)
    x_out = x = tf.keras.layers.Dense(1, activation='relu')(x)
    return tf.keras.Model([input_books, input_users], x_out)


def extract_embeddings(model, books_df, ratings_df):
    # Extract embeddings
    book_em = model.get_layer('embedding')
    book_em_weights = book_em.get_weights()[0]
    book_em_weights.shape

    books_df_copy = books_df.copy()
    books_df_copy = books_df_copy.set_index("book_id")

    b_id = list(ratings_df.book_id.unique())
    b_id.remove(10000)
    dict_map = {}
    for i in b_id:
        dict_map[i] = books_df_copy.iloc[i]['title']

    out_v = open('vecs.tsv', 'w')
    out_m = open('meta.tsv', 'w')
    for i in b_id:
        book = dict_map[i]
    embeddings = book_em_weights[i]
    out_m.write(book + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")

    out_v.close()
    out_m.close()

