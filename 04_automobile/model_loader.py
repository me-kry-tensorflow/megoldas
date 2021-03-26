import tensorflow as tf
if __name__ == "__main__":
    new_model = tf.keras.models.load_model('saved_model/my_model')
    print(new_model.predict([111]))
