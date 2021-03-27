# https://www.tensorflow.org/tutorials/keras/classification

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


def plot_image(image_array):
    plt.figure()
    plt.imshow(image_array)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def plot_all_classes_images(train_images, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
plot_image(train_images[0])

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 28x28 pixel images
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)  # 0..9 the class id
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

model.fit(train_images, train_labels, epochs=5)


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plot_all_classes_images(train_images, class_names)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc ', test_acc)

img = (np.expand_dims(test_images[1], 0))
prediction = model.predict(img)
class_index = np.argmax(prediction)
print('prediction of 1 test image is ' + class_names[class_index])





