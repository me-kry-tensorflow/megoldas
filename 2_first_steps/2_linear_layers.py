import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# https://stackoverflow.com/questions/57387485/how-to-choose-units-for-dense-in-tensorflow-keras
# a retegek es azokban levo neuronok hiperparameterek
# grid search vagy random search

# https://machinelearningknowledge.ai/keras-dense-layer-explained-for-beginners/
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)

# Create 3 layers
#layer1 = layers.Dense(2, activation="relu", name="layer1")
#layer2 = layers.Dense(3, activation="relu", name="layer2")
#layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
#x = tf.ones((3, 3))
#y = layer3(layer2(layer1(x)))