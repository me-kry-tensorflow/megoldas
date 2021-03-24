import tensorflow as tf
from trainingResults import TrainingResults

TRUE_W = 3.0
TRUE_B = 2.0
NUM_EXAMPLES = 1000


def generate_data():
    # A vector of random x values
    x = tf.random.normal(shape=[NUM_EXAMPLES])

    # Generate some noise
    noise = tf.random.normal(shape=[NUM_EXAMPLES])

    # Calculate y
    y = x * TRUE_W + TRUE_B + noise
    return x, y


def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))


# Given a callable model, inputs, outputs, and a learning rate...
def train(model, x, y, learning_rate):

    with tf.GradientTape() as t:
        # Trainable variables are automatically tracked by GradientTape
        current_loss = loss(y, model(x))

    # Use GradientTape to calculate the gradients with respect to W and b
    dw, db = t.gradient(current_loss, [model.w, model.b])

    # Subtract the gradient scaled by the learning rate
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)


Ws, bs = [], []
epochs = range(10)


# Define a training loop
def training_loop(model, x, y):

    for epoch in epochs:  # Update the model with the single giant batch
        train(model, x, y, learning_rate=0.1)

        # Track this before I update
        Ws.append(model.w.numpy())
        bs.append(model.b.numpy())
        current_loss = loss(y, model(x))

        print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" %
              (epoch, Ws[-1], bs[-1], current_loss))

    return TrainingResults(epochs, Ws, bs)
