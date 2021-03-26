import tensorflow as tf

TRUE_W = 3.0
TRUE_B = 2.0
NUM_EXAMPLES = 1000


def get_data():
    """
    Generalj egy majdnem linearis adathalmazt (`x`, `y`). A fuggveny ezt a ket tombot adja vissza.
    :return: x, y
    """
    # A vector of random x values
    x = tf.random.normal(shape=[NUM_EXAMPLES])

    # Generate some noise
    noise = tf.random.normal(shape=[NUM_EXAMPLES])

    # Calculate y
    y = x * TRUE_W + TRUE_B + noise
    return x, y


def loss(target_y, predicted_y):
    """
    A valosagos es szamolt eremeny ertekek kulonbsegek gyokeinek atlaga.
    :rtype: a tenyleges es szamolt kimenet veszteseg erteke
    """
    return tf.reduce_mean(tf.square(target_y - predicted_y))


# Given a callable model, inputs, outputs, and a learning rate...
def train(model, x, y, learning_rate):
    """
    A tanitas egy itaracioja. szamolja a kulonbseget, korrigalja a model w es b parametereit
    :param model: modositando modell
    :param x: NUM_EXAMPLES meretu tomb, a betanito halmaz bementei
    :param y: NUM_EXAMPLES meretu tomb, a betanito halmaz kimenetei
    :param learning_rate: tanulasi rata 0< es 1>
    """
    with tf.GradientTape() as t:
        # Trainable variables are automatically tracked by GradientTape
        current_loss = loss(y, model(x))

    # Use GradientTape to calculate the gradients with respect to W and b
    dw, db = t.gradient(current_loss, [model.w, model.b])

    # Subtract the gradient scaled by the learning rate
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)


# Define a training loop
# fv aminek bemenete NUM_EXAMPLES meretu tomb

def training_loop(model, x, y):
    """
    Ciklusban meghivja a tanulo fvt
    :param model: a model objektum, amit tanitani kell
    :param x: NUM_EXAMPLES meretu tomb, a betanito halmaz bementei
    :param y: NUM_EXAMPLES meretu tomb, a betanito halmaz kimenetei
    :return: epochs, Ws, bs
    """
    Ws, bs = [], []
    epochs = range(10)
    for epoch in epochs:  # Update the model with the single giant batch
        train(model, x, y, learning_rate=0.1)

        # Track this before I update
        Ws.append(model.w.numpy())
        bs.append(model.b.numpy())
        current_loss = loss(y, model(x))

        print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" %
              (epoch, Ws[-1], bs[-1], current_loss))

    return epochs, Ws, bs
