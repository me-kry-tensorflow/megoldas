import tensorflow as tf
import matplotlib.pyplot as plt
from myModel import MyModel
from globals import loss, training_loop


# The actual line
TRUE_W = 3.0
TRUE_B = 2.0

NUM_EXAMPLES = 1000

# A vector of random x values
x = tf.random.normal(shape=[NUM_EXAMPLES])

# Generate some noise
noise = tf.random.normal(shape=[NUM_EXAMPLES])

# Calculate y
y = x * TRUE_W + TRUE_B + noise

# Plot all the data

plt.scatter(x, y, c="b")
plt.show()

model = MyModel()

# List the variables tf.modules's built-in variable aggregation.
print("Variables:", model.variables)

plt.scatter(x, y, c="b")
plt.scatter(x, model(x), c="r")
plt.show()

print("Current loss: %1.6f" % loss(y, model(x)).numpy())

print("Starting: W=%1.2f b=%1.2f, loss=%2.5f" %
      (model.w, model.b, loss(y, model(x))))

# Do the training
results = training_loop(model, x, y)

# Plot it
plt.plot(results.epochs, results.Ws, "r",
         results.epochs, results.bs, "b")

plt.plot([TRUE_W] * len(results.epochs), "r--",
         [TRUE_B] * len(results.epochs), "b--")

plt.legend(["W", "b", "True W", "True b"])
plt.show()

# Visualize how the trained model performs
plt.scatter(x, y, c="b")
plt.scatter(x, model(x), c="r")
plt.show()

print("Current loss: %1.6f" % loss(model(x), y).numpy())