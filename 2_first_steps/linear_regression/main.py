import matplotlib.pyplot as plt
from myModel import MyModel
from globals import loss, training_loop, generate_data, TRUE_W, TRUE_B

if __name__ == "__main__":
    x, y = generate_data()
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
