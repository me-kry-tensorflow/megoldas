from src.from_cratch.perceptron import perceptron, predict, weights

epochs = range(100000)

for epoch in epochs:
    perceptron(1, 1, 1)  # True or true
    perceptron(1, 0, 1)  # True or false
    perceptron(0, 1, 1)  # False or true
    perceptron(0, 0, 0)  # False or false
print(weights)

print(predict(1, 0))

print(predict(0, 0))
