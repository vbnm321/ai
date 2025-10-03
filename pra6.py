# pra:6 AIM: Implement Naive-Bayes learning algo for RWP(Restaurant Waiting Problem).


import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        # Initialize weights randomly with mean 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of sigmoid function
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Training process
    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    # Think / feedforward function
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("Beginning randomly generated weights: ")
    print(neural_network.synaptic_weights)

    # Training dataset
    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])
    training_outputs = np.array([[0, 1, 1, 0]]).T

    # Train the neural network
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending weights after training: ")
    print(neural_network.synaptic_weights)

    # User input
    user_input_one = float(input("User Input One: "))
    user_input_two = float(input("User Input Two: "))
    user_input_three = float(input("User Input Three: "))

    print("Considering new situation: ", user_input_one, user_input_two, user_input_three)
    print("New output data: ")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
