import numpy as np 

def sigmoid(x) :
    return 1.0/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def get_training_inputs():
    return np.array([[0, 0, 1],
                     [1, 1, 1], 
                     [1, 0, 1],
                     [0, 1, 1]])

def get_training_outputs():
    return np.array([0, 1, 1, 0]).reshape(4, 1)

def get_init_weights(n):
    """
    Initially, simply return random weights in [-1, 1)
    """
    return np.random.uniform(-1.0, 1.0, size=(n, 1))

def train_nn(training_inputs, training_outputs, initial_weights: np.ndarray, niter, errors_data, alpha = 1.0):
    """
    training_inputs: asdasdasda
    ...
    errors_data: output - stores the errors per iteration
    """
    w1, w2, w3, w4 = initial_weights

    for ii in range(niter):
        # Forward propagation
        input_layer = training_inputs
        first_neuron = sigmoid(np.dot(input_layer, w1))
        second_neuron = sigmoid(np.dot(input_layer, w2))

        neurons = np.array([first_neuron, second_neuron])
        neuron_weights = np.array([w3, w4])

        outputs = sigmoid(np.dot(neurons, neuron_weights))

        # Backward propagation
        errors = outputs - training_outputs
        deltaw = errors*sigmoid_prime(outputs)
        deltaw = np.dot(input_layer.T, deltaw)
        w = w - alpha*deltaw
        # Save errors for plotting later
        errors_data[ii] = errors.reshape((4,))
    return outputs, w

np.random.seed(1) # what happens if you comment this?
inputs_t = get_training_inputs()
outputs_t = get_training_outputs()
weights = get_init_weights()

NITER = 500
errors = np.zeros((NITER, 4))
outputs, weights = train_nn(inputs_t, outputs_t, weights, NITER, errors, alpha=0.9)
print("Training outputs:")
print(outputs_t)
print("Results after training:")
print(outputs)
print(weights)