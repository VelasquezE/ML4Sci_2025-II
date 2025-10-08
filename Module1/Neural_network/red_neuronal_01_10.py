import numpy as np 

def sigmoid(x) :
    return 1.0 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def get_training_inputs():
    return np.array([[0, 0, 1],
                     [1, 1, 1], 
                     [1, 0, 1],
                     [0, 1, 1]])

def get_training_outputs():
    return np.array([0, 1, 1, 0]).reshape(4, 1)

N_NEURONS = 2
N_INPUTS = 3

# Each column is going to hold the weights of each neuron

np.random.seed(1)

# Weights for getting the inputs into the hidden layer
input_weights = np.random.uniform(-1.0, 1.0, size = (N_INPUTS, N_NEURONS))

# Weights for getting the neuron results into the output
neuron_weights = np.random.uniform(-1.0, 1.0, size = (N_NEURONS, 1))

NITER = 10000
ALPHA = 0.01 

input_layer = get_training_inputs()
training_outputs = get_training_outputs()

for ii in range(NITER):
    z1 = np.dot(input_layer, input_weights)
    x1 = sigmoid(z1)

    z2 = np.dot(x1, neuron_weights)
    outputs = sigmoid(z2)

    errors = outputs - training_outputs

    delta2 = errors * sigmoid_prime(outputs)
    diff_neuron_weights = np.dot(x1.T, delta2)

    delta1 = np.dot(delta2, neuron_weights.T) * sigmoid_prime(x1)
    diff_input_weights = np.dot(input_layer.T, delta1)

    input_weights -= ALPHA * diff_input_weights
    neuron_weights -= ALPHA * diff_neuron_weights

input_new = np.array([1, 0, 0]).reshape(1, 3)
out0 = sigmoid(np.dot(input_new, input_weights))
out1 = sigmoid(np.dot(out0, neuron_weights))

print(out1)
    


