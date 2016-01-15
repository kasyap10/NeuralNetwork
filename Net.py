import numpy as np
import math
import time
import gzip
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plot
import os

#Add for sure: ReLU (Max), L2, MSE (not ideal but not much other choice), Mini_batch matrix
#Maybe: Momentum, Dropout
#Future: Softmax + ReLU, perhaps?
def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open(os.getcwd() + '/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = np.array(list(zip(training_inputs, training_results)))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = np.array(list(zip(validation_inputs, va_d[1])))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = np.array(list(zip(test_inputs, te_d[1])))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

class NeuralNetwork:
    def __init__(self, layers, learningrate=0, lmb=0, sample_size=0, training_data=0, epochs=0, probability=0.5):
        self.layers = layers
        self.learningrate = learningrate
        self.sample_size = sample_size
        self.lmb = lmb
        self.epochs = epochs
        self.probability = probability
        self.weights = [.1*np.random.randn(layers[i+1], layers[i]) for i in range(0, len(layers) - 1)]
        self.biases = [np.random.randn(layers[i]) for i in range(1, len(layers))] #Problematic?

        #for j in range(0,epochs) :
         #   np.random.shuffle(training_data)
          #  for i in len(training_data):
           #     tdarray = np.array(training_data[i:i+sample_size])
            #    self.train(tdarray)

    def run(self, input, randM = None):
        activations = []
        activations.append(input)
        for i in range(0, len(self.weights)):
            activations.append(self.max(activations[i]@np.transpose(self.weights[i]) + self.biases[i]))
            if randM is not None :
                activations[i] *= randM[i]
        return activations

    def train(self, td):
        inputs = np.array([example[0] for example in td])
        outputs = np.array([example[1] for example in td])
        randM = [np.random.binomial(1,self.probability,size=(np.shape(self.layers[i-1],self.layers[i]))) for i in range(len(self.layers),1,-1)]
        activations = np.array(self.run(inputs), randM)
        #Below part is where bias gradient is calculated
        del_b = np.array()
        del_b.append((1/activations[-1].size)*(activations[-1]-outputs[-1])*self.max_prime(activations[-1])) #Bias change of output with Hadamard, hopefully?
        del_b.extend([(np.transpose(self.weights[i])@np.tile(del_b[i].transpose(), (self.layers[i], 1)))* randM[i] for i in range(self.layers,1,-1)]) #bias for hidden layers...what to do about input layer? Also extend vs append
        #remember to divide sum by number of NONZERO entries
        #now for the weights
        #Multiply activation matrix by del_b matrix, hopefully the zeros in del_b will replace the Hadamard 1's and 0's matrix
        del_w = [activations[i]@del_b[i] for i in range(self.layers,1,-1)] #don't forget to divide by # of nonzero weights
        #TODO: Divide del_w by the number of nonzero entries to get avg gradient
        #TODO: Update the weights and biases, taking into account L2
    def sigmoid(self,z):
        f = [1/(1+np.e**-i) for i in z]
        return f
    def max(self, z):
        f = .5 * (z + abs(z))
        return f
    def max_prime(self,z):
        if z > 0 :
            return 1
        else :
            return 0
    def softplus(self, z):
        f = np.log(np.exp(z) + 1)
        return f
    def cross_entropy(self, o, e):
        return (1/o.size)*np.sum(np.nan_to_num(-e*np.log(o)-(1-e)*np.log(1-o)))
    def mean_squared_error(self, o, e):
        return (.5/o.size)*np.sum((o-e)^2)
    def generate_rand_matrix(self, arr, p):
        matrix = [[1 if np.random.rand() < p else 0 for elem in row] for row in np.zeros[arr.shape]]
        return matrix
    def write(self, link):
        np.savetxt(link, self.biases[0], delimiter=",")



#print (n.weights)
#print (n.biases)

#startime = time.time()
#output = n.Run(np.array([0.0, 1.0]))
#print (output)
#print ("Time: " + str(time.time() - startime))
t, v, test = load_data_wrapper()

#plot.imshow(t[1].reshape((28,28)), cmap=cm.Greys_r)
#plot.show()
#print('The input' + str(t[0][0]))
n = NeuralNetwork([2,2,3])
#print(n.run(np.array([[0.0,1.0],[1.0,0.0],[1.0,1.0]])))
#print("Biases:")
#print([np.transpos# e(bias) for bias in n.biases])
n.write('file.csv')