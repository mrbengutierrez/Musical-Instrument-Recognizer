import numpy as np
#import matplotlib as plt
#import scipy

def sigmoid(x):
    """ This function computes the sigmoid of x (np.array)"""
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
    """ This function computes the sigmoid derivative of x (np.array)"""
    return x * (1.0 - x)

class NeuralNetworkException(Exception):
    """ General Purpose Exception for class NeuralNetwork"""
    def __init___(self,message):
        Exception.__init__(self,message)
    
class NeuralNetwork:
    def __init__(self, lengths):
        """ Initializes the neural network
            lengths (list): list of layer lengths, must have length >= 3
        """
        self.lengths = lengths

        # weights: index 1 = layer index, index 2 = feature index
        # last index in each layer is bias weight
        self.weights = np.array()
        self.activations = np.array()
        for i,length in enumerate(self.lengths):
            self.weights[i] = np.random.rand(1,length+1)
            self.activations[i] = np.random.rand(1,length+1)

    def train(self,x,y):
        """ Trains neural network
            x (list): input training vector
            y (list): output training vector
        """
        # check to make sure training case has layer lengths
        if len(x) !=  self.lengths[0]:
            message = 'input training vector length does not match length of NeuralNetwork input vector'
            raise NeuralNetworkException(message)
        if len(y) != self.lengths[-1]:
            message = 'output training vector length does not match length of NeuralNetwork input vector'
            raise NeuralNetworkException(message)

        self.input = x
        print('x = '+ str(x))
        print('x.shape = ' + str(x.shape[1]))
        self.weights1   = np.random.rand(self.input.shape[1],4)
        print('self.weights1 = ' + str(self.weights1))
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        
        a = self.feedforward(x)
        self.backprop(a,y)
    def feedforward(self,x):
        """ Propagate input vector x across weights to find activations"""
        
        a = []  #list of activation vectors
        for i,weight in enumerate(self.weights):
            if i == 0:
                a[i] = sigmoid(np.dot(weight,x))
            else:
                a[i] = sigmoid(np.dot(weight,a[i-1]))
        
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self,a,y):
        """Use chain rule to find derivative of loss function of weights"""

        deltas = []
        for in range(self.weights):
            if i == 0:
                delta[0] = a[-1]-y
            else:
                delta[i] =  self.weights[-1].T
        #delta_weights = []
        #for i in range(self.weights):
            #if i == 0:
                #delta_weights[0] = np.dot(a[-2].T, (2*(y-a[-1]) * sigmoid_derivative(a[-1]) ))
            #else:
                #delta_weights[i] = np.dot(a[-i-2].T, 2*(a[-i]-a[-i-1])*sigmoid_derivative(a[-i-1]))
            
        #d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        #d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        #self.weights1 += d_weights1
        #self.weights2 += d_weights2
        for i in self.weights:
            self.weights[i] = self.weights[i] += delta_weights[-i]


if __name__ == "__main__":
    raise NeuralNetworkException('cat in the house')
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(nn.output)
