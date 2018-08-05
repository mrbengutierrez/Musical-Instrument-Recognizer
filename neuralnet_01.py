"""
This file contains a general purpose neural network that can be used for many
applications
"""

import numpy as np # necessary unless want to rewrite
import matplotlib.pyplot as plt # only necessary for plot functions
import json #only necessary for storing/loading weights from file

def sigmoid(x):
    """ This function computes the sigmoid of x"""
    return 1.0/(1.0 + np.exp(-x))

def sigmoidDerivative(x):
    """ This function computes the sigmoid derivative of x"""
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    """ This function computes the tanh of x"""
    return np.tanh(x)

def tanhDerivative(x):
    """ This function computes the tanh derivative of x"""
    return 1.0 - x**2

def linear(x):
    """ This function returns x"""
    return x

def linearDerivative(x):
    """ This function returns 1"""
    return 1.0

class NeuralNetworkException(Exception):
    """ General Purpose Exception for class NeuralNetwork"""
    def __init___(self,message):
        Exception.__init__(self,message)
        
class NeuralNetwork:
    """ General Purpose Neural Network"""
    def __init__(self,layers,activeFn = 'sigmoid'):
        """layers is list of layer lengths"""

        if activeFn == 'sigmoid':
            self.activeFn = sigmoid
            self.activeFnDerivative = sigmoidDerivative
        elif activeFn == 'tanh':
            self.activeFn = tanh
            self.activeFnDerivative = tanhDerivative
        elif activeFn == 'linear':
            self.activeFn = linear
            self.activeFnDerivative = linearDerivative
        else:
            raise ValueError('Invalid activation Function')
        
        self.layers = layers
        self.theta = []
        for i in range(1,len(layers)):
            if i != len(layers)-1: 
                weight_matrix = 2*np.random.rand(layers[i-1] + 1, layers[i] + 1) -1
            else:
                weight_matrix = 2*np.random.rand(layers[i-1] + 1, layers[i]) -1
            self.theta.append(weight_matrix)

    def storeWeights(self,filename,comment = 'No Comment'):
        """Stores Weights in filename.
            filename (string): ex. 'data.txt'
            comment (string): message to be stored in file
        """
        # Store weights as lists
        stored_weights = []
        for i in range(len(self.theta)):
            stored_weights.append([])
            for j in range(len(self.theta[i])):
                stored_weights[i].append([])
                for value in self.theta[i][j]:
                    stored_weights[i][j].append(float(value))
        print(stored_weights)
        data = {}
        data['theta'] = stored_weights
        data['layers'] = self.layers
        data['comment'] = comment
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)
        return
    
    def loadWeights(self,filename):
        """Loads Weights in filename. Note WILL overwrite layer shape and number.
            filename (string): ex. 'data.txt'
        """
        with open('data.txt') as json_file:  
            data = json.load(json_file)
            
        # weight matrices are stored as lists, so turn them into np.arrays
        self.theta = []
        for weight_matrix in data['theta']:
            self.theta.append(np.array(weight_matrix))         
        self.layers = data['layers']
        print()
        print('Weights Loaded.')
        print('Layers: ' + str(self.layers))
        print('Comment: ' + str(data['comment']))
        return
      
        
    def trainRandom(self,X,Y,learning_rate=1.0,intervals = 100):
        """Trains the neural networks with a list of input vectors x
           and a list of output vectors Y. Iterates in Random Order.
        """        
        for _ in range(intervals):
            for _ in range(len(X)):
                r = np.random.choice(len(X))
                self.trainSample(X[r],Y[r])

    def trainSequential(self,X,Y,learning_rate=1.0,intervals = 100):
        """Trains the neural networks with a list of input vectors x
           and a list of output vectors Y. Iterates in Sequential Order.
        """        
        for _ in range(intervals):
            for i in range(len(X)):
                self.trainSample(X[i],Y[i])

    def trainWithPlots(self,X,Y,learning_rate = 1.0,intervals = 100):
        """Trains the neural networks with a list of input vectors x
           and a list of output vectors Y.
           Plots Cost function over each iteration.
           Iterates in Sequential Order.
        """     
        J = []
        for _ in range(intervals):
            for i in range(len(X)):
                self.trainSample(X[i],Y[i])
                J.append( self.lossFunction(X[i],Y[i]) )
        plt.plot(J)
        plt.ylabel('Cost')
        plt.xlabel('Training Sample')
        plt.title('Cost Function vs. Number of Training Samples')
        plt.show()
    
    def trainSample(self,x,y,learning_rate=1.0):
        """Trains the neural networks with a single input vector x
           and a single output vector y"""
        a = self.forwardProp(x)
        self.backProp(a,y,learning_rate)
        
    def forwardProp(self,x):
        """Forward Propagates x through the Neural Network"""
        x = np.array(x)
        a = [np.append([1],x)]

        for l in range(len(self.theta)):
            inner = np.dot(a[l],self.theta[l])
            a.append( self.activeFn( inner) )
        return a
    
    def backProp(self,a,y,learning_rate):
        """Backward propagates y through the Neural Network using activations a"""
        y = np.array(y)
        delta = y - a[-1]
        deltas = [delta * self.activeFnDerivative(a[-1])]

        for layer in range(len(a)-2,0,-1):
            deltas.append(deltas[-1].dot(self.theta[layer].T)*self.activeFnDerivative(a[layer]))
        deltas.reverse()

        for i in range(len(self.theta)):
            active_layer = np.atleast_2d(a[i])
            delta = np.atleast_2d(deltas[i])
            self.theta[i] += learning_rate * active_layer.T.dot(delta)
            
    def predict(self,x):
        """Predicts an output vector for a given input vector x"""
        return self.forwardProp(x)[-1]
    
    def lossFunction(self,x,y):
        """Computes the loss function for a given input vector and output vector"""
        a = self.forwardProp(x)
        h = a[-1]
        J = 0
        for i in range(len(h)):
            #J += (-1/self.layers[0]) * (y[i]*np.log(h[i])+(1-y[i])*np.log(1-h[i]))
            J += 0.5*(h[i]-y[i])**2
        return J           

    def printWeights(self):
        """Prints all of the weight matrices"""
        for i in range(len(self.theta)):
            self.printWeight(i)
            print()
            
    def printWeight(self,i):
        """Prints the weight matrix at index i of self.theta"""
        print('theta(' + str(i) + ') =  ')
        for row in self.theta[i]:
            print(str(row))            

        
        


def main():
    #neuralXorTest()

    net = NeuralNetwork([2,2,1])
    net.storeWeights('data.txt', comment = 'Hi, World')
    net.loadWeights('data.txt')

def neuralXorTest():
    net = NeuralNetwork([2,2,1],'tanh')

    X = [[0,0],[1,0],[0,1],[1,1]];
    Y = [[0],[1],[1],[0]]
    J = []

    net.trainWithPlots(X,Y,intervals=10000)
    for i in range(len(Y)):
        y_out = net.predict(X[i])
        print('Test Case: ' + str(X[i]) + ', Result: ' + str(y_out) )
    net.printWeights()

    
if __name__ == '__main__':
    main()
