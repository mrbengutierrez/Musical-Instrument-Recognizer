"""
This file contains a general purpose neural network that can be used for many
applications

NOTE: for the activation functions: only sigmoid, tanh, and arctan, sin currently work
"""

import numpy as np # necessary unless want to rewrite
import matplotlib.pyplot as plt # only necessary for plot functions
import json #only necessary for storing/loading weights from file


def sigmoid(x):
    """ This function computes the sigmoid of x for NeuralNetwork"""
    return 1.0/(1.0 + np.exp(-x))

def sigmoidDerivative(x):
    """ This function computes the sigmoid derivative of x for NeuralNetwork
        (Note: Not Real Derivative)
    """
    return x*(1.0-x)

def tanh(x):
    """ This function computes the tanh of x for NeuralNetwork"""
    return np.tanh(x)

def tanhDerivative(x):
    """ This function computes the tanh derivative of x for NeuralNetwork
        (Note: Not Real Derivative)
    """
    return 1.0 - x**2

def linear(x): # WORKS HORRIBLY
    """ This function returns x"""
    return x

def linearDerivative(x): # WORKS HORRIBLY
    """ This function returns 1
        (Note: Not Real Derivative)
    """
    if type(x) == type(int) or type(x) == type(float):
        return 1.0
    else:
        output = [1.0 for _ in x]
        return np.array(output)

def arctan(x):
    """This function returns the arctan of x for NeuralNetwork"""
    return np.arctan(x)

def arctanDerivative(x):
    """This function returns the arctan derivative of x for NeuralNetwork
        (Note: Not Real Derivative)
    """
    return 1.0/(np.tan(x)**2+1.0)

def relu(x): # DOES NOT WORK WITH NN
    """This function returns the reLu of x
        (Note: Not Real Derivative)
    """
    if type(x) == type(int) or type(x) == type(float):
        if x < 0.0:
            return 0.0
        return x
    output = []
    for value in x:
        if value < 0.0:
            output.append(0.0)
        else:
            output.append(value)
    return np.array(output)

def reluDerivative(x): # DOES NOT WORK WITH NN
    """This function returns the reLu derivative of x
        (Note: Not Real Derivative)
    """
    if type(x) == type(int) or type(x) == type(float):
        if x < 0.0:
            return 0.0
        return 1.0
    output = []
    for value in x:
        if value < 0.0:
            output.append(0.0)
        else:
            output.append(1.0)
    return np.array(output)

def sinc(x): # DOES NOT WORK WITH NN
    """This function returns the sinc of x"""
    epsilon = 10^-4
    if type(x) == type(int) or type(x) == type(float):
        if x < epsilon and x > -epsilon: # if x == 0
            return 1.0
        return np.sin(x)/x
    output = []
    for value in x:
        if value < epsilon and value > -epsilon: # if value == 0
            output.append(1.0)
        else:
            output.append(np.sin(value)/value)
    return np.array(output)

def sincDerivative(x): # DOES NOT WORK WITH NN
    """This function returns the sinc derivative of x
        (Note: Not Real Derivative)
    """
    epsilon = 10^-4
    if type(x) == type(int) or type(x) == type(float):
        if x < epsilon and x > -epsilon: # if x == 0
            return 0.0
        return np.cos(x)/x - np.sin(x)/(x**2)
    output = []
    for value in x:
        if value < epsilon and value > -epsilon: # if value == 0
            output.append(0.0)
        else:
            output.append(np.cos(value)/value - np.sin(value)/(value**2))
    return np.array(output)

def sin(x): 
    """This function returns the sine of x"""
    return np.sin(x)

def sinDerivative(x): 
    """This function returns the sine derivative of x
        (Note: Not Real Derivative)
    """
    return np.cos(np.arcsin(x))

def binary(x): # DOES NOT WORK WITH NN
    """This function returns the binary step of x"""
    if type(x) == type(int) or type(x) == type(float):
        if x <0.0:
            return 0.0
        return 1.0
    output = []
    for value in x:
        if value < 0.0:
            output.append(0.0)
        else:
            output.append(1.0)
    return np.array(output)

def binaryDerivative(x): # DOES NOT WORK WITH NN
    """This function returns the binary step derivative of x
        (Note: Not Real Derivative)
    """
    epsilon = 10^-4
    if type(x) == type(int) or type(x) == type(float):
        if x < epsilon and x > -epsilon: # if x == 0
            return 10^8
        return 0.0
    output = []
    for value in x:
        if value < epsilon and value > -epsilon: # if value == 0
            output.append(0.0)
        else:
            output.append(10^8)
    return np.array(output)

def softsign(x): # DOES NOT WORK WITH NN
    """This function returns the softsign of x"""
    return x/(1.0+abs(x))

def softsignDerivative(x): # DOES NOT WORK WITH NN
    """This function returns the softsign derivative of x
        (Note: Not Real Derivative)
    """
    return 1.0/(1+abs(x))**2

def gaussian(x): # DOES NOT WORK WITH NN
    """This function returns the guassian of x"""
    return np.exp(-x**2)

def gaussianDerivative(x): # DOES NOT WORK WITH NN
    """This function returns the gaussian derivative of x
        (Note: Not Real Derivative)
    """
    return -2.0*x*np.exp(-x**2)

def softplus(x): # DOES NOT WORK WITH NN
    """This function returns the softplus of x"""
    return np.log(1+np.exp(x))

def softplusDerivative(x): # DOES NOT WORK WITH NN
    """This function returns the softplusDerivative of x
        (Note: Not Real Derivative)
    """
    return 1.0/(1.0+np.exp(-x))

def bent(x): # DOES NOT WORK WITH NN
    """This function returns the bent identity of x"""
    return (sqrt(x**2 + 1.0) - 1.0)/2.0 + x

def bentDerivative(x): # DOES NOT WORK WITH NN
    """This function returns the bent identity derivative of x
        (Note: Not Real Derivative)
    """
    return x/(2.0*sqrt(x**2 + 1.0)) + 1.0


class NeuralNetworkException(Exception):
    """ General Purpose Exception for class NeuralNetwork"""
    def __init___(self,message):
        Exception.__init__(self,message)
        
class NeuralNetwork:
    """ General Purpose Neural Network"""
    activation_dict = {'sigmoid': [sigmoid,sigmoidDerivative],
                       'tanh': [tanh,tanhDerivative],
                       'linear':[linear,linearDerivative],
                       'arctan': [arctan,arctanDerivative],
                       'relu': [relu,reluDerivative],
                       'sinc': [sinc,sincDerivative],
                       'sin': [sin,sinDerivative],
                       'binary': [binary,binaryDerivative],
                       'softsign': [softsign,softsignDerivative],
                       'guassian': [gaussian,gaussianDerivative],
                       'softplus': [softplus,softplusDerivative],
                       'bent': [bent,bentDerivative]
                       }
    def __init__(self,layers,activeFn = 'sigmoid'):
        """layers is list of layer lengths"""

        # get activation function
        if activeFn in NeuralNetwork.activation_dict:
            self.activeFn = NeuralNetwork.activation_dict[activeFn][0]
            self.activeFnDerivative = NeuralNetwork.activation_dict[activeFn][1]
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
    neuralXorTest()

    #net = NeuralNetwork([2,2,1])
    #net.storeWeights('data.txt', comment = 'Hi, World')
    #net.loadWeights('data.txt')

def neuralXorTest():
    net = NeuralNetwork([2,2,1],'sin')

    X = [[0,0],[1,0],[0,1],[1,1]];
    Y = [[0],[1],[1],[0]]
    J = []

    net.trainWithPlots(X,Y,learning_rate=0.2,intervals=1000)
    for i in range(len(Y)):
        y_out = net.predict(X[i])
        print('Test Case: ' + str(X[i]) + ', Result: ' + str(y_out) )
    net.printWeights()

    
if __name__ == '__main__':
    main()
