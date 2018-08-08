"""
This file contains a general purpose neural network that can be used for many
applications

NOTE: Working activation fns: sigmoid, tanh, arctan, sin, gaussian, softplus
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

def arctan(x):
    """This function returns the arctan of x for NeuralNetwork"""
    return np.arctan(x)

def arctanDerivative(x):
    """This function returns the arctan derivative of x for NeuralNetwork
        (Note: Not Real Derivative)
    """
    return 1.0/(np.tan(x)**2+1.0)

def sin(x): 
    """This function returns the sine of x"""
    return np.sin(x)

def sinDerivative(x): 
    """This function returns the sine derivative of x
        (Note: Not Real Derivative)
    """
    return np.cos(np.arcsin(x))

def gaussian(x): 
    """This function returns the gaussian of x"""
    return np.exp(-x**2)

def gaussianDerivative(x): 
    """This function returns the gaussian derivative of x
        (Note: Not Real Derivative)
    """
    return -2.0*x*(np.sqrt(-np.log(x)))

def softplus(x): 
    """This function returns the softplus of x"""
    return np.log(1+np.exp(x))

def softplusDerivative(x): 
    """This function returns the softplusDerivative of x
        (Note: Not Real Derivative)
    """
    return 1.0-np.exp(-x)

def getMax(array_list):
    """Returns a tuple (index,value) of the maximum in an 1D array or list"""
    m = array_list[0]
    m_index = 0
    for i,value in enumerate(array_list):
        if value > m:
            m = value
            m_index = i
    return (m_index,m)



class NeuralNetworkException(Exception):
    """ General Purpose Exception for class NeuralNetwork"""
    def __init___(self,message):
        Exception.__init__(self,message)
        
class NeuralNetwork:
    """ General Purpose Neural Network"""
    activation_dict = {'sigmoid': [sigmoid,sigmoidDerivative],
                       'tanh': [tanh,tanhDerivative],
                       'arctan': [arctan,arctanDerivative],
                       'sin': [sin,sinDerivative],
                       'gaussian': [gaussian,gaussianDerivative],
                       'softplus': [softplus,softplusDerivative],
                       }
    def __init__(self,layers,activeFn = 'sigmoid'):
        """layers is list of layer lengths"""

        # get activation function
        self.setActivationFn(activeFn)
        
        self.layers = layers
        self.theta = []
        # Generate weight matrix with random weights -1 to 1
        for i in range(1,len(layers)):
            if i != len(layers)-1: 
                weight_matrix = 2*np.random.rand(layers[i-1] + 1, layers[i] + 1) -1
            else:
                weight_matrix = 2*np.random.rand(layers[i-1] + 1, layers[i]) -1
            self.theta.append(weight_matrix)

    def setActivationFn(self,activeFn):
        """Sets the activation function"""
        if activeFn in NeuralNetwork.activation_dict:
            self.activeFn = NeuralNetwork.activation_dict[activeFn][0]
            self.activeFnDerivative = NeuralNetwork.activation_dict[activeFn][1]
        else:
            raise ValueError('Invalid activation Function')        
        
    def getLayers(self):
        """Return a list of layer lengths"""
        return self.layers
    
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
        with open(filename) as json_file:  
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

    def trainWithPlots(self,X,Y,learning_rate = 1.0,intervals = 100,thres_high = 0.8,thres_low =0.5):
        """Trains the neural networks with a list of input vectors x
           and a list of output vectors Y.
           Plots Cost function over each iteration.
           Iterates in Sequential Order.
        """     
        J = []
        training_rate = []
        m = 0.0
        m_array = []
        for _ in range(intervals):
            for i in range(len(X)):
                if self.trainTestSample(X[i],Y[i],thres_high,thres_low) == True:
                    m += 1.0
                m_array.append(m)
                J.append( self.lossFunction(X[i],Y[i]) )
        m_array = np.array(m_array)
        n = len(X)*intervals
        n_array = [i+1 for i in range(n)]
        n_array = np.array(n_array)
        training_rate = np.divide(m_array,n_array)
        plt.plot(J)
        plt.ylabel('Cost')
        plt.xlabel('Training Sample')
        plt.title('Cost Function vs. Number of Training Samples')
        plt.show()
        plt.plot(training_rate)
        plt.ylabel('Average Training Rate')
        plt.xlabel('Training Sample')
        plt.title('Average Training Rate vs. Number of Training Samples')
        plt.show()
        print('Final Aggregate Average Training Rate = ' + str(training_rate[-1]))
        n_last = 100
        if n > n_last:
            m_final = m_array[n-n_last:]
            n_final = n_array[n-n_last:]
            training_final = np.sum(np.divide(m_final,n_final))/n_last
            print('Training rate of last ' + str(n_last) + ' data points = ' + str(training_final))
            
    
    def trainSample(self,x,y,learning_rate=1.0):
        """Trains the neural networks with a single input vector x
           and a single output vector y"""
        a = self.forwardProp(x)
        self.backProp(a,y,learning_rate)

    def trainTestSample(self,x,y,learning_rate=1.0,thres_high = 0.8,thres_low =0.5):
        """Trains the neural networks with a single input vector x
            and a single output vector y.
            Returns true if the Tested Sample is predicted correctly by
            the neural network before backpropagation.
            Prediction if target probabibility is above threshold, and
            
            Assumes that the target vector has only one positive value.
        """
        a = self.forwardProp(x)
        self.backProp(a,y,learning_rate)
        max_index, max_value = getMax(a[-1])
        for i in range(len(y)):
            if y[i] >= 0.5 and a[-1][i] < thres_high:
                return False
            elif y[i] < 0.5 and a[-1][i] > thres_low:
                return False
        return True
        
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
    net = NeuralNetwork([2,2,1],'tanh')

    X = [[0,0],[1,0],[0,1],[1,1]];
    Y = [[0],[1],[1],[0]]
    #J = []

    net.trainWithPlots(X,Y,learning_rate=0.2,intervals=100)
    for i in range(len(Y)):
        y_out = net.predict(X[i])
        print('Test Case: ' + str(X[i]) + ', Result: ' + str(y_out) )
    net.printWeights()

    
if __name__ == '__main__':
    main()






































'''
List of unimplemented Activation Functions, perhaps to be used in a future update.

def bent(x): # DOES NOT WORK WITH NN
    """This function returns the bent identity of x"""
    return (sqrt(x**2 + 1.0) - 1.0)/2.0 + x

def bentDerivative(x): # DOES NOT WORK WITH NN
    """This function returns the bent identity derivative of x
        (Note: Not Real Derivative)
    """
    return x/(2.0*sqrt(x**2 + 1.0)) + 1.0

def silu(x): # DOES NOT WORK WITH NN
    """This function returns the sigmoid-weight linear unit of x"""
    return x*sigmoid(x)

def siluDerivative(x): # DOES NOT WORK WITH NN
    """This function returns the silu derivative of x
        (Note: Not Real Derivative)
    """
    return x + (np.log(x)-np.log(1.0-x))*(1.0+x)
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
def softsign(x): # DOES NOT WORK WITH NN
    """This function returns the softsign of x"""
    return x/(1.0+abs(x))

def softsignDerivative(x): # DOES NOT WORK WITH NN
    """This function returns the softsign derivative of x
        (Note: Not Real Derivative)
    """
    return 1.0/(1+abs(x))**2
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
'''
