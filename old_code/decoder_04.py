import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ This function computes the sigmoid of x (np.array)"""
    return 1.0/(1 + np.exp(-x))

def sigmoidDerivative(x):
    """ This function computes the sigmoid derivative of x (np.array)"""
    return sigmoid(x) * (1.0 - sigmoid(x))

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
        self.theta = [] # self.theta contains a list of weight matrices,
                        # index corresponds to layer

        #num of weight matrices is num_layers - 1
        for i in range(len(lengths)-1):              
            weight_matrix = -np.random.rand(lengths[i+1],lengths[i]+1)
            self.theta.append(weight_matrix)

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

    def train(self,x,y):
        
        a = self.feedForward(x)
        self.backProp(a,y)
        pass

    def testCase(self,x):
        return self.feedForward(x)[-1]
    
    def feedForward(self,x):
        a = [] # list of activation vectors
        for i in range(len(self.theta)):
            if i == 0:
                x = np.append([1],x) # append 1 to front of vector for bias
                a.append( sigmoid(np.dot(self.theta[i],x)) )
            else:
                a[i-1] = np.append([1],a[i-1]) # append 1 to front of vector for bias
                a.append( sigmoid(np.dot(self.theta[i],a[i-1])) )
        return a
    
    def backProp(self,a,y):
        learning_rate = 1.0
        reg_term = 0
        deltas = []

        # calculate deltas
        for i in range(len(self.theta)):
            if i == 0:
                deltas.append( (a[-1]-y)*sigmoidDerivative(a[-1]) )
                #print('deltas[0] = ' + str(deltas[0]))
            else:
                #self.printWeight(-i)
                #print('self.theta['+str(-i)+' ].T = ' + str(self.theta[-i].T))
                #print('sigmoidDerivative(a[-i-1]) = '+ str(sigmoidDerivative(a[-i-1])))

                if i == 1:      
                    inner = np.dot(np.transpose(self.theta[-i]),deltas[i-1])
                else:
                    inner = np.dot(np.transpose(self.theta[-i]),deltas[i-1][1:])
                      
                #print('inner = ' + str(inner))
                deltas.append( inner*sigmoidDerivative(a[-i-1]) )
                #print('deltas[' + str(i) + '] = ' + str(deltas[i]))
                
            #print('sigmoidDerivative(a[-i-1]) = '+ str(sigmoidDerivative(a[-i-1])))
            #print('a[' + str(-i-1) + '] = ' + str(a[-i-1]))
            #print('deltas[' + str(i) + '] = ' + str(deltas[i]))
            #print()
        # use deltas to compute loss function derivatives
        #print()
        #print()
        loss_deriv = []
        for i in range(len(self.theta)):
            #print('deltas['+str(-i-1)+'] = ' + str(deltas[-i-1]))
            #print('a['+str(i)+'].T = ' + str(a[i].T))
            loss_deriv.append( np.dot(deltas[-i-1],a[i].T) )
            #print(loss_deriv)
        #update weights
        for i in range(len(self.theta)):
            self.theta[i] += learning_rate*loss_deriv[i]
            if i != 0: # regularization term
                self.theta[i] += reg_term*self.theta[i]

    def lossFunction(self,x,y):
        a = self.feedForward(x)
        h = a[-1]
        J = 0
        for i in range(len(h)):
            J += (-1/self.lengths[0]) * (y[i]*np.log(h[i])+(1-y[i])*np.log(1-h[i]))
            #J += 0.5*(h[i]-y[i])**2
        return J            
            
           
            

def main():
    #net = NeuralNetwork([2,3,1])
    #net.printWeights()
    #net.printWeights()
    #net.train(np.array([2,3,4]),np.array([2]))
    #net.printWeights()
    neuralOrTest()
    
def neuralOrTest():
    net = NeuralNetwork([2,2,1])
    print('Initial Weights: ')
    net.printWeights()
    print('----')
    X = [[0,0],[1,0],[0,1],[1,1]];
    Y = [[0],[1],[1],[1]]
    J = []
    for i in range(100):
        for j in range(len(Y)):
            net.train(X[j],Y[j])
            J.append( net.lossFunction(X[j],Y[j]))
    for i in range(len(Y)):
        y_out = net.testCase(X[i])
        print('Test Case: ' + str(X[i]) + ', Result: ' + str(y_out) )
    print()
    print('----')
    print('Final Weights: ')
    net.printWeights()
    plt.plot(J)
    plt.ylabel('Cost')
    plt.xlabel('Training Sample')
    plt.show()
    
            
if __name__ == '__main__':
    main()
