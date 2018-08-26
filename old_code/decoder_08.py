import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoidDerivative(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanhDerivative(x):
    return 1.0 - x**2

class NeuralNetwork:

    def __init__(self,layers,activeFn = 'sigmoid'):
        """layers is list of layer lengths"""
        if activeFn == 'tanh':
            self.activeFn = tanh
            self.activeFnDerivative = tanhDerivative
        elif activeFn == 'sigmoid':
            self.activeFn = sigmoid
            self.activeFnDerivative = sigmoidDerivative
        self.layers = layers
        self.theta = []
        for i in range(1,len(layers)):
            if i != len(layers)-1: 
                weight_matrix = 2*np.random.rand(layers[i-1] + 1, layers[i] + 1) -1
                #np.random.uniform(low = -1, high = 1, size=(layers[i-1]+1,layers[i]+1))
            else:
                weight_matrix = 2*np.random.rand(layers[i-1] + 1, layers[i]) -1
            #print('weight_matrix = ' +str(weight_matrix))
            self.theta.append(weight_matrix)
            
    def trainBatch(self,X,y,learning_rate=1.0,intervals = 100):
        pass
    def predict(self,x):
        return self.forwardProp(x)[-1]
    def train(self,x,y,learning_rate=1.0):
        a = self.forwardProp(x)
        #print('a = ' + str(a))
        self.backProp(a,y,learning_rate)
        
    def forwardProp(self,x):
        x = np.array(x)
        a = [np.append([1],x)]

        for l in range(len(self.theta)):
            #print('a['+str(l)+'] = ' + str(a[l]))
            #print('self.theta['+str(l)+'] = ' + str(self.theta[l]))
            inner = np.dot(a[l],self.theta[l])
            a.append( self.activeFn( inner) )
        return a
    def backProp(self,a,y,learning_rate):
        #StartHere
        y = np.array(y)
        delta = y - a[-1]
        deltas = [delta * self.activeFnDerivative(a[-1])]

        for layer in range(len(a)-2,0,-1):
            #print('deltas['+str(-1)+'] = ' + str(deltas[-1]))
            #print('self.theta['+str(layer)+'].T = ' + str(self.theta[layer].T))
            #print('self.theta['+str(l)'].T = ' + str(self.theta[l].T))
            deltas.append(deltas[-1].dot(self.theta[layer].T)*self.activeFnDerivative(a[layer]))
        deltas.reverse()
        #print('deltas = ' + str(deltas))

        for i in range(len(self.theta)):
            active_layer = np.atleast_2d(a[i])
            delta = np.atleast_2d(deltas[i])
            self.theta[i] += learning_rate * active_layer.T.dot(delta)
        
        
            

        
        


def main():
    #net = NeuralNetwork([2,2,1])
    #net.train([1,0],[1])
    neuralOrTest()

def neuralOrTest():
    net = NeuralNetwork([2,2,1],'tanh')
    #print('Initial Weights: ')
    #net.printWeights()
    #print('----')
    X = [[0,0],[1,0],[0,1],[1,1]];
    Y = [[0],[1],[1],[0]]
    J = []
    for i in range(10000):
        for j in range(len(Y)):
            net.train(X[j],Y[j])
            #J.append( net.lossFunction(X[j],Y[j]))
    for i in range(len(Y)):
        y_out = net.predict(X[i])
        print('Test Case: ' + str(X[i]) + ', Result: ' + str(y_out) )
    #print()
    #print('----')
    #print('Final Weights: ')
    #net.printWeights()
    #plt.plot(J)
    #plt.ylabel('Cost')
    #plt.xlabel('Training Sample')
    #plt.show()
    
if __name__ == '__main__':
    main()
