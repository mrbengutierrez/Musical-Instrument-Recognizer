"""
DESCRIPTION
Application Project Interface for Instrument Detection Software
Preprocess: Preprocesses audio data before sending to Neural Network
NeuralNetwork: General purpose neural network that can be used for many
applications

See demo in in main()

Please contact Benjamin Gutierrez for any questions or errors.


MIT License

Copyright (c) 2018  The-Instrumental-Specialists

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Standard Library Dependencies
# os
# glob
# json
# time

# Third Party Dependencies (Please install prior to use)
# scipy
# numpy
# matplotlib

import neuralnet as NN
import preprocess as PP

#--- START PREPROCESS API ----------------------------------------------------------
def processFFT(filename,length = 256,q=1,fs_in=8000,divide=4,plot=False):
    """returns one sided FFT amplitudes of filename
        filename (string): ex) 'sax.wav'
        length (int): Number of datapoints of one-sided fft (must be even,preferably a power of 2)
        q (int): (optional argument) Downsampling Rate 
        fs_in (int): (optional argument) throw ValueError if fs of filename != fs_in
        divide (int): (optional argument) 1/divide*Nsamples is taken from FFT (preferably even)
        plot (bool): (optional argument) plots the one sided FFT if True, otherwise does not plot
        
        Note: length < total_time*fs//(2*q*divide)
        Ex) length = 256 < (0.25sec)*(44100Hz)//(2*4*2) = 689
    """
    return PP.processFFT(filename,length,q,fs_in,divide,plot)


def processMFCC(filename,subsample=2048):
    """Preprocesses file. Returns Mel-frequency Cepstral Coefficients.
        filename (string): wavfile
        subsample: number of datapoints in wavfile to use in FFT
    """
    #assumes 8000Hz, but works with 44,100Hz and other sample rates.

    return PP.processMFCC(filename,subsample)

def mean(array_list):
    """Returns the mean of an array or list"""
    return PP.mean(array_list)

def downsample(sig,fs,q):
    """
    sig (list,array): sound/data signal
    q (int): downsample factor
    """
    return PP.downsample(sig,fs,q)

class Preprocess(PP.Preprocess):
    def __init__(self):
        """data_file (string): contains the file to load or store data, ex)data.txt
            process (bool): if False, load data from data_file,
                            if True, process data in directory & store in data_file
            directory (string): (optional) directory of data to be processed
        """
        PP.Preprocess.__init__(self)

    def getXY(self):
        """Returns X (List of Input Vectors), and Y (List of Output Vectors)
            for preprocessed data
            ex) X = [[0,0],[0,1],[1,0],[1,1]]
            ex) Y = [[0],[1],[1],[0]]
        """
        return super().getXY()

    def getInputLength(self):
        """Returns length of Input Layer"""
        return super().getInputLength()

    def getOutputLength(self):
        """Returns length of Output Layer"""
        return super().getOutputLength()
    
    def getFileList(self):
        """Returns a dictionary with key:value 'Output Name':[file list]
            ex) {'sax':['sax1.wav','sax2.wav','sax3.wav']}
        """
        return super().getFileList()

    def getOutputVectors(self):
        """ Returns a dictionary with key:value 'OutputName':output vector
        Ex) output['cel']: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        """
        return super().getOutputVectors()

    def getOutputNames(self):
        """Returns a list of the names of the output vectors
        ex) ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
        """
        return super().getOutputNames()

    def loadData(self,data_file):
        """Loads the data in data_file into Trainer"""
        return super().loadData(data_file)

    def processData(self,data_file,directory,comment = '',way='mfcc',opt=[1024]):
        """Processes the data in directory and stores it in data_file
            directory (string): folder of data to be processed
            data_file (string): name of file for data to be stored ex) data.txt
            comment (string): optional message to be stored with data

            way = 'fft',  opts is a list containing
            length (int): Number of datapoints of one-sided fft (must be even,preferably a power of 2)
            q (int): Downsampling Rate (must be even, preferably power of 2)
            fs_in (int):  throw ValueError if fs of filename != fs_i
            divide (int):  1/divide*Nsamples is taken from FFT (preferably even)
            plot (bool): ( plots the one sided FFT if True, otherwise does not plot            
        
            Note: length < total_time*fs/(q)
            Ex) length = 1024 < (0.25sec)*(44100Hz)/(4) = 2756


            way = 'mfcc', opts is a list containing
            subsample (int) = Number of subsamples to take from audio file.
            Note: Input Vector length is determined from processMFCC.
                    Usually it is on the order of ~ 0.5*subsample.
        """
        return super().processData(data_file,directory,comment,way,opt)
#--- END PREPROCESS API ----------------------------------------------------------







#--- START NEURALNET API ------------------------------------------------------
def sigmoid(x):
    """ This function computes the sigmoid of x for NeuralNetwork"""
    return NN.sigmoid(x)

def sigmoidDerivative(x):
    """ This function computes the sigmoid derivative of x for NeuralNetwork
        (Note: Not Real Derivative)
    """
    return NN.sigmoidDerivative(x)

def tanh(x):
    """ This function computes the tanh of x for NeuralNetwork"""
    return NN.tanh(x)

def tanhDerivative(x):
    """ This function computes the tanh derivative of x for NeuralNetwork
        (Note: Not Real Derivative)
    """
    return NN.tanhDerivative(x)

def arctan(x):
    """This function returns the arctan of x for NeuralNetwork"""
    return NN.arctan(x)

def arctanDerivative(x):
    """This function returns the arctan derivative of x for NeuralNetwork
        (Note: Not Real Derivative)
    """
    return NN.arctanDerivative(x)

def sin(x): 
    """This function returns the sine of x"""
    return NN.sin(x)

def sinDerivative(x): 
    """This function returns the sine derivative of x
        (Note: Not Real Derivative)
    """
    return NN.sinDerivative(x)

def gaussian(x): 
    """This function returns the gaussian of x"""
    return NN.gaussian(x)

def gaussianDerivative(x): 
    """This function returns the gaussian derivative of x
        (Note: Not Real Derivative)
    """
    return NN.gaussianDerivative(x)

def softplus(x): 
    """This function returns the softplus of x"""
    return NN.softplus(x)

def softplusDerivative(x): 
    """This function returns the softplusDerivative of x
        (Note: Not Real Derivative)
    """
    return NN.softplusDerivative(x)


def getMax(array_list):
    """Returns a tuple (index,value) of the maximum in an 1D array or list"""
    return NN.getMax(array_list)


        
class NeuralNetwork(NN.NeuralNetwork):
    """ General Purpose Neural Network
    activation_dict = {'sigmoid': [sigmoid,sigmoidDerivative],
                       'tanh': [tanh,tanhDerivative],
                       'arctan': [arctan,arctanDerivative],
                       'sin': [sin,sinDerivative],
                       'gaussian': [gaussian,gaussianDerivative],
                       'softplus': [softplus,softplusDerivative],
                       }
    """
    def __init__(self,layers,activeFn = 'sigmoid'):
        """layers is list of layer lengths"""
        NN.NeuralNetwork.__init__(self,layers,activeFn)
        #self.net = NN.NeuralNetwork(layers,activeFn)

    def setActivationFn(self,activeFn):
        """Sets the activation function"""
        #self.net.setActivationFn(activeFn)
        #print('s = ' + str(super()))
        return super().setActivationFn(activeFn)
        
    def getLayers(self):
        """Return a list of layer lengths"""
        return super().getLayers()
    
    def storeWeights(self,filename,comment = 'No Comment'):
        """Stores Weights in filename.
            filename (string): ex. 'data.txt'
            comment (string): message to be stored in file
        """
        return super().storeWeights(filename,comment)
    
    def loadWeights(self,filename):
        """Loads Weights in filename. Note WILL overwrite layer shape and number.
            filename (string): ex. 'data.txt'
        """
        return super().loadWeights(filename)
        
    def trainRandom(self,X,Y,learning_rate=1.0,intervals = 100):
        """Trains the neural networks with a list of input vectors x
           and a list of output vectors Y. Iterates in Random Order.
        """
        return super().trainRandom(X,Y,learning_rate,intervals)

    def trainSequential(X,Y,learning_rate=1.0,intervals = 100):
        """Trains the neural networks with a list of input vectors X
           and a list of output vectors Y. Iterates in Sequential Order.
        """
        return super().trainSequential(X,Y,learning_rate,intervals)

    def trainWithPlots(self,X,Y,learning_rate = 1.0,intervals = 100,way='max'):
        """Trains the neural networks with a list of input vectors X
           and a list of output vectors Y.
           Plots Cost function over each iteration.
           Iterates in Sequential Order.
           way (string): (optional arg) comparison method for testing
                            valid options are 'max','thres'
        """
        return super().trainWithPlots(X,Y,learning_rate,intervals,way)
  
    def trainSample(self,x,y,learning_rate=1.0):
        """Trains the neural networks with a single input vector x
           and a single output vector y"""
        return super().trainSample(x,y,learning_rate)

    def trainTestSample(self,x,y,learning_rate=1.0,way='max'):
        """Trains the neural networks with a single input vector x
            and a single output vector y.
            Takes prediction of tested sample using forward propagation
            before doing backpropagation.
            way (string): (optional arg) comparison method for testing
                            valid options are 'max','thres'

            tl:dr trains and tests one sample
        """
        return super().trainTestSample(x,y,learning_rate,way)

    def compareProb(self,prob,y,way='max'):
        """Compares y with prob, probabitity vector from last activation layer
            in backpropagation
        """
        return super().compareProb(prob,y,way)
        
    def forwardProp(self,x):
        """Forward Propagates x through the Neural Network"""
        return super().forwardProp(x)
    
    def backProp(self,a,y,learning_rate):
        """Backward propagates y through the Neural Network using activations a"""
        return super().backProp(a,y,learning_rate)

    def testBatch(self,X,Y,verbose = False,way='max'):
        """prints and returns the testing accuracy of a batch of testing vectors.
            X (list of np.arrays): list of input vectors
            Y (list of np.arrays): list of target vectors
            if verbose == True, prints out whether each test vector passed/failed.
        """
        return super().testBatch(X,Y,verbose,way)
            
    def testSample(self,x,y,way='max'):
        """ Returns true prediction is correct
            Takes prediction of tested sample using probabilities from
            forward propagation.
            way (string): (optional arg) comparison method for testing
                            valid options are 'max','thres'
        """
        return super().testSample(x,y,way)
            
    def predictProb(self,x):
        """Predicts an output vector for a given input vector x"""
        return super().predictProb(x)
    
    def lossFunction(self,x,y):
        """Computes the loss function for a given input vector and output vector"""
        return super().lossFunction(x,y)

    def printWeights(self):
        """Prints all of the weight matrices"""
        return super().printWeights()
       
    def printWeight(self,i):
        """Prints the weight matrix at index i of self.theta"""
        return super().printWeight(i)
#--- END NEURALNET API ------------------------------------------------------











def main():
    # Example 1 (NeuralNet)
    #trainXor()

    # Example 2 (Preprocess + NeuralNet)
    test6Instruments()

    # Example 3 (Preprocess + NeuralNet
    test10Instruments()
    
    # Example 3 (Preprocess + NeuralNet)
    testNotes()



def trainXor():
    """Training NeuralNet to learn the boolean XOR function"""
    
    # Initialize Neural Network with tanh activation function,
    # with an input layer of size 2, one hidden layer of size 2,
    # and one output layer of size 1
    net = NeuralNetwork([2,2,1],'tanh')

    # XOR Training and Test Data
    X = [[0,0],[1,0],[0,1],[1,1]];
    Y = [[0],[1],[1],[0]]

    # Train with plots
    net.trainWithPlots(X,Y,learning_rate=0.2,intervals=2000,way='thres')

    # Store, load, print weights
    net.storeWeights('weights/XOR.txt',comment='XOR DATA')
    net.loadWeights('weights/XOR.txt')
    net.printWeights()

    # test XOR data
    net.testBatch(X,Y,verbose=True)

    # Predict Data
    net.predictProb([0,0]) # predict probability

def train10Instruments():
    """Uses Preprocess to convert the audio data into mel-frequency cepstral coefficients.
        Feeds these coefficients into NeuralNet.
        Ten instruments are used in this example
    """
    # Preprocess Training Data
    P = Preprocess()
    #P.processData('preprocessed/instr_train_10.txt',directory='instr_train_10',way='mfcc',opt = [2048])
    P.loadData('preprocessed/instr_train_10.txt') #Load preprocessed data from file, since net has been trained
    X, Y = P.getXY()
    input_size = P.getInputLength()
    output_size = P.getOutputLength()

    # Train Neural Net
    net = NeuralNetwork([input_size,100,output_size],activeFn='sigmoid')
    net.trainWithPlots(X,Y,learning_rate=1,intervals = 100,way='max')
    net.storeWeights('weights/instr_train_10.txt')
    #net.loadWeights('weights/instr_train_10.txt') # Load weights from file, since net has been trained
    
    # Preprocess Testing Data
    Q = Preprocess()
    Q.processData('preprocessed/instr_test_10.txt',directory='instr_test_10',way='mfcc',opt=[2048])
    #Q.loadData('preprocessed/instr_test_10.txt') # Load weights from file, since net has been trained
    tX, tY = Q.getXY()

    # Test testing data
    net.testBatch(tX,tY)

def train6Instruments():
    """Uses Preprocess to convert the audio data into mel-frequency cepstral coefficients.
        Feeds these coefficients into NeuralNet.
        Six instruments are used in this example
    """
    # Preprocess Training Data
    P = Preprocess()
    #P.processData('preprocessed/instr_train_06.txt',directory='instr_train_06',way='mfcc',opt = [2048])
    P.loadData('preprocessed/instr_test_06.txt') #Load preprocessed data from file, since net has been trained
    X, Y = P.getXY()
    input_size = P.getInputLength()
    output_size = P.getOutputLength()

    # Train Neural Net
    net = NeuralNetwork([input_size,100,output_size],activeFn='sigmoid')
    #net.trainWithPlots(X,Y,learning_rate=0.1,intervals = 75,way='max')
    #net.storeWeights('weights/instr_train_06.txt')
    net.loadWeights('weights/instr_train_06.txt') # Load weights from file, since net has been trained
    
    # Preprocess Testing Data
    Q = Preprocess()
    #Q.processData('preprocessed/instr_test_06.txt',directory='instr_test_06',way='mfcc',opt=[2048])
    Q.loadData('preprocessed/instr_test_06.txt') # Load weights from file, since net has been trained
    tX, tY = Q.getXY()

    # Test testing data
    net.testBatch(tX,tY)

def trainNotes():
    """Uses Preprocess to convert the audio data into mel-frequency cepstral coefficients.
        Feeds these coefficients into NeuralNet.
        19 instruments were used to generate all the notes
    """
    # Preprocess Training Data
    P = Preprocess()
    P.processData('preprocessed/notes_train_19.txt',directory='notes_train_19',way='mfcc',opt = [2048])
    #P.loadData('preprocessed/notes_train_19.txt') #Load preprocessed data from file, since net has been trained
    X, Y = P.getXY()
    input_size = P.getInputLength()
    output_size = P.getOutputLength()

    # Train Neural Net
    net = NeuralNetwork([input_size,100,output_size],activeFn='sigmoid')
    net.trainWithPlots(X,Y,learning_rate=1,intervals = 200,way='max')
    net.storeWeights('weights/notes_train_19.txt')
    #net.loadWeights('weights/notes_train_19.txt') # Load weights from file, since net has been trained
    
    # Preprocess Testing Data
    Q = Preprocess()
    Q.processData('preprocessed/notes_test_19.txt',directory='notes_test_19',way='mfcc',opt=[2048])
    #Q.loadData('preprocessed/notes_test_19.txt') # Load weights from file, since net has been trained
    tX, tY = Q.getXY()

    # Test testing data
    net.testBatch(tX,tY)

def test6Instruments():
    """Uses Preprocess to convert the audio data into mel-frequency cepstral coefficients.
        Feeds these coefficients into NeuralNet.
        Six instruments are used in this example
    """
    # Get preprocessed training data
    P = Preprocess()
    P.loadData('preprocessed/instr_test_06.txt') #Load preprocessed data from file, since net has been trained
    X, Y = P.getXY()
    input_size = P.getInputLength()
    output_size = P.getOutputLength()

    # Load weights for neural net
    net = NeuralNetwork([input_size,100,output_size],activeFn='sigmoid')
    net.loadWeights('weights/instr_train_06.txt') # Load weights from file, since net has been trained

    # Test testing data
    print('Testing 6 Instruments Recognition')
    net.testBatch(X,Y)

def test10Instruments():
    """Uses Preprocess to convert the audio data into mel-frequency cepstral coefficients.
        Feeds these coefficients into NeuralNet.
        Ten instruments are used in this example
    """
    # Get preprocessed training data
    P = Preprocess()
    P.loadData('preprocessed/instr_test_10.txt') #Load preprocessed data from file, since net has been trained
    X, Y = P.getXY()
    input_size = P.getInputLength()
    output_size = P.getOutputLength()

    # Load weights for neural net
    net = NeuralNetwork([input_size,100,output_size],activeFn='sigmoid')
    net.loadWeights('weights/instr_train_10.txt') # Load weights from file, since net has been trained

    # Test testing data
    print('Testing 10 Instruments Recognition')
    net.testBatch(X,Y)

def testNotes():
    """Uses Preprocess to convert the audio data into mel-frequency cepstral coefficients.
        Feeds these coefficients into NeuralNet.
        19 instruments were used to generate all the notes
    """
    # Get preprocessed training data
    P = Preprocess()
    P.loadData('preprocessed/notes_test_19.txt') #Load preprocessed data from file, since net has been trained
    X, Y = P.getXY()
    input_size = P.getInputLength()
    output_size = P.getOutputLength()

    # Load weights for neural net
    net = NeuralNetwork([input_size,100,output_size],activeFn='sigmoid')
    net.loadWeights('weights/notes_train_19.txt') # Load weights from file, since net has been trained

    # Test testing data
    print('Testing Pitch Recognizition')
    net.testBatch(X,Y)
    
if __name__ == '__main__':
    main()
