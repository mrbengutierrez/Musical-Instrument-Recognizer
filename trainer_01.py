import neuralnet_01 as NN
import fft_features_02 as FF
import numpy as np
import os
import glob
import time

def getMax(array_list):
    """Returns a tuple (index,value) of the maximum in an 1D array or list"""
    m = array_list[0]
    m_index = 0
    for i,value in enumerate(array_list):
        if value > m:
            m = value
            m_index = i
    return (m_index,m)
         
    
    
class Trainer:
    def __init__(self,directory='IRMAS-TrainingData',input_nodes = 1024):
        # directory names are names of instruments
        #self.dirs =
        # ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
        self.dirs = [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]
        
        
        # example: self.files['sax'] =
        # IRMAS-TrainingData\sax\006__[sax][nod][cla]1686__1.wav
        self.files = {}
        for d in self.dirs:
            self.files[d] = [] 
            sub_dir = os.path.join(directory, d)
            for filename in glob.glob(os.path.join(sub_dir, '*.wav')):
                self.files[d].append(filename)
        self.net = NN.NeuralNetwork([input_nodes,2*input_nodes,2*input_nodes,len(self.dirs)])

        # Ex) self.output['cel']: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.output = {}
        i = 0
        for name in self.dirs:
            temp = []
            for j in range(len(self.dirs)):
                if i == j:
                    temp.append(1)
                else:
                    temp.append(0)
            self.output[name] = np.array(temp)
            i +=1
        
                
            

    def train(self,learning_rate = 1.0, epochs = 1):
        X = [] # list of input vectors
        Y = [] # list of output vectors
        for name in self.dirs:
            t1 = time.time()
            for file in self.files[name]:
                input_vector = FF.processFile(file,plot = False)
                X.append(input_vector)
                Y.append(self.output[name])
            print('Time take to process '+str(name)+ ': ' + str((time.time()-t1)/60) + 'min')
        self.net.trainWithPlots(X,Y,learning_rate,epochs)
            
                    

                    
       
                





def main():
    T = Trainer()
    T.train()

    
if __name__ == '__main__':
    main()
