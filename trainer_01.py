import neuralnet_01 as NN
import fft_features_02 as FF
import os
import glob

        
class Trainer:
    def __init__(self,directory='IRMAS-TrainingData',input_nodes = 1024):
        # directory names are names of instruments
        self.dirs = [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]
        
        # example: self.files['sax'] =
        # IRMAS-TrainingData\sax\006__[sax][nod][cla]1686__1.wav
        self.files = {}
        for d in self.dirs:
            self.files[d] = [] #
            sub_dir = os.path.join(directory, d)
            for filename in glob.glob(os.path.join(sub_dir, '*.wav')):
                self.files[d].append(filename)
        self.net = NN.NeuralNetwork([input_nodes,2*input_nodes,2*input_nodes,len(self.files)])
        print(self.net.getLayers())

    def train(self,activeFn = 'sigmoid',learning_rate = 1.0, epochs = 1):
        pass
       
                





def main():
    T = Trainer()

    
if __name__ == '__main__':
    main()
