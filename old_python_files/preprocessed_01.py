import neuralnet_01 as NN
import fft_features_02 as FF
import numpy as np
import os
import glob
import json
import time









class Preprocess:
    def __init__(self,data_file,process=False,directory='IRMAS-TrainingData',comment = ''):
        """data_file (string): contains the file to load or store data, ex)data.txt
            process (bool): if False, load data from data_file,
                            if True, process data in directory & store in data_file
            directory (string): (optional) directory of data to be processed
        """
        # Ex) self.output['cel']: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.output = {}
        
         # directory names are names of instruments
        #self.dirs =
        # ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
        self.dirs = [] # list of names of subdirectories in directory

        # example: self.files['sax'] =
        # IRMAS-TrainingData\sax\006__[sax][nod][cla]1686__1.wav
        self.files = {} # dictionary of dir:[file,file,file]

        # self.data = {} # dictionary of dir:[input_nodes,input_nodes]

        # self.X is dictionary of dir:[input_nodes1,input_nodes2]
        # self.Y is dictionary of dir:[output_nodes1,output_nodes2]
        # self.Y corresponds to self.X
        self.X = [] # list of input vectors
        self.Y = [] # list of output vectors

        if process == False:
            self.loadData(data_file)
        else: #process == True:
            self.processData(data_file,directory,comment)

    def getXY(self):
        """Returns X (List of Input Vectors), and Y (List of Output Vectors)
            for preprocessed data
            ex) X = [[0,0],[0,1],[1,0],[1,1]]
            ex) Y = [[0],[1],[1],[0]]
        """
        return (self.X,self.Y)
    
    def getFileList(self):
        """Returns a dictionary with key:value 'Output Name':[file list]
            ex) {'sax':['sax1.wav','sax2.wav','sax3.wav']}
        """
        return self.files

    def getOutputVectors(self):
        """ Returns a dictionary with key:value 'OutputName':output vector
        Ex) output['cel']: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        """
        return self.output

    def getOutputNames(self):
        """Returns a list of the names of the output vectors
        ex) ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
        """
        return self.dirs

    def loadData(self,data_file):
        """Loads the data in data_file into Trainer"""
        #Load the data from the json
        with open(data_file) as json_file:  
            data = json.load(json_file)

        # Clear all instance variables
        self.dirs = []
        self.files = {}
        self.X = []
        self.Y = []
        self.output = {}

        # stored the data into the instance variables
        self.dirs = data['dirs'] #good
        self.files = data['files'] # good
        
        # self.output is a dict() with string:np.array
        output = data['output']
        for e in output:
            self.output[e] = np.array(output[e]) # -> fine
        #self.X is a list of np.arrays
        X = data['X']
        for x in X:
            self.X.append(np.array(x))# -> fine
        #self.Y is a list of np.arrays
        Y = data['Y']
        for y in Y:
            self.Y.append(list(y))# -> fine
        #Test prints
        #print('self.dirs = ' + str(self.dirs))
        #print()
        #print('self.files = ' + str(self.files))
        #print()
        #print('self.output = ' + str(self.output))
        #print()
        #print('self.X = ' + str(self.X))
        #print()
        #print('self.Y = ' + str(self.Y))
        #print()
        print('Preprocessed data loaded from ' + str(data_file))
        print(data['comment'])
        return 
        
        

    def processData(self,data_file,directory,comment = ''):
        """Processes the data in directory and stores it in data_file
            directory (string): folder of data to be processed
            data_file (string): name of file for data to be stored ex) 
        """
        self.dirs = [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]

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

        # Ex) self.output['cel']: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
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


        #self.X = [] # list of input vectors
        #self.Y = [] # list of output vectors
        for name in self.dirs:
            t1 = time.time()
            for file in self.files[name]:
                input_vector = FF.processFile(file,plot = False)
                self.X.append(input_vector)
                self.Y.append(self.output[name])
            print('Time take to process '+str(name)+ ': ' + str((time.time()-t1)/60) + 'min')

        # Now we can store all of the data in a json
        # Need to store self.X, self.Y, self.dirs,self.output,self.files,self.data
        # self.dirs is a list of strings -> fine
        # self.files is a dict() with string:string -> fine
        # self.output is a dict() with string:np.array
        output = {}
        for d in self.output:
            out_list = []
            for value in self.output[d]:
                out_list.append(int(value))
            output[d] = out_list # -> fine
        #self.X is a list of np.arrays
        X = []
        for i in range(len(self.X)):
            x = []
            for ele in self.X[i]:
                x.append(float(ele))
            X.append(x) # -> fine
        #self.Y is a list of np.arrays
        Y = []
        for i in range(len(self.Y)):
            y = []
            for ele in self.Y[i]:
                y.append(float(ele))
            Y.append(y) # -> fine
            
        store = {}
        store['dirs'] = self.dirs # good
        store['output'] = output # good
        store['files'] = self.files # good
        store['X'] = X # good
        store['Y'] = Y # good
        store['comment'] = comment
        with open(data_file, 'w') as outfile:
            json.dump(store, outfile)
        print('Preprocessed data stored in ' + str(data_file))
        return

        

        
        








def main():
    # Note: Preprocessed data should be in folder preprocessed
    P = Preprocess('preprocessed/test_01.txt',process=True,directory='phil_temp_03',comment = 'Hello World')
    X, Y = P.getXY()
    net = NN.NeuralNetwork([1024,1024,2],'tanh')
    net.trainWithPlots(X,Y,learning_rate=1.0,intervals = 1)

    #X, Y = P.getXY()
    #files = P.getFileList()
    #output_vectors = P.getOutputVectors()
    #output_names = P.getOutputNames()
    #print()
    #print('X = ' + str(X))
    #print()
    #print('Y = ' + str(Y))
    #print()
    #print('File List = ' + str(files))
    #print()
    #print('Output Vectors = ' + str(output_vectors))
    #print()
    #print('Output Names = ' + str(output_names))

    
if __name__ == '__main__':
    main()
