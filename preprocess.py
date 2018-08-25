"""
DESCRIPTION
Preprocesses audio data before sending to Neural Network

See demo in in main()


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

import neuralnet_02 as NN
import numpy as np
import os
import glob
import json
import time



import scipy
import matplotlib.pylab as plt
import scipy.io.wavfile as wavfile
import scipy.fftpack
from scipy.fftpack import dct


def getMax(array_list):
    """Returns a tuple (index,value) of the maximum in an 1D array or list"""
    m = array_list[0]
    m_index = 0
    for i,value in enumerate(array_list):
        if value > m:
            m = value
            m_index = i
    return (m_index,m)


def processFile(filename,length = 256,q=1,fs_in=8000,divide=4,plot=False):
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
    length = length*divide
    #fs = sample rate, sound = multichannel sound signal
    try:
        fs1, sound = wavfile.read(filename)
    except ValueError:
        print(str(filename) + ' failed to process')
        return 'failed'
    if fs1 != fs_in:
        raise ValueError('Sampling rate should be ' + str(fs_in) + ' for: ' + filename)
    sig1 = sound[:,0] #left channel
    pre_emphasis = 0.97
    sig1 = np.append(sig1[0], sig1[1:] - pre_emphasis * sig1[:-1])

    
    fs2, sig2 = downsample(sig1,fs1,q)
    N2 = len(sig2)
    sig3 = sig2[N2//2-length:N2//2+length]
    #print(len(sig3))

    FFT = abs(scipy.fft(sig3))
    FFT_side = FFT[range(len(FFT)//2)]
    #freqs = scipy.fftpack.fftfreq(sig3.size, 1/fs2)
    #plt.plot(freqs,FFT)
    if len(FFT_side) != length:
        print('ERROR MESSAGE DETAILS')
        print('filename: ' + filename)
        print('length = ' + str(length))
        print('fs_in = ' + str(fs_in))
        print('q = ' + str(q))
        print('divide = ' + str(divide))
        total_time = len(sig1)/fs1
        print('total_time =  ' + str(total_time))
        print('Please check: length < total_time*fs//(2*q)')
        print('Check: ' + str(length) + ' < ' + str(total_time*fs1//(2*q)))
        raise ValueError('Length FFT_side != length: ' + str(len(FFT_side)) + ' != ' + str(length))
        
    
    FFT_log = []
    # normalize FFT
    for value in FFT_side:
        value = np.log(value)
        FFT_log.append(value)
    max_val = getMax(FFT_log)[1]
    FFT_norm = []
    for value in FFT_log:
        FFT_norm.append(value/max_val)
    
    
    FFT_side = np.array(FFT_norm)
    FFT_divided =  FFT_side[range(length//divide)]
    #plot = True
    if plot == True:
        freqs = scipy.fftpack.fftfreq(sig3.size, 1/fs2)
        freqs_divided = np.array(freqs[range(len(FFT_divided))])
        plt.plot(freqs_divided,FFT_divided) # plotting the complete fft spectrum
        plt.show()
    
    return FFT_divided


def processMFCC(filename,subsample=2048):
    #assume 8000Hz
    #amplify high frequencies

    #Setup
    try:
        fs, signal = wavfile.read(filename)  # File assumed to be in the same directory
    except:
        print(filename + ' failed to process.')
        print('Failed Read')
        print()
        return 'failed'
    half = len(signal)//2
    side = subsample//2
    signal = signal[half-side:half+side]
    if side != len(signal)//2:
        print(filename + ' failed to process.')
        print('N too small, N: ' + str(len(signal)) + ', subsample: ' + str(subsample))
        print()
        return 'failed'
    try:
        sig = signal[:,0] #get first channel
    except:
        sig = signal
    
    #Pre-Emphasis
    pre_emphasis = 0.97
    e_sig = sig[1:] - pre_emphasis * sig[0:-1] #emphasized signal
    sig_len = len(e_sig)
    
    #Framing
    fr_size = 0.025 # frame size (sec)
    fr_overlap = 0.01 # frame stride, frame overlap (sec)
    fr_len = int(round(fr_size * fs)) # frame length (sec/sec)
    fr_step = int(round(fr_overlap * fs)) # amt to step frame each time 
    num_fr = int(np.ceil(np.abs(sig_len - fr_len) / fr_step)) #Number of Frames

    padding = num_fr * fr_step + fr_len # Amount of padding between frames
    z = [0 for _ in range(padding-sig_len)]
    z = np.array(z)
    pad_sig = np.append(e_sig, z) # Pad Signal so frames equal size

    #idx = np.tile(np.linspace(0, fr_len,fr_len), (num_fr, 1)) + np.tile(np.linspace(0, num_fr * fr_step, fr_step * num_fr), (fr_len, 1)).T
    #fr = pad_sig[idx]
    idx = np.tile(np.arange(0, fr_len), (num_fr, 1)) + np.transpose(np.tile(np.arange(0, num_fr * fr_step, fr_step), (fr_len, 1)))
    fr = pad_sig[idx.astype(np.int32)]

    #Window
    NFFT = 512
    fr = fr * ( 0.54 - 0.46 * np.cos((2 * np.pi * NFFT) / (fr_len - 1)) )  # Hamming Window


    #Fourier-Transform and Power Spectrum
    #NFFT = NFFT
    mag_fr = np.absolute(np.fft.rfft(fr, NFFT))  # Magnitude of the FFT
    pow_fr = (1.0 / NFFT) * ((mag_fr) ** 2)  # Power Spectrum

    #Filter Banks
    nfilt = 40
    f_low = 0
    f_high = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(f_low, f_high, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    b = np.floor((NFFT + 1) * hz_points / fs) #bin

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for i in range(1, nfilt + 1):
        f_m_minus = int(b[i - 1])   # left
        f_m = int(b[i])             # center
        f_m_plus = int(b[i + 1])    # right

        for j in range(f_m_minus, f_m):
            fbank[i - 1, j] = (j - b[i - 1]) / (b[i] - b[i - 1])
        for j in range(f_m, f_m_plus):
            fbank[i - 1, j] = (b[i + 1] - j) / (b[i + 1] - b[i])
    fb = np.dot(pow_fr, np.transpose(fbank)) # filter banks
    fb = np.where(fb == 0, np.finfo(float).eps, fb)  # Numerical Stability
    fb = 20 * np.log10(fb)  # convert to dB

    #Mel-frequency Cepstral Coefficients (MFCCs)
    num_ceps = 12
    mfcc = dct(fb, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

    #Sinusoidal Filtering
    c_lift = 22 # dim of MFCC vector
    (n_fr, n_coeff) = mfcc.shape #number of frames number of coeff
    ncoeff_array = np.arange(n_coeff)
    lift = 1 + (c_lift / 2) * np.sin(np.pi * ncoeff_array / c_lift)
    mfcc = mfcc * lift  

    #Mean Normalization
    epsilon = 1e-8
    for i in range(len(fb)):
        fb[i] -= mean(fb) + epsilon
    for i in range(len(mfcc)):
        mfcc[i] -= mean(mfcc) + epsilon

    output = []
    for i in range(len(mfcc)):
        for j in range(len(mfcc[0])):
            output.append(mfcc[i][j])
    
    m = getMax(output)[1]
    for i,value in enumerate(output):
        output[i] = value/m
    return np.array(output)


def mean(array_list):
    """Returns the mean of an array or list"""
    count = 0.0
    for value in array_list:
        count += value
    return count/len(array_list)


    



def downsample(sig,fs,q):
    """
    sig (list,array): sound/data signal
    q (int): downsample factor
    """
    N = len(sig)//q
    new_sig = []
    for i in range(len(sig)//q):
        new_sig.append(sig[i*q])
    new_sig = np.array(new_sig)
    return (fs//q,new_sig)

class Preprocess:
    def __init__(self):
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

        #if process == False:
            #self.loadData(data_file)
        #else: #process == True:
            #self.processData(data_file,directory,comment)

    def getXY(self):
        """Returns X (List of Input Vectors), and Y (List of Output Vectors)
            for preprocessed data
            ex) X = [[0,0],[0,1],[1,0],[1,1]]
            ex) Y = [[0],[1],[1],[0]]
        """
        return (self.X,self.Y)

    def getInputLength(self):
        """Returns length of Input Layer"""
        return len(self.X[0])

    def getOutputLength(self):
        """Returns length of Output Layer"""
        return len(self.Y[0])
    
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
        #Test prints, uncomment to test if data looks correct
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
        t0 = time.time()
        for name in self.dirs:
            t1 = time.time()
            for file in self.files[name]:
                #input_vector = processFile(file,length=length1,q=q1,fs_in=fs_in1,divide=divide1,plot = False)
                if way == 'mfcc':
                    input_vector = processMFCC(file,*opt)
                elif way == 'fft':
                    input_vector = processFFT(file,*opt)
                else:
                    raise ValueError('Invalid Way, valid types include: \'mfcc\' or \'fft\'')
                if input_vector != 'failed':
                    self.X.append(input_vector)
                    self.Y.append(self.output[name])
            print('Time take to process '+str(name)+ ': ' + str((time.time()-t1)/60)[0:4] + ' min.')
        print('Total Processing Time: ' + str((time.time()-t0)/60)[0:4] + ' min.')

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
    v = processMFCC('instruments_07/banjo/banjo_A3_very-long_forte_normal.wav')
    print('len(input layer) = ' + str(len(v)))
    #raise Exception
    P = Preprocess()
    #P.processData('preprocessed/processed_01.txt',directory='instruments_07',fs_in=8000,length=input_length,q=1,divide=1,comment = 'Instrument Data')
    P.processData('preprocessed/training_02.txt',directory='instr_train_03',way='mfcc',opt = [2048])
    P.loadData('preprocessed/training_02.txt')
    X, Y = P.getXY()
    print('Input Layer Length: ' + str(len(X[0])))
    print('Output Layer Length: ' + str(len(Y[0])))
    input_size = P.getInputLength()
    output_size = P.getOutputLength()
    net = NN.NeuralNetwork([input_size,100,output_size],'sigmoid')
    net.storeWeights('weights/instr_03')
    net.loadWeights('weights/instr_03')
    #net.trainWithPlots(X,Y,learning_rate=1,intervals = 100,way='max')

    Q = Preprocess()
    Q.processData('preprocessed/testing_02.txt',directory='instr_test_03',way='mfcc',opt=[2048])
    Q.loadData('preprocessed/testing_02.txt')
    tX, tY = Q.getXY()
    net.testBatch(tX,tY)

    # Test print functions, these print statements can be used to figure
    # out how to use code
    # X, Y = P.getXY()
    # files = P.getFileList()
    # output_vectors = P.getOutputVectors()
    # output_names = P.getOutputNames()
    # print()
    # print('X = ' + str(X))
    # print()
    # print('Y = ' + str(Y))
    # print()
    # print('File List = ' + str(files))
    # print()
    # print('Output Vectors = ' + str(output_vectors))
    # print()
    # print('Output Names = ' + str(output_names))

    
if __name__ == '__main__':
    main()
