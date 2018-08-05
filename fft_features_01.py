import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import scipy.signal
import numpy as np
from matplotlib import pyplot as plt
import os
import wave
import audioop

class AudioException(Exception):
    """ General Purpose Exception for class NeuralNetwork"""
    def __init___(self,message):
        Exception.__init__(self,message)
def processFile(filename):
    #fs = sample rate, sound = multichannel sound signal
    fs, sound = wavfile.read(filename) 

    s1 = sound[:,0] #left channel

    N = len(s1) # num of samples
    Ts = 1/fs #sampletime
    T = Ts*N # total time (sec)
    new_T = 0.2
    s2 = s1[N//2-int(new_T/2*N):N//2] + s1[N//2:N//2+int(new_T/2*N)]
    N2 = len(s2) # num of samples

    q = 2 #downsample by factor of 2
    s3 = scipy.signal.resample(sound, 4096, t=None, axis=0, window=None)

def processFile2(src):
    src_read = wave.open(src, mode=None)
    f_in = src_read.getframerate()
    f_out = 11025 # 11025 =44100/4
    if f_in<16385:
        raise AudioException('Sampling rate too small')
    new_wav = downsampleWav(src,f_in,f_out,src_read.getnchannels(),outchannels=1)

    print(new_wav)
    


def downsampleWav(src, Fin=44100, Fout=11025, inchannels=2, outchannels=1):
    src_read = wave.open(src, 'r')

    n_frames = src_read.getnframes()
    data = src_read.readframes(n_frames)

    converted = audioop.ratecv(data, 2, inchannels, Fin, Fout, None)
    if outchannels == 1 and inchannels>1:
        converted = audioop.tomono(converted[0], 2, 1, 0)
    return converted

def processSignal(signal,f_range):
    pass


    

def plotFFT(filename):
    """Plots single sided FFT"""
    fs_rate, signal = wavfile.read(filename)
    len_audio = len(signal.shape)
    print(signal.shape)
    print(signal[:][0])
    if len_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    FFT = abs(scipy.fft(signal))
    FFT_side = FFT[range(N//2)]
    freqs = scipy.fftpack.fftfreq(signal.size, 1.0/fs_rate)
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(N//2)] # one side frequency range
    plt.plot(freqs_side, abs(FFT_side), "b") # plotting the complete fft spectrum
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Single-sided Amplitude')
    plt.show()

def iterateWavFiles():
    for filename in os.listdir(Training):
        if filename.endswith(".wav"): 
            if 'sax' in str(filename):
                sax_list.append(filename)
            elif 'vio' in str(filename):
                vio_list.append(filename)
            else:
                raise Exception('Instrument Name not in File')
        
    return [sax_list,vio_list]

def trainNet():
    """To Train Instrument Detection
            Initialize a neural Net with 100 - 1000 frequencies as input

            for each audio file ,
                preprocess: extract full signal
                            chop up signal (0.25 sec fragments)
                            high/low/bandpass filter
                            downsample
                            (i.e   44100 Hz --> 400 Hz)
                            make sure signal length is a power of 2
                                --> 256, 1024, to speed up fft
            
                extract fft data, (optional: extract time data)
                note: it might be good to subsample audiofile data

            create a dictionary with corresponding fft and output vector

            train dictionary using neural network

        To Train Note Detection:
            Same as above except need have labeled note data

        To predict Note and Instrument Sound at same time
            Chop audio file into small segment (0.1 sec)

            Take fft of each small segment

            Send fft data to note neural net and instrument neural net seperately

            assembly predictions in two separate vectors

            delete repeated predictions (same note)

            Plot

        
    """
    
def printPlotWav(filename):
    # Note: this code is borrowed
    # Note: FFTs are seem to be slow in python
    fs_rate, signal = wavfile.read(filename)
    print ("Frequency sampling", fs_rate)
    len_audio = len(signal.shape)
    print ("Channels", len_audio)
    if len_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    print ("Number of Samplings N", N)
    secs = N / float(fs_rate)
    print ("secs", secs)
    Ts = 1.0/fs_rate # sampling interval in time
    print ("Timestep between samples Ts", Ts)
    t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
    FFT = abs(scipy.fft(signal))
    FFT_side = FFT[range(N//2)] # one side FFT range
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(N//2)] # one side frequency range
    fft_freqs_side = np.array(freqs_side)
    plt.subplot(311)
    p1 = plt.plot(t, signal, "g") # plotting the signal
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(312)
    p2 = plt.plot(freqs, FFT, "r") # plotting the complete fft spectrum
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count dbl-sided')
    plt.subplot(313)
    p3 = plt.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count single-sided')
    plt.show()
    
    












def main():
   #plotFFT("sax.wav")
    processFile2('sax.wav')
   

if __name__ == '__main__':
    main()


