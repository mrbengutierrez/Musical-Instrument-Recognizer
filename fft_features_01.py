import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
from matplotlib import pyplot as plt
import os


def plotFFT(filename):
    """Plots single sided FFT"""
    fs_rate, signal = wavfile.read(filename)
    len_audio = len(signal.shape)
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

            for each audio file, extract fft data, (optional: extract time data)
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

        
    
    
def printPlotWav(filename):
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
   plotFFT("sax.wav")
   

if __name__ == '__main__':
    main()


