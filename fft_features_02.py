import scipy.io.wavfile as wavfile
import scipy.fftpack
import numpy as np
import scipy
import matplotlib.pylab as plt

def processFile(filename,plot = False):
    """returns FFT amplitudes of filename"""
    #fs = sample rate, sound = multichannel sound signal
    fs1, sound = wavfile.read(filename)
    if fs1 != 44100:
        raise ValueError('Sampling rate should be 44100 for: ' + filename)
    sig1 = sound[:,0] #left channel
    N1 = len(sig1)

    fs2, sig2 = downsample(sig1,fs1,4)
    N2 = len(sig2)
    Ts2 = 1/fs2 # sampletime
    T2 = Ts2*N2 # total time (sec)
    new_T = 0.15
    sig3 = sig2[N2//2-int(new_T/2*N2):N2//2] + sig2[N2//2:N2//2+int(new_T/2*N2)]
    N3 = len(sig3) # num of samples
    N4 = 2048
    sig4 = sig3[0:N4]
    Ts4 = Ts2
    
    FFT = abs(scipy.fft(sig4))
    FFT_side = FFT[range(N4//2)]
    temp = []
    # normalize FFT
    for value in FFT_side:
        temp.append(value/sum(FFT_side))
    FFT_side = np.array(temp)
    plot = True
    if plot == True:
        freqs = scipy.fftpack.fftfreq(sig4.size, Ts4)
        freqs_side = np.array(freqs[range(N4//2)])
        plt.plot(freqs_side,FFT_side) # plotting the complete fft spectrum
        plt.show()
    return FFT_side

    
    

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


    

def main():
    processFile('sax.wav')
   

if __name__ == '__main__':
    main()


