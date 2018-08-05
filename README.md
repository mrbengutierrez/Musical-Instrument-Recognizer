# Musical-Instrument-Decoder
This project attempts to decode musical notes from a musical instrument. This is a final project for cs 542 at Boston University

neuralnet_01 seems to work pretty ok. I conducted unit testing with various 2 bit logical boolean functions with different numbers of hidden layers and different layer sizes. [2,2,1] was the minimum neural net initialization that was necessary. Using sigmoid only converged half of the time. Using tanh as the activation function worked much better, and worked 100% of the time on the boolean function test cases. On a side note, there is weird negative predications sometimes when using tanh activation function. 

fft_features_01 is a work in progress

decoder_02.py, decoder_09.py can be ignored.
