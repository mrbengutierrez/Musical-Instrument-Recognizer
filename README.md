<h1>Musical Instrument Decoder</h1>

<h2>Table of Contents</h2>
  <ol>
	<li><a href="#contributors"><b>Contributors</b></a></li>
	<li><a href="#code"><b>Code</b></a></li>
	<li><a href="#abstract"><b>Abstract</b></a></li>
	<li><a href="#introduction"><b>Introduction</b></a></li>
	<li><a href="#methods"><b>Methods</b></a>
	  <ol>
	    <li><a href="#dataset"><b>Dataset</b></a></li>
		<li><a href="#preprocessing 1"><b>Preprocessing</b></a></li>
		<li><a href="#neural network"><b>Neural Network</b></a></li>
	  </ol>
	</li>
	<li><a href="#experiments"><b>Experiments and Results</b></a>
	  <ol>
	    <li><a href="#preprocessing 2"><b>Preprocessing</b></a></li>
		<li><a href="#instrument number"><b>Number of Instruments</b></a></li>
		<li><a href="#pitch"><b>Pitch Training and Testing</b></a></li>
	  </ol>
	</li>
	</li>
	<li><a href="#conclusion"><b>Conclusion</b></a>
	  <ol>
	    <li><a href="#acknowledgments"><b>Acknowledgments</b></a></li>
		<li><a href="#references"><b>References</b></a></li>
	  </ol>
	</li>
  </ol>
  
<h2 id="contributors">Contributors</h2>
<i><strong>AKA The Instrumental Specialists</strong> </i>
  
<h3><strong>Ben Gutierrez</strong></h3>
<i>Massachusetts Institute of Technology</i>
  <ul>   
    <li>Wrote all project code, including:
	  <ul>
	    <li>preprocess.py</li>
		<li>neuralnet.py</li>
		<li>note_sorter.py</li>
		<li>api.py</li>
  </ul>
    </li>
	<li>Talked about preprocessing in final presentation</li>
	<li>Wrote about preprocessing and neural network in final paper</li>
	<li>Maintains the project for future updates</li>
  </ul>
  
<h3><strong>Ellen Mak</strong></h3>
<i>Boston University</i>
  <ul>
    <li>Contributed substantially to final presentation
      <ul>
	    <li>Answered all class questions</li>
	    <li>Wrote the PowerPoint talking notes</li>
	   </ul>
    </li>
    <li>Trained and tested instrumental data
	  <ul>
	    <li>Tuned neural network parameters to get a good model for the data</li>
        <li>Generated plots of models of interest</li>
	  </ul>
	</li>
	<li>Wrote about training and testing the data in the final paper
	  <ul>
	    <li>Also formatted all plots, reference, and style in latex NIPS format</li>
      </ul>
	</li>
  </ul>
	
<h3><strong>Rohan Pahwa</strong></h3>
<i>Boston College</i>
  <ul>
    <li>Contributed to final presentation
	  <ul>
	    <li>Talked about overall process, dataset, and introduction</li>
	  </ul>
	</li>
    <li>Wrote introduction and abstract for final paper</li>
	<li>Trained and tested notes data
	  <ul>
	    <li>Tuned neural network parameters to get a good model for the data</li>
		<li>Generated plots of models of interest</li>
      </ul>
	</li>
  </ul>
 
<h2 id="code">Code</h2>
  <p>
    Please open api.py to see the Application Program Interface for the Musical Instrument Decoder software. Please look at the demos in main to learn how to use the code. The demos will be replicated below.
  </p>

 

<h2 id="abstract">Abstract</h2>
  <p>
  Neural Networks have been applied to a multitude of testing scenarios in the past. The goal here was to use neural networks and audio samples from various instruments to both identify instruments and pitch. The intent was use these developed neural networks to create a tool that could combine both pitch and instrument information help create the skeleton of sheet music by instrument. Analysis was done using log based Fast Fourier Transform and Mel Frequency Cepstral Coefficients to simplify audio information to be easily processed by the neural network without loss of important audio characteristics.
  </p>
  

<h2 id="introduction">Introduction</h2>
  <p>
  Every instrument has its own special characteristics defined by its shape, its size, the length of its strings, how it was tuned, its age, its quality, etc. Underneath all these factors, the sound of an instrument is defined by its frequencies and resonances. Recognizing these patterns in frequencies and resonances is difficult for humans but an analysis of this data can allow a neural network to recognize and categorize this data.  What humans do excel at is the the ability to identify the difference between a cheap or expensive instrument on most occasions while still recognizing that in fact they are the same instrument. Every instrument has a unique pattern that defines it and can be used to identify it regardless of most of those other factors. In addition, musical pitches also have consistent patterns from instrument to instrument that can be identified. The goal is to train a neural network to identify instruments while ignoring all the other extraneous factors regarding the instrument itself including pitch. Fast Fourier Transform (FFT) and Mel Frequency Cepstral Coefficients (MFCC) were used to create frequency data plots to train the neural networks. Various levels of testing was performed to understand the importance of training iterations, sample sizes, and over-fitting. All the data was obtained from Philharmonia Orchestra Sound Samples to keep data simple and consistent. This limited the number of extraneous factors in our testing while providing us with consistent high quality data with large sample sizes for most instruments. 
  </p>
  
  <p>
  FFT and MFCC break down our processed audio samples in two ways. FFT gives us the frequencies for each sample and treats them as if they are periodic throughout the sample and MFCC gives us coefficients for a spectrogram to see frequency data over time. These combined, provide us with enough data to create input vectors for our neural network to train and identify instruments. The same processing methods can be used to process audio for pitch identification while ignoring octaves.
  </p>

<h2 id="methods">Methods</h2>
  <p>
  This section will discuss the data set we obtained, the preprocessing process, and the neural network we implemented. 
  </p>

<h3 id="dataset">Dataset</h3>
  <p>
  Our data set was taken from the Philharmonia Orchestra Sound Samples. The data had samples of different instruments, lengths, pitch, dynamics, and articulations. The database offered 20 instruments with a total of 13,686 samples. Below is a breakdown of the number of samples per instrument.
  </p>
  
  <ul>
    <li>banjo (74 samples)</li>
	<li>bass clarinet (945 samples)</li>
	<li>bassoon (721 samples)</li>
	<li>cello (889 samples)</li>
	<li>clarinet (847 samples)</li>
	<li>contrabassoon (711 samples)</li>
	<li>cor anglais (682 samples)</li>
	<li>double bass (853 samples)</li>
	<li>flute (879 samples)</li>
	<li>french horn (651 samples)</li>
	<li>guitar (107 samples)</li>
	<li>mandolin (81 samples)</li>
	<li>oboe (597 samples)</li>
	<li>percussion (149 samples)</li>
	<li>saxophone (732 samples)</li>
	<li>trombone (832 samples)</li>
	<li>trumpet (486 samples)</li>
	<li>tuba (973 samples)</li>
	<li>viola (974 samples)</li>
	<li>violin (1502 samples)</li>
  </ul>
  
  <p>
  The samples were initially mp3 files, but we decompressed them into wav files using a mp3 to wav converter. We then labeled all of our data by putting them into their respective folder. Overall, we only trained and tested with a subset of 6 instruments (banjo, cello, clarinet, flute, saxophone, tuba).
  </p>

<h3 id="preprocessing 1">Preprocessing</h3>
<h4>Method 1: FFT Approach</h4>
  <p>
  In order to tackle the problem of instrument recognition, we tried a frequency based approach. We took samples of audio data sampled at 44,100 Hz. We then down-sampled the signals in the time domain by a factor of 4. This had the effect of lowering our sample rate by a factor of 4. In addition, down-sampling has the effect of mapping higher frequencies to lower frequencies and can cause distortion. One way to prevent this distortion is to add a low pass filter with the cutoff frequency at the new sampling rate. We did not add a low pass filter because power of our signals at high frequencies was quite small. Next we extracted 2048 sample points in the center of our signal. After this step we took the fast-frequency transform (FFT) of the shrunken signal. This gave us the frequency content of our signal. We then removed the left hand The FFT breaks down the signal into complex components and produces both negative and positive frequency components. Because our signal is real the negative frequency components are redundant. After we removed the left hand side of the FFT, we then took the log of the each of the FFT frequencies. This is based on the idea of human hearing, as humans tend to associate exponential changes as linear ones. Finally in order to regularize our data, we divided each frequency component by the maximum amplitude in the frequency spectrum.
  </p>
<h4>Method 2: MFCC Approach</h4>  
  <p>
  The second method we used was Mel-Frequency Cepstral Coefficients. The idea behind this method is to generate a spectrogram of frequency versus time.
  </p>
  
  <p>
  First, we applied a pre-emphasis filter on our audio signal. The pre-emphasis filter helps to balance the frequency spectrum by weighting higher frequencies. Higher frequencies tend to to have much smaller magnitudes than lower ones.
  </p>
  
  <p><b><u>
  pre-emphasis filter (a=filter coefficient - 0.95 or 0.97)
  </u></b></p>
  
  <img src="https://github.com/mrbengutierrez/Musical-Instrument-Decoder/blob/master/images/eqn1_24.PNG" alt="Pre-Emphasis Equation">
  
  <p>
  After apply a pre-emphasizing filter, we split the signal into separate time frames. The frequency of the signal changes over time, and by splitting the data into time frames we can can capture this additional frequency content.
  </p>
  
  <p>
  Next, we applied a hamming window to each of the time frames. The FFT assumes that the signal is infinitely periodic. If the ends of the time signal do not match up, the signal will contain some additional high frequency noise not presence in the original signal. The hamming window helps correct that by bringing the ends of the time signal closer together. 
  </p>

  <p>
  Next we took the FFT of each frame to get the frequency content for each frame. Next we took the spectral density for each time frame. This describes the distribution of power of each of the frequency components in the signal. 
  </p>
  
  <p>
  power spectrum (N typically 256 or 512)
  </p>
  
  <p>
  In our next step, we created filter banks using triangular filter in order to transform the frequencies the spectral density to their corresponding frequency bands. We used the Mel-scale because it distinguishes higher frequency stronger than lower frequencies. Once we applied the filter bank across each of the power frames obtained a spectrogram with frequency along one axis and time along another axis. The equations below are the triangular filters applied to each of the power frames.
  </p>
  
  <p>
  We then applied the discrete time cosine transform (DCT). The DCT helps to decorrelate the mel-frequency cepstral ceofficients (MFCCs). We chose to use the lower 12 MFCCs because higher coefficients represent fast changes in frequency between two adjacent time frames. The frequency content from musical instrument signal should not be changing rapidly between time frames.
  </p>
  
  <p>
  Next we applied a sinusoidal filter to the MFCCs to penalize higher MFCCs. This has been done in speech recognition algorithms to help with signal with signals with high signal to noise ratios$^{[4]}$. We used the assumption that our signal to noise ratio was low because our audio data was recorded at a high sample rate of 44,100 Hz and with h, the sinusoidal filtering increases the flexibility of the algorithm if it were to be used in real life with noisy audio signals.
  </p>
  
  <p>
  The final step of pre-processing was to normalize and regularize the MFCCs. Regularization was accomplished by subtracting the mean of MFCCS for each time row. Normalization was accomplished by dividing the MFCCS by the max coefficient for each time row. By normalizing our data we insured that our inputs into our neural network would be in the range between [-1:1].
  </p>

<h3 id="neural network">Neural Network</h3>
  <p>
  We used a fully connected artificial neural net (ANN) as our choice of machine learning algorithm. We chose to use an ANN because of its popularity in image classification problems. An image can be thought of as a two-dimensional signal, which is similar to the concept of MFCCS with both time and frequency labels for each coefficient. 
  </p>

  <p>
  We also were not certain if the variation between instruments or notes was due to a linear combination among features. If it happened to be the case that the the variation among instruments or notes was due to a linear combinations of the features we would have used multivariate regression. In our log based FFT method, the features were the logs of the FFT components, whereas in the MFCC method the features were the coefficients themselves. In both methods, it is hard to know whether or not the output variables are linear combinations of the features.
  </p>
  
  <p>
  We used forward propagation that were used to calculate the hypothesis vector. We then used stochastic gradient descent without regularization in order learning from our input vectors.
  </p>
  
  <p>
  When training are data, we would control the number of iterations, the type of activation function, the number of layers, and the size of each layer. In addition, we randomized the order that our input feature vectors were sent into our training algorithm. We had first implemented a sequential process to when training with our input vectors, but notice improvements in our training accuracy when training using a random lottery based process.
  </p>
  
  <p><b><u>
  Cost Function
  </u></b></p>
  
  <b><i>J(t) = (y[i] - a[i])<sup>2</sup></i></b>
  
  <p>
    <i>J</i>: Cost of weight matrix <i>t</i><br>
	<i>y[i]</i>: Target values of layer <i>i</i><br>
	<i>a[i]</i>: Activation values of layer <i>i</i>
  </p>
     
  
<h2 id="experiments">Experiments and Results</h2>
  <p>
  In this section, we will discuss how our process evolved throughout this project and the results obtained at each step. 
  </p>

<h3 id="preprocessing 2">Preprocessing</h3>  
  <p>
  Throughout this project we have encountered many difficulties with deciding how to preprocess the data. Since we were working with audio, preprocessing was an essential part of obtaining the appropriate data to put into our neural network. We can see that changing the preprocessing step significantly changed our results. 
  </p>
  
  <p>
   First, our preprocessing steps include taking the Fast Fourier Transform of the sound input, then downsampling the data by a factor of 4. We then trained and tested the preprocessed instrument data through our neural network (learning rate=0.01, interval=100, activation function=sigmoid). The neural network had 1024 input nodes, one hidden layer with 100 nodes, and 3 output nodes. As seen on Figure 1, the cost function for 3 instruments decreased overall, but the graph would have consistent spikes throughout the graph. Also the training accuracy through each sample would initially shoot up to 100% then range somewhere between 97% to 99%. Although our training accuracy was 98.7%, the testing accuracy was 33.3%, which is as good as randomly guessing. We once obtained our best testing accuracy of 40%, but this was not consistent. 
  </p>
  
  <p>
  After getting these results, we decided that we needed to change our preprocessing methods. Instead of just taking the Fast Fourier Transform, we also took the Mel-Frequency Cepstral Coefficients. Inputting 3 instruments (cello, flute, and saxophone), our neural network (learning rate=1, interval=100, activation function=sigmoid) had 36 input nodes, 100 nodes in the hidden layer, and 3 output nodes. After training the preprocessed instrument data, we were able to get a 97.1% training accuracy with a desirable graph shown in Figure 2. After testing our test data, we were able to obtain a testing accuracy of 86.7%. 
  </p>

<h3 id="instrument number">Increasing the Number of Instruments</h3> 
  <p>
  After changing our preprocessing method, we tried training and testing with an increased number of instruments from 3 instruments (cello, flute, saxophone) to 5 instruments (cello, clarinet, banjo, flute, saxophone). Unfortunately we obtained a lower training accuracy of 81.3% and a testing accuracy of 62%. 
  </p>
  
  <p>
  Moving forward, we wondered if we could increase our accuracy by looking at the interval, or the number of iterations through our data. We thought that we might have been overfitting our data, making our training and testing accuracy lower. We first tried decreasing our interval from 100 to 20, which is one fifth since we are using 5 instruments instead of 3. Both the training and testing accuracy decreased. We then doubled our interval to 40 iterations. After, we increased and decreased our interval and we saw that the interval of 50 gave us the best result, as shown in Table 1. 
  </p>
  
  <table>
    <tr>
	  <td>Intervals</td>
	  <td>Training Accuracy</td>
	  <td>Testing Accuracy</td>
	</tr>
    <tr>
	  <td>20</td>
	  <td>68.8%</td>
	  <td>60%</td>
	</tr>
    <tr>
	  <td>40</td>
	  <td>77.6%</td>
	  <td>70%</td>
	</tr>
	  <td>45</td>
	  <td>76.4%</td>
	  <td>70%</td>
    <tr>
	  <td>50</td>
	  <td>78.5%</td>
	  <td>78%</td>
	</tr>
    <tr>
	  <td>55</td>
	  <td>78.4%</td>
	  <td>76%</td>
	</tr>
    <tr>
	  <td>100</td>
	  <td>81.3%</td>
	  <td>62%</td>
    </tr>	
  </table>

<h3 id="sample size">Increasing Sample Size</h3>   
  <p>
  Finally, we looked at the sampling size for each category and how it affected our training and testing accuracy. We saw that our sampling size for banjo was only 74, so we tried switching it out with an instrument with a larger sample size. We replaced the banjo with the tuba and ended up with a training accuracy of 94%  and a testing accuracy of 90%, shows in Figure 3. This was the highest testing accuracy that we obtained. 
  </p>

<h3 id="pitch">Pitch Training and Testing</h3>    
  <p>
  We also tried training and testing with the pitch. We saw that when we used 2 pitches (B and F), we obtained a training accuracy of 91.5% and a testing accuracy of 86.6%. Unfortunately, when we increased to three notes (B, C, and F), the accuracy for both training and testing drastically went down to 82% and 67.5%, as seen in Figure 4. Although we did not further investigate the problem, we think we might need to further preprocess the data or process the data specifically for the pitch. 
  </p>

<h2 id="conclusion">Conclusion</h2>  
  <p>
  Overall, we saw that the preprocessing step, the number of instruments, and the sample size affected our accuracy. Since we were dealing with audio, preprocessing was essential to our project. We saw that changing our methods from only taking the Fast Fourier Transform to taking the Fast Fourier Transform and the Mel-Frequency Cepstral Coefficient drastically changed our results for the better. Also when increasing the number of instruments, we came across the problem of overfitting and the importance of sample size. To further our study, we would want to try to train and test the pitch and the duration of each note to eventually generate a score when music is playing by overlaying the results obtained from both neural networks.
  </p>

<h3 id="acknowledgments">Acknowledgments</h3>
  <p>
    This work was done for the Machine Learning course (CS542) at Boston University.
  </p>

<h3 id="references">References</h3>  
  <ol>
    <li><p>
	Fayek, Haytham. <q>Speech Processing for Machine Learning: Filter Banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between.</q> <i>Speech Processing for Machine Learning: Filter Banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between</i>, Haytham Fayek, 21 Apr. 2016, haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html.
	</p></li>
    <li><p>
	Han, Yoonchang, et al. <q>Deep Convolutional Neural Networks for Predominant Instrument Recognition in Polyphonic Music</q>. <i>IEEE/ACM Transactions on Audio, Speech, and Language Processing</i>, vol. 25, no. 1, 2017, pp. 208 to 221., doi:10.1109/taslp.2016.2632307.
	</p></li>
	<li><p>
	Mitra, Vikramjit, et al. <q>Robust Features in Deep-Learning-Based Speech Recognition.</q> <i>New Era for Robust Speech Recognition</i>, 2017, pp. 187 to 217., doi:10.1007/978-3-319-64680-0_8.
	</p></li>
	<li><p>
	Yoshioka, T., and M.j.f. Gales. <q>Environmentally Robust ASR Front-End for Deep Neural Network Acoustic Models.</q> <i>Computer Speech & Language</i>, vol. 31, no. 1, 2014, pp. 65 to 86., sdoi:10.1016/j.csl.2014.11.008.
	</p></li>
  </ol>
  

  

  
s