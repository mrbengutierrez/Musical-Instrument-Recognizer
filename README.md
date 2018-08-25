
<h2>Table of Contents</h2>
  <ol>
	<li><a href="#Contributors"><b>Contributors</b></a></li>
	<li><a href="#Code"><b>Code</b></a></li>
	<li><a href="#Abstract"><b>Abstract</b></a></li>
	<li><a href="#Introduction"><b>Introduction</b></a></li>
	<li><a href="#Methods"><b>Methods</b></a>
	  <ol>
	    <li><a href="#Dataset"><b>Dataset</b></a></li>
	  </ol>
	</li>
  </ol>
  
<h2 id="Contributors">Contributors</h2>
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
 
<h2 id="Code">Code</h2>
  <p>
    Please open api.py to see the Application Program Interface for the Musical Instrument Decoder software. Please look at the demos in main to learn how to use the code. The demos will be replicated below.
  </p>

 

<h2 id="Abstract">Abstract</h2>
  <p>
  Neural Networks have been applied to a multitude of testing scenarios in the past. The goal here was to use neural networks and audio samples from various instruments to both identify instruments and pitch. The intent was use these developed neural networks to create a tool that could combine both pitch and instrument information help create the skeleton of sheet music by instrument. Analysis was done using log based Fast Fourier Transform and Mel Frequency Cepstral Coefficients to simplify audio information to be easily processed by the neural network without loss of important audio characteristics.
  </p>
  

<h2 id="Introduction">Introduction</h2>
  <p>
  Every instrument has its own special characteristics defined by its shape, its size, the length of its strings, how it was tuned, its age, its quality, etc. Underneath all these factors, the sound of an instrument is defined by its frequencies and resonances. Recognizing these patterns in frequencies and resonances is difficult for humans but an analysis of this data can allow a neural network to recognize and categorize this data.  What humans do excel at is the the ability to identify the difference between a cheap or expensive instrument on most occasions while still recognizing that in fact they are the same instrument. Every instrument has a unique pattern that defines it and can be used to identify it regardless of most of those other factors. In addition, musical pitches also have consistent patterns from instrument to instrument that can be identified. The goal is to train a neural network to identify instruments while ignoring all the other extraneous factors regarding the instrument itself including pitch. Fast Fourier Transform (FFT) and Mel Frequency Cepstral Coefficients (MFCC) were used to create frequency data plots to train the neural networks. Various levels of testing was performed to understand the importance of training iterations, sample sizes, and over-fitting. All the data was obtained from Philharmonia Orchestra Sound Samples to keep data simple and consistent. This limited the number of extraneous factors in our testing while providing us with consistent high quality data with large sample sizes for most instruments. 
  </p>
  
  <p>
  FFT and MFCC break down our processed audio samples in two ways. FFT gives us the frequencies for each sample and treats them as if they are periodic throughout the sample and MFCC gives us coefficients for a spectrogram to see frequency data over time. These combined, provide us with enough data to create input vectors for our neural network to train and identify instruments. The same processing methods can be used to process audio for pitch identification while ignoring octaves.
  </p>

<h2 id="Methods">Methods</h2>
  <p>
  This section will discuss the data set we obtained, the preprocessing process, and the neural network we implemented. 
  </p>

<h3 id="Dataset">Dataset</h3>
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
  
