
Summary
-------
This is a simple implementation of a neural network for recognizing handwritten digits
from the MNIST database. It's a toy project to convince potential employers I know
something about neural networks, ;)
Two different layer designs are tested:

1. An autoencoder is trained separately to detect features in m x m patches selected randomly
   from the training data. Its encoding part is then used as the first layer in a neural network
   consisting of two fully connected layers. The output layer has 10 nodes, each giving
   the probability that the input image is classified as one of the digits 0,1,...,9:
 
         feature extraction layer     - input layer
	 fully connected layer        - hidden layer
	 fully connected layer        - output layer

2. The second design consists of a convolutional layer followed by a pooling stage and
   a non-linearity. This combination of layers serves for extracting feature vectors which
   are then fed into a fully connected layer. The fully connected layer has gain 10 output
   nodes giving the probabilities for detecting a particular digit:

         2d convolutional layer       - input layer
	 2d pooling layer
	 non-linearity layer
	 fully connected layer        - output layer


Installation
------------
To be able to run the scripts from the folder NN/, you have to add the current folder
to your PYTHONPATH:

   export PYTHONPATH=/your/path/to/NeuralToyNetwork:$PYTHONPATH

The scripts were tested with python 2.7 and recent versions of numpy (1.11.0) and scipy (0.17.0). 


Training
--------
Since training a neural network written in python on the CPU takes a long time,
pretrained networks are stored in the DATA/ folder. When training a network further,
the previously optimized parameters are read from disk and are used to initialize
the network. After a few hundred optimization steps the improved parameters are written
back to disk. In this way the network can be improved iteratively.

To train the autoencoder run

   python train_autoencoder.py

The trained autoencoder will be written to `DATA/trained_ac.pydata`. Images of the
optimized features are displayed at the end.

To train the fully connected network 1 run

   python train_nn.py

The trained neural network will be written to `DATA/trained_nn.pydata`.

To train the convolutional network 2 run

   python train_cnn.py

The trained neural network will be written to `DATA/trained_cnn.pydata`.


Validation
----------
To assess the performance of the trained neural network its output on the training
data is compared with the expected assignments.

To evaluate neural network 1, type

   python handwriting_recognition_testing.py trained_nn

which should produce the following table:

   correct recognition rate: 0.927
   test error rate (%): 7.3

   Digit         Success rate (%)
   ===============================
     0               96.3
     1               97.1
     2               91.6
     3               90.5
     4               93.1
     5               90.1
     6               95.2
     7               92.7
     8               89.5
     9               90.3

Similarly, to evaluate the convolutional network 2, type

    python handwriting_recognition_testing.py trained_cnn

which results in the following table (numbers may differ depending on the level of training):

    correct recognition rate: 0.814
    test error rate (%): 18.6

    Digit         Success rate (%)
    ===============================
      0               89.4
      1               93.3
      2               80.2
      3               78.7
      4               74.9
      5               69.5
      6               90.8
      7               77.6
      8               73.4
      9               83.2


Depending on the number of optimization cycles the success rates might be higher. 
The conclusion that the convolutional network 2 is inferior to network 1 is probably
not justified, since further training could improve network 2.


Files
-----
 NN/NeuralNetwork.py   -  Layer types, forward and backward propagation
 NN/DatabaseMNIST.py   -  code for reading images and labels from MNIST database
 NN/Optimize.py        -  BFGS algorithm for optimization
 
 NN/train_nn.py,
 NN/train_cnn.py,      -  scripts for training the autoencoder and the neural networks 1 and 2
 NN/train_ac.py                 
 NN/handwriting_recognition_testing.py
	               -  script for testing performance of neural networks on the MNIST data

 DATA/                 -  contains the training data (bitmaps of digits and manually assigned labels)
     	                  and the pretrained neural networks
			  (`trained_ac.pydata`, `trained_nn.pydata` and `trained_cnn.pydata`)


Author
------
Alexander Humeniuk


