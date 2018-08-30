#!/usr/bin/env python
"""
Validate the performance of a trained neural network on the t10k test data
from the MNIST database of handwritten digits
"""

from NN import NeuralNetwork
from NN import DatabaseMNIST as Database

import os.path
import numpy as np
from scipy import optimize

if __name__ == "__main__":
    import sys
    import os.path
    
    if len(sys.argv) < 2:
        print "Usage: python %s <name of neural network>" % os.path.basename(sys.argv[0])
        print "  validate performance of neural network for detecting handwritten digits"
        exit(-1)
        
    # name of neural network
    nn_name = sys.argv[1]    #  e.g. 'trained_cnn'
    assert nn_name in ["trained_nn", "trained_cnn"]
    # load the trained neural network
    #pydata_file = os.path.join(Database.data_dir, "trained_nn.pydata")
    #pydata_file = os.path.join(Database.data_dir, "trained_cnn.pydata")
    pydata_file = os.path.join(Database.data_dir, "%s.pydata" % nn_name)
    nn = NeuralNetwork.NeuralNetwork( NeuralNetwork.load_layers( pydata_file ) )
    
    # read test set (x,y)
    #  x: image with handwritten digit
    #  y: manually assigned digit
    images_it = Database.read_idx_images(os.path.join(Database.data_dir, "t10k-images-idx3-ubyte") )
    labels_it = Database.read_idx_labels(os.path.join(Database.data_dir, "t10k-labels-idx1-ubyte") )

    test_set = []
    for i,(img,label_vec) in enumerate(zip(images_it, labels_it)):
        x = img.flatten()
        y = np.array(label_vec)
        test_set.append( (x,y) )
        dim_out = len(y)

    # count number of successful classifications
    count_tests = 0
    count_success = 0
    # Some digits are easier to recognize, so we break the success rate down
    # by digits
    count_tests_by_class = [0 for i in range(0, dim_out)]
    count_success_by_class = [0 for i in range(0, dim_out)]
    # test performance of the neural network on the training data
    for i,(xi,yi) in enumerate(test_set):
        # correct label
        label = yi.tolist().index(1)
        # yi_nn is the probability vector assigned by the neural network
        yi_nn = nn.feed_forward(xi)
        label_nn = np.argmax(yi_nn)
        # statistics
        count_tests += 1
        if label == label_nn:
            count_success += 1
            count_success_by_class[label] += 1
        count_tests_by_class[label] += 1
        
        #print "x(%d) -> %s ?= %s" % (i, label_nn, label)

    recognition_rate = float(count_success)/float(count_tests)
    print "correct recognition rate: %1.3f" % recognition_rate
    print "test error rate (%%): %2.1f" % ( (1.0 - recognition_rate)*100.0 )

    # success rate for each digit
    print ""
    print "Digit         Success rate (%%)"
    print "==============================="
    for i in range(0, dim_out):
        print "  %d              %5.1f" % (i, float(count_success_by_class[i])/float(count_tests_by_class[i])*100.0 )
