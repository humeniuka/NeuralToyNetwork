#!/usr/bin/env python
"""
A multilayer feed-forward neural network is trained to recognize handwritten digits
in image with 28x28 pixels.
"""

from NN import NeuralNetwork
# import layer types
from NN.NeuralNetwork import FullyConnectedLayer, Convolutional2dLayer, \
    Pooling2dLayer, NonLinearityLayer
# import MNIST database
from NN import DatabaseMNIST as Database
from NN import Optimize

import os.path
import numpy as np
from scipy import optimize

if __name__ == "__main__":    
    # read training set (x,y)
    #  x: image with handwritten digit
    #  y: manually assigned digit
    images_it = Database.read_idx_images(os.path.join(Database.data_dir, "train-images-idx3-ubyte") )
    labels_it = Database.read_idx_labels(os.path.join(Database.data_dir, "train-labels-idx1-ubyte") )

    training_set = []
    for i,(img,label_vec) in enumerate(zip(images_it, labels_it)):
        if 50000 <= i:
            # only use a batch of 50000 images for training
            break
        x = img.flatten()
        y = np.array(label_vec)
        training_set.append( (x,y) )
        dim_in = len(x)
        dim_out = len(y)


    try:
        # load pretrained neural network ...
        pydata_file = os.path.join(Database.data_dir, "trained_cnn.pydata")
        print "load neural network from '%s'" % pydata_file
        nn = NeuralNetwork.NeuralNetwork( NeuralNetwork.load_layers( pydata_file ) )
    except IOError as e:
        print "could not load neural network: %s" % e
        # ... or build neural network
        m = 28
        ti = 4
        tj = 4
        nr_features = 5
        nr_hidden_nodes = (m/ti)*(m/tj)*nr_features
        nn = NeuralNetwork.NeuralNetwork([
            Convolutional2dLayer(ni=m,nj=m, nr_channels=1, mi=ti, mj=tj, nr_features=nr_features),
            Pooling2dLayer(ni=m,nj=m, nr_channels=nr_features, ti=ti, tj=tj),
            NonLinearityLayer(dim=nr_hidden_nodes),
            FullyConnectedLayer(dim_in=nr_hidden_nodes, dim_out=10)   # output layer
        ])
        
    # train the neural network
    trainer = NeuralNetwork.Trainer(nn)

    print "Train neural network"
    trainer.setTrainingSet(training_set)
    trainer.setBatchSize(1000)
    p0, J = trainer.objective_function()
    res = optimize.minimize(J, p0, method='L-BFGS-B', jac=True,
                            options={'maxiter': 100, 'disp': True},
                            )#callback=lambda x: trainer.new_random_batch())
    p_opt = res.x
    print "optimized parameters"
    print p_opt
    # evaluate cost function of optimized neural network on the training set
    cost, grad_cost = J(p_opt)
    print "cost= %s" % cost

    # test performance of the neural network on the training data
#    for i,(xi,yi) in enumerate(training_set):
#        print "x(%d) -> %s ?= %s" % (i, nn.feed_forward(xi), yi)

    # save the trained neural network
    pydata_file = os.path.join(Database.data_dir, "trained_cnn.pydata")
    NeuralNetwork.save_layers( nn.getLayers(), pydata_file )
    print "trained neural network saved to '%s'" % pydata_file

    
