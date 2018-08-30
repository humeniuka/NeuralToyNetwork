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
            # only use a batch of 50 000 images for training
            break
        x = img.flatten()
        y = np.array(label_vec)
        training_set.append( (x,y) )
        dim_in = len(x)
        dim_out = len(y)


    try:
        # load pretrained neural network ...
        pydata_file = os.path.join(Database.data_dir, "trained_nn.pydata")
        print "load neural network from '%s'" % pydata_file
        nn = NeuralNetwork.NeuralNetwork( NeuralNetwork.load_layers( pydata_file ) )
    except IOError as e:
        print "could not load neural network: %s" % e
        # ... or build neural network
        # The features are extracted by the first layer of an autoencoder trained separately
        pydata_file = os.path.join(Database.data_dir, "trained_ac.pydata")
        print "load autoencoder from '%s'" % pydata_file
        ac = NeuralNetwork.NeuralNetwork( NeuralNetwork.load_layers( pydata_file ) )
        layers = ac.getLayers()
        feature_layer = layers[0]
        # feature extractor is not optimized
        feature_layer.freezeAll()
    
        # 1 hidden layer with 30 nodes
        nr_hidden_nodes = 30

        nn = NeuralNetwork.NeuralNetwork([
            feature_layer, 
            FullyConnectedLayer(dim_in=feature_layer.dim_out,
                                dim_out=nr_hidden_nodes),
            FullyConnectedLayer(dim_in=nr_hidden_nodes, dim_out=10)   # output layer
        ])
        
    # train the neural network
    trainer = NeuralNetwork.Trainer(nn)

    print "Train neural network"
    trainer.setTrainingSet(training_set)
    p0, J = trainer.objective_function()
    res = optimize.minimize(J, p0, method='L-BFGS-B', jac=True,
                            options={'maxiter': 100, 'disp': True})
    """
    # mini-batch 
    trainer.setBatchSize(2000)
    p0, J = trainer.objective_function()
    # my implementation of Steepest Descent
    res = Optimize.minimize(J, p0, gtol=1.0e-6, method='Steepest Descent',
                            line_search_method=None, steplen=0.01,  maxiter=100,
                            callback=lambda x: trainer.new_random_batch())
    """
    """
    trainer.setBatchSize(2000)
    p0, J = trainer.objective_function()
    # scipy's minimize
    res = optimize.minimize(J, p0, method='L-BFGS-B', jac=True,
                            options={'maxiter': 100, 'disp': True},
                            callback=lambda x: trainer.new_random_batch())
    """
    """
    # train on randomly selected subset of training set
    #
    # The problem with my own implementation of BFGS is that for neural networks with
    # many parameters the Hessian does not fit into memory
    import random
    trainer.setTrainingSet(random.sample(training_set, 200))
    trainer.setBatchSize(None)
    p0, J = trainer.objective_function()
    res = Optimize.minimize(J, p0, gtol=1.0e-6, method="BFGS", maxiter=100)
    """
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
    pydata_file = os.path.join(Database.data_dir, "trained_nn.pydata")
    NeuralNetwork.save_layers( nn.getLayers(), pydata_file )
    print "trained neural network saved to '%s'" % pydata_file

    
