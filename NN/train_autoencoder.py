#!/usr/bin/env python
"""
An autoencoder tries to find a compressed representation of the input. 
The number of input nodes is equal to the number of output nodes, but the hidden
layer contains fewer nodes. The activations in the hidden layer encode the most
important  features of the data.

The autoencoder minimizes the following cost function (y = x)

  J(W,b;x)  1/2 ||h_{W,b}(x) - x||

"""

from NN import NeuralNetwork
from NN import DatabaseMNIST as Database
from NN.NeuralNetwork import FullyConnectedLayer

import os.path
import numpy as np
from scipy import optimize

def visualize_weights(W, shape):
    """
    given a weight matrix W_ij, plot the input x^(max) that maximizes the activation

       zi = sum_j W_ij x_j + b_i

    subject to ||x||^2=1. The constraint is enforced by a Lagrange multiplier lambda. The
    function to minimize

       Li = zi  + lambda (sum_j x_j^2 - 1)

    has a maximum for 

                    W_ij
       x_j = -----------------------
             sqrt{ sum_j (W_ij)^2 }

    The vector x_j is reshaped to a 2d matrix that can be plotted as an image.
    """
    import matplotlib.pyplot as plt

    # concatenate all filters into a column
    filters = []
    border = np.zeros( (1, shape[1]) )
    for i in range(0, W.shape[0]):
        # compute input that maximally activates feature i
        xmax = W[i,:] / np.sqrt( np.sum(W[i,:]**2 ) )
        filters.append( np.reshape(xmax, shape) )

    # plot filters in 2d grid arrangement
    n = int( np.sqrt(len(filters)) )
    fig, axarr = plt.subplots(nrows=n,ncols=n)

    ifilter = 0
    for i in range(0, n):
        for j in range(0, n):
            #axarr[i,j].set_title("Feature %d" % ifilter)
            axarr[i,j].imshow(filters[ifilter], cmap='gray')
            axarr[i,j].get_xaxis().set_visible(False)
            axarr[i,j].get_yaxis().set_visible(False)

            ifilter += 1
            if ifilter >= len(filters):
                break
        if ifilter >= len(filters):
            break
            
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # For the autoencoder the training set consists of randomly selected m x m  patches
    # from the MNIST images. The target output is equal to the input, (x,y=x).
    images_it = Database.read_idx_images(os.path.join(Database.data_dir, "train-images-idx3-ubyte") )

    # size of patches
    m = 28  #10 
    
    training_set = []
    for i,img in enumerate(images_it):
        if i > 60000:
            # only use first N images
            break
        # randomly select a m x m region in the image
        i0 = np.random.randint(0, img.shape[0]-m+1)
        j0 = np.random.randint(0, img.shape[1]-m+1)
        img_mxm = img[i0:i0+m,j0:j0+m]

        x = img_mxm.flatten()
        training_set.append( (x,x) )
        dim_in = len(x)

    nr_hidden_nodes = 25
    # autoencoder with 1 hidden layer, input and output dimensions are the same

    # try to load weights from previous optimization
    try:
        pydata_file = os.path.join(Database.data_dir, "trained_ac.pydata")
        print "load autoencoder from '%s'" % pydata_file
        ac = NeuralNetwork.NeuralNetwork( NeuralNetwork.load_layers( pydata_file ) )
        layers = ac.getLayers()
    except IOError as e:
        print e
        # build neural network of autoencoder
        layers = [
            FullyConnectedLayer(dim_in=dim_in,           dim_out=nr_hidden_nodes,
                                activation='sigmoid'),
            FullyConnectedLayer(dim_in=nr_hidden_nodes, dim_out=dim_in,
                                activation='sigmoid')
        ]

        ac = NeuralNetwork.NeuralNetwork(layers)
            
    # train autoencoder
    trainer = NeuralNetwork.Trainer(ac)
    trainer.setTrainingSet(training_set)

    print "Train neural network"
    wb0, J = trainer.objective_function()
    # train the neural network
    res = optimize.minimize(J, wb0, jac=True, method='L-BFGS-B',
                            options={'disp': True, 'gtol': 1.0e-7, 'maxiter':100})
    wb_opt = res.x
    print "optimized weights and bias"
    print wb_opt
    # evaluate cost function of optimized neural network on the training set
    cost, grad_cost = J(wb_opt)
    print "cost= %s" % cost

    # save weights
    pydata_file = os.path.join(Database.data_dir, "trained_ac.pydata")
    NeuralNetwork.save_layers( ac.getLayers(), pydata_file )
    print "trained neural network saved to '%s'" % pydata_file
    
    # visualize features
    input_layer = ac.getLayers()[0]
    W = input_layer.getWeights()
    visualize_weights(W, (m,m))

    # predicted reconstructed images
    import matplotlib.pyplot as plt
    for i,(xi,yi) in enumerate(training_set):
        yi_ac = ac.feed_forward(xi)

        f, axarr = plt.subplots(2)
        axarr[0].set_title("input")
        axarr[0].imshow(np.reshape(yi, (m,m)))
        axarr[1].set_title("reconstructed")
        axarr[1].imshow(np.reshape(yi_ac, (m,m)))

        plt.show()
        
        
