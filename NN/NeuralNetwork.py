#!/usr/bin/env python
"""
multilayer feed-forward neural network with different layer types:
  -  fully connected layer
  -  convolutional layer
  -  pooling layer
  -  non-linearity layer
"""
from __future__ import with_statement

import numpy as np
import numpy.linalg as la

import random

# use the same random numbers every time
#np.random.seed(0)

class Layer(object):
    """
    base class for any layer

    derived classes are 
      - convolutional layer, CN
      - fully connected layer, FC
      - pooling layer, Pool

    The derived classes have to overload the member functions
      - __init__()
      - feed_forward()
      - propagate_back()

    """
    def __init__(self, nr_param, dim_in, dim_out):
        """
        nr_param: number of internal parameters (weights, bias or filters)
        dim_in: number of input units to this layer, dimension of input vector `x`
        dim_out: number of output units of this layer, dimension of output vector `y`
        """
        self.nr_param = nr_param
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        # vector with internal parameters that can be trained, the parameters are initialized
        # with small random numbers
        eps = 1.0e-2
        self.parameters = eps* 2.0 * (np.random.rand( self.nr_param ) - 0.5)
        # array with 0's and 1's, for each parameter i the integer self.active[i] indicates whether
        # this parameter is active (1) or not (0). 
        self.active = np.ones(self.nr_param)
        # gradient of the cost function w/r/t parameters 
        self.gradients = np.zeros(self.nr_param)

    def freezeAll(self):
        """freeze parameters so that this layer is not optimized"""
        self.active *= 0

    def unfreezeAll(self):
        """all parameters are activated for optimization"""
        self.active = np.ones(self.nr_params)
        
    def getDimensions(self):
        return (self.dim_in, self.dim_out)
        
    def getParameters(self, only_active=False):
        """
        retrieve vector with values of adjustable parameters defining the layer
        
        Optional
        --------
        only_active  :   flag which controls whether all parameter are returned (False)
                         or only those which are not frozen during the optimization (True)
        """
        if only_active == True:
            params_active = self.parameters[self.active == 1]
            p = params_active
        else:
            p = np.copy( self.parameters )
        return p
    
    def setParameters(self, p, only_active=False):
        """
        set values of adjustable parameters
        """
        if only_active == True:
            self.parameters[self.active == 1] = p
        else:
            self.parameters = np.copy( p )

    def getGradients(self, only_active=False):
        """
        derivatives of the layer output with respect to the adjustable parameters
        """
        if only_active == True:
            grad_active = self.gradients[self.active == 1]
            dJdp = grad_active
        else:
            dJdp = np.copy( self.gradients )
        return dJdp

    def feed_forward(self, x):
        """
        compute output of this layer given the input vector `x`

           y = f(x)

        Parameters
        ----------
        x: 1d numpy array of dimension (dim_in)

        Returns
        -------
        y: 1d numpy array of dimension (dim_out)
        """
        assert x.shape == (self.dim_in)
        y = np.zeros(self.dim_out)
        return y
        
    def propagate_back(self, dJdy):
        """
        propagate gradients back through this layer

        Given the gradient of the cost function `J` w/r/t the output of the layer, `y`,
        the gradient w/r/t to its input is computed.

           dJ/dx = dJ/dy df/dx

        In addition the gradients w/r/t to the internal parameters (weights, bias),
        represented by the vector p, are updated

           dJ/dp = dJ/dy df/dp

        Parameters
        ----------
        dJdy: 1d numpy array of dimension (dim_out), gradient w/r/t to output

        Returns
        -------
        dJdx: 1d numpy array of dimension (dim_in), gradient w/r/t to input
        """
        assert dJdy.shape == (self.dim_out)
        dJdx = np.zeros(self.dim_in)
        return dJdx

class ActivationFunctions:
    """
    This class encapsulates different non-linear activation functions.

    Example:

      >>> f, dfdz = ActivationFunctions.getFunc('sigmoid')

      >>> z = 0.1
      >>> f(z), dfdz(z)

    """
    @staticmethod
    def getFunc(name):
        if name == "sigmoid":
            # sigmoid activation function f(z) = 1/(1+exp(-z))
            def f(z):
                return 1.0/(1.0+np.exp(-z))
            # derivative of activation function df/dz
            def dfdz(z):
                return np.exp(-z)/(1 + np.exp(-z))**2
        elif name == "relu":
            # linear rectifier function  f(z) = max(0, z)
            def f(z):
                return np.maximum(0, z)
            def dfdz(z):
                #         /  0     if z <= 0
                # df/dz = {
                #         \  1     if z > 0
                return 0.5*(np.sign(z)+1.0)
        else:
            raise ValueError("No activation function with name '%s'!" % name)
            
        return f, dfdz

    
class FullyConnectedLayer(Layer):
    def __init__(self, dim_in='', dim_out='', activation='sigmoid'):
        # all nodes are connected, the number of weights is dim_in*dim_out
        # and there are dim_out biases
        nr_param = dim_in*dim_out + dim_out
        # 
        super(FullyConnectedLayer, self).__init__(nr_param, dim_in, dim_out)

        # select non-linear activation function and derivative
        self.activation = activation
        self.f, self.dfdz = ActivationFunctions.getFunc(activation)

        # intermediate quantities that are updated in each forward propagation
        # current inputs
        self.x = np.zeros(dim_in)
        #   z_i denotes the total weighted sum of inputs to unit i
        self.z = np.zeros(dim_out)
        #  a_i is the activation or output of unit i
        self.a = np.zeros(dim_out)
        
    def feed_forward(self, x):
        """
           y(i) = f( sum_j W(i,j) x(j) + b(i) )
        """
        # the first dim_in*dim_out elements in the parameter vector
        # are the weights, the remaining elements are the basis
        nw = self.dim_in*self.dim_out
        weights = self.parameters[:nw]
        bias = self.parameters[nw:]
        W = np.reshape(weights, (self.dim_out, self.dim_in))

        # save input
        self.x = x
        # activation
        self.z = np.dot(W, x) + bias
        # output
        self.a = self.f(self.z)
        y = np.copy(self.a)
        return y

    def propagate_back(self, dJdy):
        """
        given the gradient of the cost function J w/r/t the output `y` of this layer
        compute 
          -  the gradient of cost function J w/r/t the parameters of this layer,  dJ/dw 
          -  and the gradient of J w/r/t the input `x` to this layer

        Only dJ/dx is returned, the gradient dJ/dw is stored internally.
        """
        nw = self.dim_in*self.dim_out
        weights = self.parameters[:nw]
        W = np.reshape(weights, (self.dim_out, self.dim_in))
        dfdz = self.dfdz(self.z)

        # compute Jacobian for previous layer
        # dJ/dx(i) = sum_j dJ/dy(j) f'(z(j)) W(j,i) 
        dJdx = np.dot(dJdy*dfdz, W)

        # update gradients w/r/t parameters
        #
        # gradient on weights
        #  dJ/dW(i,j) = dJ/dy(i) f'(z(i)) x(j)
        gradW = np.outer(dJdy*dfdz, self.x)
        # gradient on bias
        #  dJ/db(i) = dJ/dy(i) f'(z(i))
        gradB = dJdy*dfdz

        self.gradients[:nw] = gradW.flatten()
        self.gradients[nw:] = gradB

        return dJdx

    #### functions specific to this layer type ###
    def getWeights(self):
        # the first dim_in*dim_out elements in the parameter vector
        # are the weights, the remaining elements are the basis
        nw = self.dim_in*self.dim_out
        weights = self.parameters[:nw]
        bias = self.parameters[nw:]
        W = np.reshape(weights, (self.dim_out, self.dim_in))
        return W

from scipy import signal

class Convolutional2dLayer(Layer):
    def __init__(self, ni='', nj='', nr_channels='', mi='', mj='', nr_features=''):
        """
        convole the input volume with shape (ni,nj, nr_channels) where ni and nj correspond to
        width and height of a two-dimensional image with `nr_features` filters with width
        mi and height mj.

             y(i,j;n) = sum_m sum_p sum_q w(p,q;m,n) x(i-(p-p0), j-(q-q0); m)


                      = sum_m convoled2d{ x(:,:, m), w(:,:,m,n) }

        with 
        
             p0 = mi/2  (if mi even)  or  mi/2-1 (mi odd)
             q0 = mj/2  (if mj even)  or  mj/2-1 (mj odd)

        The filter is centered on each pixel. Elements in x and w outside the valid ranges
        are padded with zeros. The convolutional layer takes an input volume, convolves
        it with all filters and produces and output volume:

             V_in in R^(ni,nj,nr_channels) -------> V_out in R^(ni,nj,nr_features)

        The input and output volumes are reshaped into 1d vectors.

        Chain rule for transforming gradient w/r/t output, dJ/dy, into gradient w/r/t input, dJ/dx:

                                            d J             d y(i',j';n')
           dJ/dx(i,j;n) = sum_(i',j',n')  --------------  ---------------
                                           d y(i',j';n')    d x(i,j;n)

                                            d J
                        = sum_(i',j',n')  --------------  w(i'-i+p0, j'-j+q0; n', n)
                                           d y(i',j';n')

                                                                d J
                        = sum_(i',j',n') w(i',j';n',n) ----------------------------
                                                        d y(i'+(i-p0),j'+(j-q0);n')

                        = sum_n' correlate2d{ dJ/dy(:,:,n'), w(:,:,n',n) }

        
        Chain rule for gradients w/r/t to filter parameters w:

                                              d J            d y(i',j';n')
           dJ/dw(p,q;m,n) = sum_(i',j',n') -------------  ----------------
                                           d y(i',j';n')    d w(p,q;m,n)

                                           d J
                          = sum_(i',j') -------------  x(i'-(p-p0),j'-(q-q0); m)
                                         d y(i',j';n)

        shift summation variables
        
           i = i'-(p-p0)    <=>   i' = i+(p-p0)
           j = j'-(q-q0)    <=>   j' = j+(q-q0)

        and sum over i and j instead of i' and j':

                                                           d J
           dJ/dw(p,q;m,n) = sum_(i',j') x(i,j; m)  -----------------------
                                                   d y(i+(p-p0), j+(q-q0); n)

          
                          = correlate2d{ dJdy(:,:, n), x(:,:, m) }

        Only those elements of the correlation function are needed that correspond
        to the elements in the filter `w` that are not set to zero because of the 
        zero-padding. 
        """                            
        #
        self.shape_in = (ni,nj, nr_channels)
        self.shape_out = (ni,nj, nr_features)
        self.shape_features = (mi,mj, nr_channels, nr_features)
        
        nr_param = nr_features*nr_channels*mi*mj
        dim_in = ni*nj*nr_channels
        dim_out = ni*nj*nr_features
        #
        super(Convolutional2dLayer, self).__init__(nr_param, dim_in, dim_out)

        # (p0,q0) and (i0,j0) are offsets for shifting the filter to the center
        # and for recovering the filter gradients of shape (mi,mj) from the
        # correlated full image.
        self.mi = mi
        self.mj = mj
        if mi % 2 == 1:
            self.p0 = mi/2
        else:
            self.p0 = mi/2-1
        if mj % 2 == 1:
            self.q0 = mj/2
        else:
            self.q0 = mj/2-1
        if ni % 2 == 1:
            self.i0 = ni/2
        else:
            self.i0 = ni/2-1
        if nj % 2 == 1:
            self.j0 = nj/2
        else:
            self.j0 = nj/2-1
        
        # intermediate quantities that are updated in each forward propagation
        # current inputs
        self.x = np.zeros(self.shape_in)

    def feed_forward(self, x):
        mi,mj, nr_channels, nr_features = self.shape_features
        # get parameters for all feature maps
        w = np.reshape( self.parameters, self.shape_features )
        # input volume
        x = np.reshape(x, self.shape_in)
        # save current input, we need it later in the backpropagation
        self.x = x
        # output volume
        y = np.zeros(self.shape_out, dtype=float)
        # convolution and scalar product with all feature maps
        for n in range(0, nr_features):
            for m in range(0, nr_channels):
                y[:,:,n] += signal.convolve2d(x[:,:,m], w[:,:,m,n],
                                              boundary='fill', mode='same')
                
        return y.flatten()

    def propagate_back(self, dJdy):
        dJdy = np.reshape(dJdy, self.shape_out)
        mi,mj, nr_channels, nr_features = self.shape_features
        # get parameters for all feature maps
        w = np.reshape( self.parameters, self.shape_features )
        
        # compute Jacobian for previous layer
        dJdx = np.zeros(self.shape_in, dtype=float)
        for n in range(0, nr_features):
            for m in range(0, nr_channels):
                dJdx[:,:,m] += signal.correlate2d(dJdy[:,:,n], w[:,:,m,n],
                                                  boundary='fill', mode='same')
        # update gradients w/r/t parameters
        dJdw = np.zeros(self.shape_features, dtype=float)
        # The gradients dJ/dw are only a small rectangular block [sp:ep,sq:eq]
        # in the full image correlate2d(dJdy, x). 
        sp = self.i0-self.p0
        ep = sp+mi
        sq = self.j0-self.q0
        eq = sq+mj
        for n in range(0, nr_features):
            for m in range(0, nr_channels):
                dJdw[:,:,m,n] = signal.correlate2d(dJdy[:,:,n], self.x[:,:,m],
                                                  boundary='fill', mode='same')[sp:ep,sq:eq]

        self.gradients = dJdw.flatten()
        
        return dJdx.flatten()

    
class PoolingFunctions:
    """
    This class encapsulates different pooling functions.

    Example:

      >>> pool, deriv_pool = PoolingFunctions.getFunc('max')

      >>> pool(np.array([[1,2],[3,4]]))

    """
    @staticmethod
    def getFunc(name):
        if name == 'max':
            def pool(tile):
                # 
                return np.amax(tile)
            def deriv_pool(tile):
                # gradient of pooling function w/r/t to elements of tile
                # Find the largest element in the tile
                ijmax = np.argmax(tile)
                imax,jmax = np.unravel_index(ijmax, tile.shape)
                # The gradient w/r/t the largest element is 1, w/r/t all other elements 0
                deriv = np.zeros(tile.shape)
                deriv[imax,jmax] = 1.0
                return deriv
            
        elif name == 'avg':
            def pool(tile):
                return np.mean(tile)
            def deriv_pool(tile):
                deriv = np.ones(tile.shape) / tile.size
                return deriv
                        
        else:
            raise ValueError("Unknown pooling function '%s'!" % name)
            
        return pool, deriv_pool

    
class Pooling2dLayer(Layer):
    def __init__(self, ni='', nj='', nr_channels='', ti='', tj='', pooling='max'):
        """
        The input image with width `ni` and height `nj` is subdivided into non-overlapping
        tiles of shape `ti`x`tj`.  For each tile only a single value is passed on to the
        next layer, thus reducing the size of the input volume from

            (ni,nj,nr_channels)   to   (ceil(ni/ti), ceil(nj/tj), nr_channels)

        Each channel is processed separately. In `max`-pooling only the value of the largest
        element in a tile is passed on, in `avg`-pooling the average value over a tile is passed on.

        Parameters:
        -----------
        (ni,nj): width and height of input image
        nr_channels: number of channels in input image
        (ti,tj): size of tile, the input image is padded with zero if the tiles do not fit exactly
        pooling: string, type of pooling function, can be 'max' or 'avg'.
        """
        # size of tiles
        self.ti = ti
        self.tj = tj
        # dimensions of input volume
        self.shape_in = (ni,nj, nr_channels)
        # number of tiles in x-direction  ceiling(ni/ti)
        self.shape_out = (int(np.ceil(ni/ti)), int(np.ceil(nj/tj)), nr_channels)
        # a pooling layer has no adjustable parameters
        nr_param = 0
        dim_in = np.prod(self.shape_in)
        dim_out = np.prod(self.shape_out)
        #
        super(Pooling2dLayer, self).__init__(nr_param, dim_in, dim_out)

        # select pooling function
        self.pooling = pooling
        self.pool, self.deriv_pool = PoolingFunctions.getFunc(pooling)
        
        # intermediate quantities that are updated in each forward propagation
        # current inputs
        self.x = np.zeros(self.shape_in)

    def feed_forward(self, x):
        ni,nj, nr_channels = self.shape_in
        # input volume
        x = np.reshape(x, self.shape_in)
        # output volume
        y = np.zeros(self.shape_out)
        # 
        nr_tiles_i, nr_tiles_j, nr_channels = self.shape_out

        for n in range(0, nr_channels):
            # apply pooling to each channel `n` separately
            for i in range(0, nr_tiles_i):
                istart = i*self.ti
                iend = min((i+1)*self.ti, ni)
                for j in range(0, nr_tiles_j):
                    jstart = j*self.tj
                    jend = min((j+1)*self.tj, nj)
                    
                    tile = x[istart:iend, jstart:jend, n]
                    y[i,j,n] = self.pool(tile)

        return y.flatten()

    def propagate_back(self, dJdy):
        # gradient w/r/t to output volume y
        dJdy = np.reshape(dJdy, self.shape_out)
        # gradient w/r/t input volume x
        dJdx = np.zeros(self.shape_in, dtype=float)
        #
        ni,nj, nr_channels = self.shape_in
        nr_tiles_i, nr_tiles_j, nr_channels = self.shape_out

        # compute gradient of cost function w/r/t input x, dJdx
        for n in range(0, nr_channels):
            # apply pooling to each channel `n` separately
            for i in range(0, nr_tiles_i):
                istart = i*self.ti
                iend = min((i+1)*self.ti, ni)
                for j in range(0, nr_tiles_j):
                    jstart = j*self.tj
                    jend = min((j+1)*self.tj, nj)

                    tile = self.x[istart:iend, jstart:jend, n]
                    dJdx[istart:iend, jstart:jend, n] = dJdy[i,j,n] * self.deriv_pool(tile)


        # There are no free parameters, so no need to compute dJ/dw
        return dJdx.flatten()
    
class NonLinearityLayer(Layer):
    def __init__(self, dim='', activation='sigmoid'):
        """
        apply a non-linear activation function `f` elementwise

          y = f(x)
        """
        # no free parameters
        # input and output dimensions remain the same
        super(NonLinearityLayer, self).__init__(0, dim, dim)

        # select non-linear activation function and derivative
        self.activation = activation
        self.f, self.dfdz = ActivationFunctions.getFunc(activation)

        # intermediate quantities that are updated in each forward propagation
        # current inputs
        self.x = np.zeros(dim)

    def feed_forward(self, x):

        # save current input
        self.x = x
        # apply non-linear function element-wise to input vector
        y = self.f(x)
        return y

    def propagate_back(self, dJdy):
        # chain rule   dJ/dx = dJ/dy dy/dx = dJ/dy f'(x)
        dJdx = dJdy * self.dfdz(self.x)

        return dJdx
    
        
class NeuralNetwork(object):
    """
    a neural network has the same interface as a single layer
    """
    def __init__(self, layers):
        self.layers = layers        
        # number of layers
        self.nl = len(self.layers)
        #
        self.dim_in = self.layers[0].getDimensions()[0]
        self.dim_out = self.layers[-1].getDimensions()[1]

        # check that layers have the correct input and output dimensions to be stacked together
        for l in range(0, self.nl-1):
            dim_out_l = self.layers[l].getDimensions()[1]
            dim_in_lp1= self.layers[l+1].getDimensions()[0] 
            # out dimension of layer l == in dimension of layer l+1
            assert dim_out_l == dim_in_lp1, \
                "Out dimension of layer %d (%d) does not match in dimension of layer %d (%d)!" % (l, dim_out_l, l+1, dim_in_lp1)
            
    def getDimensions(self):
        return (self.dim_in, self.dim_out)

    def getParameters(self, only_active=False):
        """
        concatenate the parameters of all layers
        """
        p = []
        for l in range(0, self.nl):
            pl = self.layers[l].getParameters(only_active=only_active)
            p.append( pl )
        p = np.hstack(p)
        return p
    
    def setParameters(self, p, only_active=False):
        """
        redistribute the elements of the parameter vector over the individual layers
        """
        start = 0
        end = 0
        for l in range(0, self.nl):
            # length of chunk of parameters belonging to layer l
            num_param_l = len( self.layers[l].getParameters(only_active=only_active) )
            end = start + num_param_l
            # set parameter of layer l
            pl = p[start:end]
            self.layers[l].setParameters(pl, only_active=only_active)
            # move to parameters for next layer
            start = end
        assert end == len(p)
            
    def getGradients(self, only_active=False):
        """
        concatenate gradients w/r/t parameter from all layers
        """
        dJdp = []
        for l in range(0, self.nl):
            dJdpl = self.layers[l].getGradients(only_active=only_active)
            dJdp.append( dJdpl )
        dJdp = np.hstack(dJdp)
        return dJdp
    
    def feed_forward(self, x):
        """
        evaluate the neural network with the current parameters

              y = f_nn(x)

        by propagating forward through all layers
        """
        y = x
        for l in range(0, self.nl):
            y = self.layers[l].feed_forward(y)
        return y
            
    def propagate_back(self, dJdy):
        """
        propagate errors backward through the layer and update gradients
        w/r/t to the parameters of the neural network
        """
        dJdx = dJdy
        # from layer N to layer 0
        for l in range(self.nl-1, -1, -1):
            dJdx = self.layers[l].propagate_back(dJdx)

        return dJdx

    def getLayers(self):
        return self.layers
    

class Trainer(object):
    def __init__(self, nn):
        self.nn = nn
        self.batch_size = None
        
    def setTrainingSet(self, training_set):
        """        
        set the training data used for optimizing the neural network

        Parameters
        ----------
        training_set: list of tuples (x,y) with input x and desired output y
        """
        dim_in, dim_out = self.nn.getDimensions()
        # check that training data has the correct dimensions
        for i,(xi,yi) in enumerate(training_set):
            assert len(xi) == dim_in, "Neural network has %d input units, input vector x^(%d) in training set has different dimension (%d)!" % (dim_in, i, len(xi))
            assert len(yi) == dim_out, "Neural network has %d output units, output vector y^(%d) in training set has different dimension (%d)!" % (dim_out, i, len(yi))
            
        self.training_set = training_set
        self.mini_batch = self.training_set
    #
    # Mini-Batches for stochastic gradient descent
    #
    def setBatchSize(self, batch_size=None):
        """
        In stochastic gradient descent, the error and gradient is only calculated for a
        random selection of n=`size` examples from the training date. In each iteration step the
        selection changes, so that the error surface is not a continuous function anymore.
        Therefore the optimization algorithm should only use the instantaneous gradient to 
        determine the new parameters (e.g. steepest descent is fine but BFGS will not work). 
        In order not to interfere with linesearches the minibatch should only change after each
        macro iteration. A call to `new_random_batch()` will select randomly a new mini batch.

        If the batch size is set to `None`, the entire training set is used.
        
        Parameters
        ----------
        size: size of each minibatch
        """
        self.batch_size = batch_size
        self.new_random_batch()
        
    def get_current_batch(self):
        return self.mini_batch

    def new_random_batch(self):
        """
        select randomly a new mini-batch, used as a callback function to be executed
        at the end of each optimization step
        """
        # select mini-batch
        if self.batch_size == None:
            # use the entire training set
            self.mini_batch = self.training_set
        else:
            # select randomly `batch_size` examples (xi,yi) from the training set
            self.mini_batch = random.sample(self.training_set, self.batch_size)
        
    #
    # Error function and gradients
    #        
    def error_gradients(self, x, y):
        """
        compute error of neural network for a single training example (x,y)

           J(p;x,y) = 1/2 || y - h_(p)(x) ||^2

        using forward propagation and its gradients
        with respect to the parameters p using backpropagation.

        Parameters
        ----------
        x,y: input and desired output for a single training example

        Returns
        -------
        J: scalar, deviation squared between the output of the 
             neural network y_nn and the desired output y, 1/2 ||y - y_nn||^2
        dJdp: 1d numpy array, gradient of error w/r/t to the parameters of
             the neural network
        """
        # evaluate neural network,
        # 1/2 |y - y_nn|^2 is the error of the neural network which we wish to minimize
        y_nn = self.nn.feed_forward(x)
        J = 0.5 * la.norm( y - y_nn )**2
        # gradient of error function
        dJdy = - (y - y_nn)    # gradient dJdy = dJ/dy_nn NOT dJ/dy

        self.nn.propagate_back(dJdy)

        # gradients w/r/t parameters 
        dJdp = self.nn.getGradients()
        
        return J, dJdp

    def cost_function(self, training_set):
        """
        compute the mean error over a training set with m examples (x(i),y(i))
        
           J(p) = 1/m sum_i J(p;x(i),y(i))
        """
        J = 0.0
        #
        m = len(training_set)
        # add squared errors
        for (x,y) in training_set:
            y_nn = self.nn.feed_forward(x)
            Ji = 0.5 * la.norm(y - y_nn)**2
            J += 1.0/m * Ji

        return J

    def objective_function(self):
        """
        prepare objective function for optimizing the neural network. 
        Before calling this function a training set must have been assigned using `setTrainingSet()`.

        Returns
        -------
        p:  current values of parameterss
        J: callable function that computes J(p) and its gradients
        """
        p0 = self.nn.getParameters()
        
        def f(p):
            """
            cost function J(p) that should be optimized as a function of the
            weights W and bias b. It computes J and the gradients dJ/dp
            """
            self.nn.setParameters(p)
            J = 0.0
            dJdp = np.zeros(p0.shape)

            mini_batch = self.get_current_batch()
            
            m = len(mini_batch)
            for (xi,yi) in mini_batch:
                Ji, dJidp = self.error_gradients(xi,yi)
                
                J += 1.0/m * Ji
                dJdp += 1.0/m * dJidp
                
            return J, dJdp

        return p0, f

##### functions for saving and loading (trained) neural networks ########
import copy

def save_layers(layers, filename_py):
    """
    save data for all layers to a human-readable file

    Example:
    
       >>> nn = NeuralNetwork(layers)
       >>> ...
       >>> save_layers(nn.getLayers(), "/tmp/layers.pydata")
    
    """
    import pprint
    # convert class instances to dictionaries
    layer_dics = []
    for layer in layers:
        layer_dic = copy.deepcopy( layer.__dict__ )
        # We need to save the class type so that we can cast the data
        # back to the appropriate class
        layer_dic['layer_type'] = layer.__class__.__name__
        # only the name of the activation function can be saved
        if hasattr(layer, "activation"):
            del layer_dic['f']
            del layer_dic['dfdz']
        # only the name of the pooling function can be saved
        if hasattr(layer, "pooling"):
            del layer_dic['pool']
            del layer_dic['deriv_pool']
            
        layer_dics.append( layer_dic )

    # write list of dictionaries to file in 'pretty' format
    np.set_printoptions(threshold=1000000000)
    with open(filename_py, 'w') as fh:
        fh.write( pprint.pformat(layer_dics, depth=None) )
        

def load_layers(filename_py):
    """
    load layers that were save by `save_layers`

    Example:
    
       >>> layers = load_layers("/tmp/layers.pydata")
       >>> nn = NeuralNetwork(layers)

    """
    from numpy import array, float64
    
    with open(filename_py, "r") as fh:
        layer_dics = eval( " ".join(fh.readlines()) )
    assert type(layer_dics) == type([])  #
    # interprete dictionaries as subclasses of type `Layer`
    layers = [_layer_from_dictionary(dic) for dic in layer_dics]

    return layers
    
    
def _layer_from_dictionary(dic):
    """
    convert a dictionary to a subclass of 'Layer'
    """
    layer = Layer(dic['nr_param'], dic['dim_in'], dic['dim_out'])
    layer.__dict__ = dic
    # cast layer to appropriate type
    name2class = {
        "FullyConnectedLayer": FullyConnectedLayer,
        "Convolutional2dLayer": Convolutional2dLayer,
        "Pooling2dLayer": Pooling2dLayer,
        "NonLinearityLayer": NonLinearityLayer
    }
    layer.__class__ = name2class[dic['layer_type']]

    # assign activation function
    if hasattr(layer, "activation"):
        layer.f, layer.dfdz = ActivationFunctions.getFunc(layer.activation)
    # assign pooling function
    if hasattr(layer, "pooling"):
        layer.pool, layer.deriv_pool = PoolingFunctions.getFunc(layer.pooling)

    return layer

    
    
##### TESTING #########
def numerical_gradient(f,x0,h=1.0e-5):
    """
    compute gradient of f at x0 by numerical differentiation

    Parameters:
    ===========
    f: scalar function of a 1D vector x
    x0: 1D numpy array

    Returns:
    ========
    1D numpy array with difference quotient 
       df/dx_i = (f(x0 + h*ei) - f(x0))/h 
    """
    n = len(x0)
    f0 = f(x0)
    dfdx = np.zeros(n)
    for i in range(0, n):
        print "numerical gradient: %d of %d" % (i,n)
        ei = np.zeros(n)
        ei[i] = 1.0 # unit vector
#        # forward gradient
#        x_hi = x0 + h*ei
#        dfdx[i] = (f(x_hi) - f0)/h
        # symmetric gradient
        x_mhi = x0 - h*ei
        x_phi = x0 + h*ei
        dfdx[i] = (f(x_phi) - f(x_mhi))/(2.0*h)
    return dfdx

def test_cost_function_gradients(trainer):
    from scipy import optimize

    p0, J = trainer.objective_function()

    def func(p):
        return J(p)[0]

    def grad(p):
        return J(p)[1]

    grad_ana = grad(p0)
    grad_num = numerical_gradient(func, p0)

    print "numerical gradient of cost function"
    print grad_num
    print "analytical gradient of cost function"
    print grad_ana
    print "difference"
    print grad_num - grad_ana
    
    err = optimize.check_grad(func, grad, p0)
    print "|grad (numerical) - grad(analytical)|= %e" % err

def test_fc_layer():
    nn = NeuralNetwork([
        FullyConnectedLayer(dim_in=4, dim_out=3),
        FullyConnectedLayer(dim_in=3, dim_out=2)
        ])
    trainer = Trainer(nn)

    (x1,y1) = np.array([0.0, 1.0, 2.0, 3.0]), np.array([10.0, 20.0])
    (x2,y2) = np.array([0.0, -1.0, 2.0, -3.0]), np.array([4.0, -3.0])

    training_set = [(x1,y1), (x2,y2)]
    trainer.setTrainingSet(training_set)
    test_cost_function_gradients(trainer)

def test_conv_layer():
    nn = NeuralNetwork([
        Convolutional2dLayer(ni=10,nj=10, nr_channels=3,   mi=3,mj=3, nr_features=5),
        NonLinearityLayer(dim=10*10*5),
        Convolutional2dLayer(ni=10,nj=10, nr_channels=5,   mi=3,mj=3, nr_features=2),
        Pooling2dLayer(ni=10,nj=10, nr_channels=2,   ti=2, tj=2, pooling='avg'),
        FullyConnectedLayer(dim_in=5*5*2, dim_out=4, activation='relu'),
        ])
    trainer = Trainer(nn)
    x1 = np.random.rand(10,10,3).flatten()
    #y1 = np.random.rand(10,10,1).flatten()
    y1 = np.random.rand(4)

    training_set = [(x1,y1)]
    trainer.setTrainingSet(training_set)
    test_cost_function_gradients(trainer)

    # save 
    save_layers(nn.getLayers(), "/tmp/neural_network_layers.pydata")
    nn = NeuralNetwork( load_layers("/tmp/neural_network_layers.pydata") )

    
def test_weight_gradient_corr1d():
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    w0 = np.array([-1,0,1])

    n = len(x)
    m = len(w0)
    
    def f(w):
        y = np.zeros(n)
        for i in range(0, n):
            for j in range(0, m):
                if 0 <= i-j < n:
                    y[i] += w[j] * x[i-j]
        J = 0.5 * np.sum(y**2)

        dJdw = np.zeros(m)
        for j in range(0, m):
            for k in range(0, n):
                if 0 <= j+k < n:
                    dJdw[j] += x[k] * y[j+k]
                    
        return J, dJdw

    def func(w):
        return f(w)[0]

    def grad(w):
        return f(w)[1]

    grad_ana = grad(w0)
    grad_num = numerical_gradient(func, w0)

    print "numerical gradient of cost function"
    print grad_num
    print "analytical gradient of cost function"
    print grad_ana
    print "difference"
    print grad_num - grad_ana

def test_weight_gradient_corr2d():
    x = np.array([[1,2,3,4,5,6,7,8,9],
                  [11,22,32,4,5,60,71,-8,-9],
                  [21,-2,3,41,5,6,7,8,9],
                  [1,12,-33,4,5,63,72,-8,-9]])
    """
    w0 = np.array([[-1,0,1,1,2],
                   [0, 5,0,1,-1],
                   [ 1,0,-1,-1,-2]])
    """
    w0 = np.array([[-1,0],
                   [0, 5]])

    ni,nj = x.shape
    mi,mj = w0.shape

    # center filter on each pixel
    if mi % 2 == 1:
        p0 = mi/2
    else:
        p0 = mi/2-1
    if mj % 2 == 1:
        q0 = mj/2
    else:
        q0 = mj/2-1
        
    def f(w):
        y = np.zeros((ni,nj))
        for i in range(0, ni):
            for j in range(0, nj):
                for p in range(0, mi):
                    for q in range(0, mj):
                        if (0 <= i-p+p0 < ni) and (0 <= j-q+q0 < nj):
                            y[i,j] += w[p,q] * x[i-p+p0,j-q+q0]
        # use scipy instead of python loops
        y = signal.convolve2d(x, w, boundary='fill', mode='same')

        J = 0.5 * np.sum(y**2)

        dJdw = np.zeros((mi,mj))
        for p in range(0, mi):
            for q in range(0, mj):
                for i in range(0, ni):
                    for j in range(0, nj):
                        if (0 <= p-p0+i < ni) and (0 <= q-q0+j < nj):
                            dJdw[p,q] += x[i,j] * y[p-p0+i,q-q0+j]

        print "dJdw"
        print dJdw
        # use scipy instead of python loops
        if ni % 2 == 1:
            i0 = ni/2
        else:
            i0 = ni/2-1
        if nj % 2 == 1:
            j0 = nj/2
        else:
            j0 = nj/2-1

        print i0-p0, i0-p0+mi
        print j0-q0, j0-q0+mj
        dJdw = signal.correlate2d(y, x, boundary='fill', mode='same')
        # find window belonging to the non-zero parameters w[p,q]
        dJdw = dJdw[i0-p0:i0-p0+mi, j0-q0:j0-q0+mj]
        print "dJdw (correlate2d)"
        print dJdw
        
        return J, dJdw

    def func(w):
        w = np.reshape(w, (mi,mj))
        return f(w)[0]

    def grad(w):
        w = np.reshape(w, (mi,mj))
        J, dJdw = f(w)
        return dJdw.flatten()

    w0fl = w0.flatten()
    grad_ana = grad(w0fl)
    grad_num = numerical_gradient(func, w0fl)

    print "numerical gradient of cost function"
    print grad_num
    print "analytical gradient of cost function"
    print grad_ana
    print "difference"
    print grad_num - grad_ana

    
if __name__ == "__main__":
    test_fc_layer()
    test_conv_layer()
    #test_weight_gradient_corr1d()
    #test_weight_gradient_corr2d()
