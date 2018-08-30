"""
 read MNIST database of labelled handwritten digits

The MNIST database can be obtained from
 http://yann.lecun.com/exdb/mnist/
"""
import numpy as np
import struct
import os.path

# Directory where training data and neural network are stored,
# relative to this file the path is '../DATA'
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../DATA")


def read_idx_images(filename):
    """
    read training set from an idx-file and return an interator to the arrays of handwritten digits

    Parameters
    ----------
    filename: path to MNIST training set in idx format

    Returns
    -------
    iterator: list of 2D arrays
    """
    fh = open(filename, "rb")
    # first 2 bytes are always zero
    byte1 = ord(fh.read(1))
    byte2 = ord(fh.read(1))
    assert byte1 == 0 and byte2 == 0
    # third bye codes the type of data
    byte3 = ord(fh.read(1))
    assert byte3 == 0x08  # unsigned byte
    # 4-th byte codes the number of dimensions
    dim = ord(fh.read(1))
    #print "dimension of data: %d" % dim
    assert dim == 3
    shape = []
    for d in range(0, dim):
        bytes = fh.read(4)
        n = struct.unpack('>i', bytes)[0]
        shape.append(n)
    #print "shape of date: %s" % shape
    data = np.fromfile(fh, dtype=np.uint8)
    images = np.reshape(data, shape)

    nimg = shape[0]
    for i in range(0, nimg):
        # scale image date to the range [0,1]
        yield images[i,:,:]/255.0
        
def read_idx_labels(filename):
    """
    read label assignments for handwritten digits and convert them to a unit vectors

    Parameters
    ----------
    filename: path to MNIST idx file with label assignments

    Returns
    -------
    iterator: list of unit vectors y encoding the labels
              if y[i]=1, the label is i
    """
    fh = open(filename, "rb")
    # first 2 bytes are always zero
    byte1 = ord(fh.read(1))
    byte2 = ord(fh.read(1))
    assert byte1 == 0 and byte2 == 0
    # third bye codes the type of data
    byte3 = ord(fh.read(1))
    assert byte3 == 0x08  # unsigned byte
    # 4-th byte codes the number of dimensions
    dim = ord(fh.read(1))
    #print "dimension of data: %d" % dim
    assert dim == 1
    shape = []
    for d in range(0, dim):
        bytes = fh.read(4)
        n = struct.unpack('>i', bytes)[0]
        shape.append(n)
    #print "shape of date: %s" % shape
    data = np.fromfile(fh, dtype=np.uint8)
    labels = np.reshape(data, shape)

    # The labels are integers, where each number encodes a class. The output of
    # the neural network is a vector of probabilities. The probability for classifying
    # the input x as belonging to class i is y[i] (ranging from 0 to 1).
    # Therefore the labels have to be tranformed into unit vectors for each class.
    # 
    #  1  ->  [0,1,0,0,0,0,0,0,0,0]
    #  9  ->  [0,0,0,0,0,0,0,0,0,1]
    #

    # the length of the vector y equals the number of classes
    nr_classes = labels.max() - labels.min() + 1
    
    nimg = shape[0]
    for i in range(0, nimg):
        # convert label to unit vector
        y = np.zeros(nr_classes, dtype=int).tolist()
        y[labels[i]] = 1
        yield y

def save_idx_images(images, filename):
    """
    save images to an idx file

    Parameters
    ----------
    images   :  3d array with shape (nr.images, nx, ny)
                pixel brightness should be in the range [0,1]
    filename :  path to the binary output file
    """
    fh = open(filename, "wb")
    # first 2 bytes are zero
    fh.write(chr(0))
    fh.write(chr(0))
    # data type
    fh.write(chr(8))
    # number of dimensions
    dim = len(images.shape)
    fh.write(chr(dim))
    # shape of data
    for d in range(0, dim):
        fh.write( struct.pack('>i', images.shape[d]) )
    # save data after scaling to the range [0,255]
    data = np.array(images*255, dtype=np.uint8)
    data.tofile(fh)
    fh.close()

def save_idx_labels(labels, filename):
    """
    save label assignments

    Parameters
    ----------
    labels     : list of integers indicating the class of each sample
    filename   : path to output file
    """
    fh = open(filename, "wb")
    # first 2 bytes are zero
    fh.write(chr(0))
    fh.write(chr(0))
    # data type
    fh.write(chr(8))
    # number of dimensions
    dim = len(labels.shape)
    fh.write(chr(dim))
    # shape of data
    for d in range(0, dim):
        fh.write( struct.pack('>i', labels.shape[d]) )

    data = np.array(labels, dtype=np.uint8)
    data.tofile(fh)
    fh.close()

        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os.path
    # read training set (x,y)
    #  x: image with handwritten digit
    #  y: manually assigned digit
    images_it = read_idx_images(os.path.join(data_dir, "train-images-idx3-ubyte"))
    labels_it = read_idx_labels(os.path.join(data_dir, "train-labels-idx1-ubyte"))
    
    for img,label_vec in zip(images_it, labels_it):
        label = label_vec.index(1)
        plt.imshow(img)
        plt.title("%d" % label)
        plt.show()
        
    
