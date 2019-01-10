import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn import datasets
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
mnist_dir = os.path.join(ROOT_DIR, 'data/MNIST/')


class toy_dataset(object):
    def __init__(self):
        pass

    def _build_A_toy_dataset(self, N, num_points):

        """
            Generate random data from a N dimensional multivariate distribution
        """
        self.N = N  # number of dimensions
        self.num_points = num_points  # number of data points for each distribution
        self.stdError = 0.1  # sigma value error

        Mean = np.random.randint(10, size=N)
        # sample uniform random variable
        X = np.hstack((np.random.uniform(0, 1, N), np.random.uniform(0, 1, N)))
        X = X[:, np.newaxis]
        X = np.reshape(X, (N, 2))
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Pr = (1 / np.sqrt(2 * np.pi)) * np.exp((-1 / (2 * self.stdError)) * distance.euclidean(X[i], X[j]))
                uniform = np.random.uniform(0, 1, 1)  # draw one smaple from the uniform variable
                if (Pr >= uniform):  # If the value is greater than the sample
                    A[i, j] = 1  # we put an edje
                else:  # if the value of Pr is less we put a zero
                    A[i, j] = 0

        Precision = np.zeros((N, N))
        # based on the random adjecency matrix we make a random precision matrix with the edges replaced by 0.245
        for i in range(N):
            for j in range(i + 1):
                if (j == i):
                    Precision[i, j] = 1
                else:
                    if (A[i, j] == 1):
                        Precision[i, j] = 0.245
                        Precision[j, i] = 0.245

        Covariance = np.linalg.inv(Precision)  # covariance is the inverse of the precision

        data = np.random.multivariate_normal(Mean, Covariance,
                                             self.num_points)  # generate the data based on mean and covariance
        X_train, X_test, = train_test_split(data, test_size=0.2)

        return X_train, X_test

    def _load_CIFAR10(self, path):
        """

        Load CIFAR10

        link ~ http://www.cs.toronto.edu/~kriz/cifar.html

        Data Description of 5 batch files.

        Each of the batch files contains a dictionary with the following elements:
        data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
        The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
        The image is stored in row-major order, so that the first 32 entries of the array are the red channel values
        of the first row of the image.
        labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith
        image in the array data.

        The dataset contains another file, called batches.meta. It too contains a Python dictionary object. I
        t has the following entries:
        label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above.
        For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.
        
        """

        def load_batch(filename):
            with open(filename, 'rb') as f:
                datadict = pickle.load(f, encoding='latin1')
                X = datadict['data']
                Y = datadict['labels']
                X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(
                    "float")  # make the last column the color channel
                Y = np.array(Y)
                return X, Y

        x_train = []
        y_train = []
        for batch in range(1, 6):
            print("=====>Loading Batch file: data_batch_{}<=====".format(batch))

            batch_filename = os.path.join(path, 'data_batch_{}'.format(batch))
            # print(batch_filename)
            X, Y = load_batch(batch_filename)
            x_train.append(X)
            y_train.append(Y)

        X_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)

        print("-----------------------------------------")
        print("           CIFAR10 is Loaded")
        print("-----------------------------------------")

        X_test, y_test = load_batch(os.path.join(path, 'test_batch'))

        return X_train, y_train, X_test, y_test

    def _load_MNIST(self, path):
        print("Loading MNIST...")
        """
            Load CIFAR10

            link ~ http://yann.lecun.com/exdb/mnist/

            Data Description:
            
            The data is stored in a very simple file format designed for storing vectors and multidimensional matrices.
            General info on this format is given at the end of this page, but you don't need to read that to use the data files.
            All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors.
            Users of Intel processors and other low-endian machines must flip the bytes of the header.

            There are 4 files:

            train-images-idx3-ubyte: training set images
            train-labels-idx1-ubyte: training set labels
            t10k-images-idx3-ubyte:  test set images
            t10k-labels-idx1-ubyte:  test set labels

        """

        def load_filename(prefix, path):
            intType = np.dtype('int32').newbyteorder('>')
            nMetaDataBytes = 4 * intType.itemsize

            data = np.fromfile(path + "/" + prefix + '-images-idx3-ubyte', dtype='ubyte')
            magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
            data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])

            labels = np.fromfile(path + "/" + prefix + '-labels-idx1-ubyte',
                                 dtype='ubyte')[2 * intType.itemsize:]

            return data, labels

        trainingImages, trainingLabels = load_filename("train", path)
        testImages, testLabels = load_filename("t10k", path)
        print("-----------------------------------------")
        print("           MNIST is Loaded")
        print("-----------------------------------------")
        return trainingImages, trainingLabels, testImages, testLabels

# Load MNIST ONLINE. Sometimes gets it gets you http error.
# def _load_mnist():
#     '''
#     Load the digits dataset
#     fetch_mldata ... dataname is on mldata.org, data_home
#     load 10 classes, from 0 to 9
#     '''
#     mnist = datasets.fetch_mldata('MNIST original')
#     n_train = 60000  # The size of training set
#     # Split dataset into training set (60000) and testing set (10000)
#     data_train = mnist.data[:n_train]
#     target_train = mnist.target[:n_train]
#     data_test = mnist.data[n_train:]
#     target_test = mnist.target[n_train:]
#     return (data_train.astype(np.float32), target_train.astype(np.float32),
#             data_test.astype(np.float32), target_test.astype(np.float32))


if __name__ == '__main__':



    trainingImages, trainingLabels, testImages, testLabels = toy_dataset()._load_MNIST(mnist_dir)



