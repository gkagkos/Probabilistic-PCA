import os

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

from Toy_Dataset_Generator import toy_dataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the script

mnist_dir = os.path.join(ROOT_DIR, 'data/MNIST/')

class DataTransformation(object):

    def __init__(self, kernel=None, gamma=None, coef=1, degree=3):
        self.kernel = kernel
        self.gamma = gamma
        self.coef = coef
        self.degree = degree

    def transform_data(self, data):
        if self.kernel == "rbf":
            data =self.rbf_kernel(data, self.gamma)
        elif self.kernel == "poly":
            data = self.polynomial_kernel(data, self.degree, self.gamma, self.coef)
        elif self.kernel == "sigmoid":
            data = self.sigmoid_kernel(data, self.gamma, self.coef)
        else:
            raise Exception(" '{0}' is not an available option".format(self.kernel))

        # Centering the symmetric NxN kernel matrix.
        N = data.shape[0]
        one_n = np.ones((N, N)) / N
        data = data - one_n.dot(data) - data.dot(one_n) + one_n.dot(data).dot(one_n)
        return data

    def rbf_kernel_pca(self, X, gamma, n_components):
        """
        Implementation of a RBF kernel PCA.

        Arguments:
            X: A MxN dataset as NumPy array where the samples are stored as rows (M),
               and the attributes defined as columns (N).
            gamma: A free parameter (coefficient) for the RBF kernel.
            n_components: The number of components to be returned.

        """
        # Calculating the squared Euclidean distances for every pair of points
        # in the MxN dimensional dataset.
        sq_dists = pdist(X, 'sqeuclidean')

        # Converting the pairwise distances into a symmetric MxM matrix.
        mat_sq_dists = squareform(sq_dists)

        # Computing the MxM kernel matrix.
        # rbf kernel
        K = exp(-gamma * mat_sq_dists)

        # Centering the symmetric NxN kernel matrix.
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        # Obtaining eigenvalues in descending order with corresponding
        # eigenvectors from the symmetric matrix.
        eigvals, eigvecs = eigh(K)

        # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
        X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

        return X_pc

    def sigmoid_kernel(self, data, gamma=None, coef=1):
        """
            Compute the sigmoid kernel between X and Y::
                K(X, Y) = tanh(gamma <X, Y> + coef0)


            Arguments:
                data : ndarray of shape (n_samples_data, n_features)
                gamma : float, default None
                    If None, defaults to 1.0 / n_features
                coef : float, default 1

            Returns:
                matrix : array of shape (n_samples_data, n_samples_data)


        """

        if gamma is None:
            gamma = 1.0 / data.shape[1]

        data = np.dot(np.array(data), np.array(data).T)
        data *= gamma
        data += coef
        np.tanh(data, data)  # compute tanh in-place

        return data

    def rbf_kernel(self, data, gamma=None):
        """
            Compute the rbf (gaussian) kernel between X and Y::

                K(x, y) = exp(-gamma ||x-y||^2)

            Arguments:
                data : array of shape (n_data_samples, n_features)
                gamma : float, default None
                    If None, defaults to 1.0 / n_features

            Returns:
                matrix : array of shape (n_samples_data, n_samples_data)

        """
        if gamma is None:
            gamma = 1.0 / data.shape[1]

        sq_dists = pdist(data, 'sqeuclidean')

        # Converting the pairwise distances into a symmetric MxM matrix.
        mat_sq_dists = squareform(sq_dists)

        # Computing the MxM kernel matrix.
        # rbf kernel
        data = exp(-gamma * mat_sq_dists)

        return data

    def polynomial_kernel(self, data, degree=3, gamma=None, coef=1):
        """
            Compute the polynomial kernel between X and Y::

                K(X, Y) = (gamma <X, Y> + coef0)^degree

            Arguments:
                data : ndarray of shape (n_samples__data, n_features)
                degree : int, default 3
                gamma : float, default None
                    if None, defaults to 1.0 / n_features
                coef : float, default 1

            Returns:
                matrix : array of shape (n_samples_data, n_samples_data)


        """

        if gamma is None:
            gamma = 1.0 / data.shape[1]

        data = np.dot(np.array(data), np.array(data).T)
        data *= gamma
        data += coef
        data **= degree

        return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn import metrics
    from sklearn.datasets import make_moons
    from PPCA import PPCA
    from sklearn.kernel_approximation import AdditiveChi2Sampler



    # RBF TEST
    # sklearn_rbf = metrics.pairwise.rbf_kernel(X)
    # my_rbf = kpca.transform_data(X)
    #
    # result1 = np.allclose(sklearn_rbf, my_rbf)
    #
    # assert result1 is True, "rbf implementation is not ok"

    # polynomial test
    # sklearn_poly = metrics.pairwise.polynomial_kernel(X)
    # my_poly = kpca.polynomial_kernel(X)
    #
    # result2 = np.allclose(sklearn_poly, my_poly)
    #
    # assert result2 is True, "polynomial implementation is not ok"

    # sigmoid test
    # sklearn_sigmoid = metrics.pairwise.sigmoid_kernel(X)
    # my_sigmoid = kpca.sigmoid_kernel(X)
    #
    # result3 = np.allclose(sklearn_sigmoid, my_sigmoid)
    #
    # assert result3 is True, "sigmoid implementation is not ok"

    # rbf kernel pca example
    # X_pc = kpca.rbf_kernel_pca(X, gamma=15, n_components=2)

    kpca = DataTransformation(kernel="rbf",gamma=4)

    X, y = make_moons(n_samples=100, random_state=123)


    X_pc = kpca.transform_data(X)

    ppca = PPCA(latent_dimensions=30, max_iterations=50)

    ppca._fit(X_pc)
    reduced_data = ppca._transform_data(X_pc)

    X_pc = reduced_data

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pc[y == 0, 0], X_pc[y == 0, 1], color='red', alpha=0.5)
    plt.scatter(X_pc[y == 1, 0], X_pc[y == 1, 1], color='blue', alpha=0.5)

    plt.title('First 2 principal components after RBF Kernel PCA')
    plt.text(-0.18, 0.18, 'gamma = 15', fontsize=12)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
