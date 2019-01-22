from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np


class Kernel_PCA(object):

    def __init__(self, kernel=None, gamma=None, coef=1, degree=3):
        self.kernel = kernel
        self.gamma = gamma
        self.coef = coef
        self.degree = degree

    def transform_data(self, data):
        if self.kernel == "rbf":
            data = self.rbf_kernel(data, self.gamma)
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
    pass
