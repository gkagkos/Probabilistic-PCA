import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Toy_Dataset_Generator import toy_dataset
import Utils


class PPCA(object):
    def __init__(self, latent_dimensions=2, sigma=1, max_iterations=20):

        self.Latent = latent_dimensions  # hidden dimensions
        self.sigma = sigma  # std of the noise
        self.max_iterations = max_iterations  # maximum iterations to do
        self.mean = None
        self.W = None  # W = projection matrix DxL

    def _fit(self, data):
        self.data = data  # our original data
        self.mean = np.mean(self.data, axis=0)  # mean of the model

        self.Num_points = data.shape[0]  # number of data points
        self.Dim = data.shape[1]  # number of dimensions of the data

        self.EM()

        return data

    def EM(self):

        """Perform Expectation Maximazation Algorithm"""
        [dim, latent, mean, sigma, data] = [self.Dim, self.Latent, self.mean, self.sigma, self.data]

        W = np.random.rand(dim, latent)

        for i in range(self.max_iterations):
            print("Perform Expectation Maximazation Step Iteration={}".format(i+1))
            # Expectation Step
            M = np.linalg.inv(W.T.dot(W) + sigma * np.identity(latent))
            Xn = M.dot(np.transpose(W)).dot((data - mean).T)

            XnXn = sigma * M + Xn.dot(np.transpose(Xn))

            # Maximazation Step
            W_avg = (np.transpose(data - mean).dot(np.transpose(Xn))).dot(np.linalg.inv(XnXn))

            sigmaNew = (1 / (self.Num_points * self.Dim)) * \
                       (np.linalg.norm(data - mean) -
                        2 * np.trace(np.transpose(Xn).dot(np.transpose(W_avg)).dot((data - mean).T))) + \
                       np.trace(XnXn.dot(np.transpose(W_avg).dot(W_avg)))

            sigmaNew = np.absolute(sigmaNew)

            W = W_avg
            sigma = sigmaNew

        self.W = W
        self.sigma = sigma

    # def ML(self):
    #
    #     """Perform Maximum Likelihood Algorithm"""
    #
    #     [data, mean, latent, sigma, N] = [self.data, self.mean, self.Latent, self.sigma, self.Num_points]
    #     print()
    #     [u, s, v] = np.linalg.svd(data - mean)
    #
    #
    #
    #     sigma = 1.0 / (N - latent) * np.sum(s[latent:] ** 2)
    #
    #
    #
    #     self.W = w
    #     print(w.shape)
    #     self.sigma = sigma


    def _transform_data(self, data):

        """ Transform the data to the latent subspace """
        print("Transform the data to the latent subspace")
        [W, sigma, mean] = [self.W, self.sigma, self.mean]

        M = np.transpose(W).dot(W) + sigma * np.eye(self.Latent)  # M = W.T*W + sigma^2*I
        Minv = np.linalg.inv(M)  # LxL

        latent_data = Minv.dot(np.transpose(W)).dot(np.transpose(data - mean))
        latent_data = np.transpose(latent_data)  # NxL
        return latent_data

    def _inverse_transform(self, data):

        """ Transform the reduced data to the original size """

        print("Reconstruct the Dataset to the original size")
        # calculate the tuned M
        M = np.transpose(self.W).dot(self.W) + self.sigma * np.eye(self.Latent)

        # create a simulation of the old data after beeing transformed with PCA
        created_data = self.W.dot(np.linalg.inv((self.W.T).dot(self.W))).dot(M).dot(data.T).T + self.mean

        return created_data


if __name__ == "__main__":
    pass
    # num_points = 5000  # number of data points
    # N = 50  # data dimensionality
    # K = 5  # latent dimensionality
    #
    # toy = toy_dataset(num_points=num_points, N=N)
    # data = toy._build_toy_dataset()
    # data_train, data_test = train_test_split(data, test_size=0.2)
    #
    # ppca = PPCA()
    #
    # fitted_data = ppca._fit(data_train,ML=True)
    #
    # reduced_data = ppca._transform_data(fitted_data)
    # created_data = ppca._inverse_transform(reduced_data)
    #
    # trained_num_points = (int)(num_points * 0.8)
    # error = Utils.get_relative_error(data_train, created_data, num_points=trained_num_points)
    #
    #