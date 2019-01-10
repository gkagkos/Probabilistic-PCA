from Toy_Dataset_Generator import toy_dataset
from Utils import *
from PPCA import PPCA
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
cifar10_dir = os.path.join(ROOT_DIR, 'data/CIFAR10/')  # Directory of the cifar dataset
mnist_dir = os.path.join(ROOT_DIR, 'data/MNIST/')

# for CIFAR10
num_pics_CIFAR10 = 10000  # number of pictures to use for train

# for MNIST
num_pics_MNIST = 10000

# for the toy datasets
num_points = 10000  # number of data points
N = 50  # data dimensionality MUST BE ALWAYS SMALLER THAN LATENT

latent = 10  # latent dimensionality
max_iterations = 2  # number of maximum iterations

cifar = False
mnist = True
multivariate = False


def calculate_for_Cifar(num_pics_to_load):
    plt.ion()

    X_train, y_train, X_test, y_test = toy_dataset()._load_CIFAR10(cifar10_dir)

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_train = X_train[:num_pics_to_load, :]  # take only the first num_pics pictures

    print(X_train.shape)
    ppca = PPCA(latent_dimensions=latent, max_iterations=max_iterations)

    print("=======>Training Phase<=======")
    fitted_data = ppca._fit(X_train)
    reduced_data = ppca._transform_data(fitted_data)
    created_data = ppca._inverse_transform(reduced_data)
    error_train = get_relative_error(X_train, created_data, num_pics_to_load)

    plt.figure()
    plt.xlabel('Error(%)')
    plt.ylabel('Count')
    plt.title('Error of Reconstructing CIFAR Train Set with PPCA(' + str(ppca.Latent) + " components)")
    plt.hist(list(error_train), bins=100, color="#3F5D7D")  # fancy color
    plt.show()

    # classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_classes = len(classes)
    # samples_per_class = 10
    # created_data = np.reshape(created_data, (created_data.shape[0], 32, 32, 3))
    #
    # # visualize reconstructed data
    # # randomly select 70 images from 1 to num_pics
    # indexes = random.sample(range(num_pics_to_load), 70)
    # indexes = np.reshape(indexes, (10, 7))
    #
    # for j in range(num_classes):
    #     for i, idx in enumerate(indexes[j]):
    #         plt_idx = i * num_classes + j + 1
    #         plt.subplot(samples_per_class, num_classes, plt_idx)
    #         plt.imshow(created_data[idx].astype('uint8'))
    #         plt.axis('off')
    #     plt.suptitle("PPCA Reconstructed CIFAR-10 with " + str(ppca.Latent) + " components")
    #     plt.show()

    print("=======>Testing Phase<=======")
    reduced_data = ppca._transform_data(X_test)
    created_data = ppca._inverse_transform(reduced_data)
    error_test = get_relative_error(X_test, created_data, num_pics_to_load)

    plt.figure()
    plt.xlabel('Error(%)')
    plt.ylabel('Count')
    plt.title('Error of Reconstructing CIFAR Test Set with PPCA(' + str(ppca.Latent) + " components)")
    plt.hist(list(error_test), bins=100, color="#3F5D7D")  # fancy color
    plt.show()


def calculate_for_Mnist(num_pics_to_load):
    X_train, y_train, X_test, y_test = toy_dataset()._load_MNIST(mnist_dir)

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_train = X_train[:num_pics_to_load, :]  # take only the first num_pics pictures

    ppca = PPCA(latent_dimensions=latent, max_iterations=max_iterations)
    #
    # print("=======>Training Phase<=======")
    # fitted_data = ppca._fit(X_train)
    # reduced_data = ppca._transform_data(fitted_data)
    # created_data = ppca._inverse_transform(reduced_data)
    # error_train = get_relative_error(X_train, created_data, num_pics_to_load)

    fitted_data_2 = PCA(n_components='mle',svd_solver='full')
    fitted_data_2.fit(X_train)
    components = fitted_data_2.n_components_

    # reduced_data_2 = PCA.transform(fitted_data_2)
    # created_data_2 = PCA.inverse_transform(reduced_data_2)
    # error_train_2 = get_relative_error(X_train, created_data_2, num_pics_to_load)




    plt.figure()
    plt.xlabel('Error(%)')
    plt.ylabel('Count')
    plt.title('Error of Reconstructing MNIST Train Set with PPCA(' + str(ppca.Latent) + " components)")
    plt.hist(list(error_train_2), bins=100, color="#3F5D7D")  # fancy color
    plt.show()

    # # visualize a sample of reconstructed data images
    # created_data = np.reshape(created_data, (created_data.shape[0], 28, 28))
    #
    # # randomly select 5 images from 1 to num_pics_to_load
    # rand_Images_idx = random.sample(range(num_pics_to_load), 5)
    # for i,idx in enumerate(rand_Images_idx):
    #     plt.figure()
    #     plt.imshow(created_data[idx].astype('uint8'))
    #     plt.xlabel("Actual Number {}".format(y_train[idx]))
    #     plt.show()
    #

    print("=======>Testing Phase<=======")
    reduced_data = ppca._transform_data(X_test)
    created_data = ppca._inverse_transform(reduced_data)
    error_test = get_relative_error(X_test, created_data, num_pics_to_load)

    plt.figure()
    plt.xlabel('Error(%)')
    plt.ylabel('Count')
    plt.title('Error of Reconstructing MNIST Test Set with PPCA(' + str(ppca.Latent) + " components)")
    plt.hist(list(error_test), bins=100, color="#3F5D7D")  # fancy color
    plt.show()
    # visualize a sample of reconstructed data images
    created_data = np.reshape(created_data, (created_data.shape[0], 28, 28))

    # randomly select 5 images from 1 to num_pics_to_load
    rand_Images_idx = random.sample(range(num_pics_to_load), 5)
    for i, idx in enumerate(rand_Images_idx):
        plt.figure()
        plt.imshow(created_data[idx].astype('uint8'))
        plt.xlabel("Actual Number {}".format(y_train[idx]))
        plt.show()


def calculate_for_Multivariate():
    plt.ion()

    X_train, X_test = toy_dataset()._build_A_toy_dataset(N=N, num_points=num_points)
    print(X_train.shape)
    ppca = PPCA(latent_dimensions=latent, max_iterations=max_iterations)

    print("=======>Training Phase<=======")
    fitted_data = ppca._fit(X_train)
    reduced_data = ppca._transform_data(fitted_data)
    created_data = ppca._inverse_transform(reduced_data)

    percentage = (int)(num_points * 0.8)  # get the integer number of points of 0.8% of the whole dataset

    r = range(0, percentage)
    error = get_relative_error(X_train, created_data, percentage + 1)  # +1 cuz it is fucking annoying :)
    # print(percentage, error.shape)

    plt.bar(r, error, width=1, color="blue")
    plt.xlabel('Data Points')
    plt.ylabel('Error')
    plt.title('Error of Reconstructing Training Set 1 with PPCA(' + str(ppca.Latent) + " components)")
    plt.show()

    print("=======>Testing Phase<=======")
    reduced_data = ppca._transform_data(X_test)
    created_data = ppca._inverse_transform(reduced_data)

    percentage = (int)(num_points * 0.2)  # get the integer number of points of the rest 0.2% of the whole dataset

    error = get_relative_error(X_test, created_data, percentage + 1)  # + 1 cuz it is fucking annoying :)
    r = range(0, percentage)

    plt.bar(r, error, width=1, color="blue")
    plt.xlabel('Data Points')
    plt.ylabel('Error(%)')
    plt.title('Relative Error of Reconstructing Test Set 1 with PPCA(' + str(ppca.Latent) + " components)")
    plt.show()


if __name__ == '__main__':
    if cifar is True:
        # Do PPCA on CIFAR10 data set
        calculate_for_Cifar(num_pics_to_load=num_pics_CIFAR10)
    if mnist is True:
        # Do PPCA on Mnist data set
        calculate_for_Mnist(num_pics_to_load=num_pics_MNIST)
    if multivariate is True:
        # Do PPCA on multivariate gaussian set
        calculate_for_Multivariate()

