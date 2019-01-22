from Dataset_Generator import datasets
from Utils import *
import random
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score
from PPCA import PPCA

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
cifar10_dir = os.path.join(ROOT_DIR, 'data/CIFAR10/')  # Directory of the cifar dataset
mnist_dir = os.path.join(ROOT_DIR, 'data/MNIST/')
toba_dir = os.path.join(ROOT_DIR, 'data/Toba')

# for CIFAR10
num_pics_CIFAR10 = 10000  # number of pictures to use for train

# for MNIST
num_pics_MNIST = 10000

# for the toy datasets
num_points = 10000  # number of data points
N = 50  # data dimensionality MUST BE ALWAYS SMALLER THAN LATENT
max_iterations = 2  # number of maximum iterations

cifar = False
mnist = True
multivariate = False


def compute_scores(X, n_features):
    # Imported from sklearn to check the scores with different ways. Currently is not used
    n_components = np.arange(0, n_features, 5)  # options for n_components
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X, cv=5)))
        fa_scores.append(np.mean(cross_val_score(fa, X, cv=5)))
        print("Number of component", n)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    return [n_components_fa, n_components_pca]


def calculate_for_Cifar(num_pics_to_load):
    plt.ion()

    X_train, y_train, X_test, y_test = datasets().load_CIFAR10(cifar10_dir)
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_train = X_train[:num_pics_to_load, :]  # take only the first num_pics pictures

    print(X_train.shape)

    ppca = PPCA(max_iterations=max_iterations)

    print("=======>Training Phase<=======")
    fitted_data = ppca.fit(X_train)
    reduced_data = ppca.transform_data(fitted_data)
    created_data = ppca.inverse_transform(reduced_data)
    error_train = get_relative_error(X_train, created_data, num_pics_to_load)
    print("The training avg error of the dataset is: {0}".format(np.mean(error_train)))

    print("=======>Testing Phase<=======")
    reduced_data = ppca.transform_data(X_test)
    created_data = ppca.inverse_transform(reduced_data)
    error_test = get_relative_error(X_test, created_data, num_pics_to_load)
    print("The testing avg error of the dataset is: {0}".format(np.mean(error_test)))

    plt.figure()
    plt.xlabel('Error(%)')
    plt.ylabel('Count')
    plt.title('Error of Reconstructing CIFAR Test Set with PPCA(' + str(ppca.num_components) + " components)")
    plt.hist(list(error_test), bins=100, color="#3F5D7D")  # fancy color
    plt.show()


def calculate_for_Mnist(num_pics_to_load):
    X_train, y_train, X_test, y_test = datasets().load_MNIST(mnist_dir)

    new_X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    new_X_train = new_X_train[:num_pics_to_load, :]  # take only the first num_pics pictures

    ppca = PPCA(max_iterations=max_iterations, num_components=200)

    print("=======>Training Phase<=======")
    fitted_data = ppca.fit(new_X_train)
    reduced_data = ppca.transform_data(fitted_data)
    created_data = ppca.inverse_transform(reduced_data)
    error_train = get_relative_error(new_X_train, created_data, num_pics_to_load)
    print("The avg error of the dataset is: {0}".format(np.mean(error_train)))

    print(created_data.shape)
    created_data = np.reshape(created_data, (10000, 784))
    plt.figure()
    plt.xlabel('Error(%)')
    plt.ylabel('Count of examples')
    plt.title('Error of Reconstructing MNIST with PPCA(' + str(ppca.num_components) + " components)")
    plt.hist(list(error_train), bins=100, color="#3F5D7D")  # fancy color
    plt.xlim([0, 100])

    plt.show()

    # visualize a sample of reconstructed data images
    created_data = np.reshape(created_data, (created_data.shape[0], 28, 28))

    # randomly select 5 images from 1 to num_pics_to_load
    rand_Images_idx = random.sample(range(num_pics_to_load), 3)
    for i, idx in enumerate(rand_Images_idx):
        plt.figure()
        plt.imshow(created_data[i].astype('uint8'))
        plt.xlabel("Actual "
                   "Number {}".format(y_train[i]))
        plt.title("Reconstructed image")

        plt.show()

    for i, idx in enumerate(rand_Images_idx):
        plt.figure()
        plt.imshow(X_train[i].astype('uint8'))
        plt.xlabel("Actual Number {}".format(y_train[i]))
        plt.title("Original Picture")
        plt.show()

    print("=======>Testing Phase<=======")
    reduced_data = ppca.transform_data(X_test)
    created_data = ppca.inverse_transform(reduced_data)
    error_test = get_relative_error(X_test, created_data, num_pics_to_load)
    print("The testing avg error of the dataset is: {0}".format(np.mean(error_test)))

    plt.figure()
    plt.xlabel('Error(%)')
    plt.ylabel('Count')
    plt.title('Error of Reconstructing MNIST Test Set with PPCA(' + str(ppca.num_components) + " components)")
    plt.hist(list(error_test), bins=100, color="#3F5D7D")  # fancy color
    plt.show()

    # visualize a sample of reconstructed data images
    created_data = np.reshape(created_data, (created_data.shape[0], 28, 28))

    # randomly select 5 images from 1 to num_pics_to_load
    rand_Images_idx = random.sample(range(num_pics_to_load), 5)
    for i, idx in enumerate(rand_Images_idx):
        plt.figure()
        plt.imshow(created_data[i].astype('uint8'))
        plt.xlabel("Actual Number {}".format(y_train[i]))
        plt.show()


def calculate_for_Multivariate():
    plt.ion()

    X_train, X_test = datasets().build_A_toy_dataset(N=N, num_points=num_points)
    # print(X_train.shape)

    ppca = PPCA(max_iterations=max_iterations)

    print("=======>Training Phase<=======")
    fitted_data = ppca.fit(X_train)
    reduced_data = ppca.transform_data(fitted_data)
    created_data = ppca.inverse_transform(reduced_data)

    percentage = int(num_points * 0.8)  # get the integer number of points of 0.8% of the whole dataset

    r = range(0, percentage)
    error = get_relative_error(X_train, created_data, percentage + 1)  # +1 cuz it is fucking annoying :)
    # print(percentage, error.shape)
    print("The training avg error of the dataset is: {0}".format(np.mean(error)))
    plt.bar(r, error, width=1, color="blue")
    plt.xlabel('Data Points')
    plt.ylabel('Error')
    plt.title('Error of Reconstructing Training Set 1 with PPCA(' + str(ppca.num_components) + " components)")
    plt.show()

    print("=======>Testing Phase<=======")
    reduced_data = ppca.transform_data(X_test)
    created_data = ppca.inverse_transform(reduced_data)

    percentage = int(num_points * 0.2)  # get the integer number of points of the rest 0.2% of the whole dataset

    error = get_relative_error(X_test, created_data, percentage + 1)  # + 1 cuz it is fucking annoying :)
    r = range(0, percentage)
    print("The testing avg error of the dataset is: {0}".format(np.mean(error)))

    plt.bar(r, error, width=1, color="blue")
    plt.xlabel('Data Points')
    plt.ylabel('Error(%)')
    plt.title('Relative Error of Reconstructing Test Set 1 with PPCA(' + str(ppca.num_components) + " components)")
    plt.show()


def calculate_for_Mnist_PCA(num_pics_to_load):
    X_train, y_train, X_test, y_test = datasets().load_MNIST(mnist_dir)

    new_X_train = np.reshape(X_train, (X_train.shape[0], -1))
    new_X_train = new_X_train[:num_pics_to_load, :]  # take only the first num_pics pictures

    pca = PCA(n_components=200)

    print("=======>Training Phase<=======")

    pca.fit(new_X_train)

    data_reduced = np.dot(new_X_train, pca.components_.T)  # transform
    created_data = np.dot(data_reduced, pca.components_)

    error_train = get_relative_error(new_X_train, created_data, num_pics_to_load)
    print("The avg error of the dataset is: {0}".format(np.mean(error_train)))
    plt.figure()
    plt.xlabel('Error(%)')
    plt.ylabel('Count of examples')
    plt.title('Error of Reconstructing MNIST with PCA (' + str(200) + " components)")
    plt.hist(list(error_train), bins=100, color="#3F5D7D")  # fancy color
    plt.xlim([0, 100])
    plt.show()

    # visualize a sample of reconstructed data images
    created_data = np.reshape(created_data, (created_data.shape[0], 28, 28))

    # randomly select 5 images from 1 to num_pics_to_load
    rand_Images_idx = random.sample(range(1000), 2)

    for i, idx in enumerate(rand_Images_idx):
        plt.figure()
        plt.imshow(created_data[i].astype('uint8'))
        plt.xlabel("Actual Number {}".format(y_train[i]))
        plt.title("Reconstructed image")
        plt.show()

    for i, idx in enumerate(rand_Images_idx):
        plt.figure()
        plt.imshow(X_train[i].astype('uint8'))
        plt.xlabel("Actual Number {}".format(y_train[i]))
        plt.title("Original Picture")
        plt.show()


if __name__ == '__main__':

    if cifar is True:
        # Do PPCA on CIFAR10 data set
        calculate_for_Cifar(num_pics_to_load=num_pics_CIFAR10)
    if mnist is True:
        # Do PPCA on Mnist data set
        calculate_for_Mnist(num_pics_to_load=num_pics_MNIST)
        # Do PCA on Mnist data set
        calculate_for_Mnist_PCA(num_pics_to_load=num_pics_MNIST)
    if multivariate is True:
        # Do PPCA on multivariate gaussian set
        calculate_for_Multivariate()
