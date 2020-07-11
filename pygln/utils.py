import numpy as np
import pandas as pd
from scipy.ndimage import interpolation
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


###################################################
# adopted from https://fsix.github.io/mnist/
def moments(image):
    c0, c1 = np.mgrid[:image.shape[0], :
                      image.shape[1]]  # A trick in numPy to create a mesh grid
    totalImage = np.sum(image)  # sum of pixels
    m0 = np.sum(c0 * image) / totalImage  # mu_x
    m1 = np.sum(c1 * image) / totalImage  # mu_y
    m00 = np.sum((c0 - m0)**2 * image) / totalImage  # var(x)
    m11 = np.sum((c1 - m1)**2 * image) / totalImage  # var(y)
    m01 = np.sum((c0 - m0) * (c1 - m1) * image) / totalImage  # covariance(x,y)
    mu_vector = np.array([m0, m1
                          ])  # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array(
        [[m00, m01],
         [m01, m11]])  # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix


def deskew(image):
    c, v = moments(image)
    alpha = v[0, 1] / v[0, 0]
    affine = np.array([[1, 0], [alpha, 1]])
    ocenter = np.array(image.shape) / 2.0
    offset = c - np.dot(affine, ocenter)
    return interpolation.affine_transform(image, affine, offset=offset)


def deskewAll(X):
    currents = []
    for i in range(len(X)):
        currents.append(deskew(X[i].reshape(28, 28)).flatten())
    return np.array(currents)


###################################################


def get_mnist(deskewed=True):
    from torchvision.datasets import MNIST

    trainset = MNIST('./data', train=True, download=True)
    X_train = trainset.data.numpy().reshape(60000, -1).astype(np.float) / 255
    if deskewed:
        X_train = deskewAll(X_train)
    y_train = trainset.targets.numpy()

    testset = MNIST('./data', train=False, download=True)
    X_test = testset.data.numpy().reshape(10000, -1).astype(np.float) / 255
    if deskewed:
        X_test = deskewAll(X_test)
    y_test = testset.targets.numpy()

    return X_train, y_train, X_test, y_test


def shuffle_data(X, y):
    assert X.shape[0] == y.shape[0]
    rng = np.random.default_rng()
    permutation = rng.permutation(X.shape[0])
    return X[permutation, :], y[permutation]


def evaluate_mnist(model,
                   deskewed=True,
                   batch_size=1,
                   num_epochs=1,
                   data_transform=None,
                   result_transform=None):
    from tqdm import tqdm

    # get MNIST data as numpy arrays
    X_train, y_train, X_test, y_test = get_mnist(deskewed)
    # randomly shuffle data
    X_train, y_train = shuffle_data(X_train, y_train)
    X_test, y_test = shuffle_data(X_test, y_test)

    if data_transform:
        X_train = data_transform(X_train)
        y_train = data_transform(y_train)
        X_test = data_transform(X_test)

    num_batches = int(np.ceil(len(X_train) / batch_size))
    for e in range(num_epochs):
        for i in tqdm(range(num_batches)):
            # set learning rate, following paper
            model.set_learning_rate(min(100 / (i + 1), 0.01))

            # get batch
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]

            # run forward with data
            _ = model.predict(X_batch, y_batch)

    # perform inference on test set
    num_batches = int(np.ceil(len(X_test) / batch_size))
    outputs = []
    for i in tqdm(range(num_batches)):
        # get batch
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        X_batch = X_test[batch_start:batch_end]

        # run forward with data
        pred = model.predict(X_batch)
        if result_transform:
            pred = result_transform(pred)
        outputs.append(pred)

    outputs = np.vstack(outputs)

    # define metrics
    classes = np.unique(y_train)
    outputs = outputs.argmax(axis=1).flatten()
    accuracy = 100 * sum(y_test == outputs) / len(y_test)
    conf_mat = pd.DataFrame(confusion_matrix(y_test, outputs),
                            index=classes,
                            columns=classes)
    prfs_mat = pd.DataFrame(precision_recall_fscore_support(y_test, outputs),
                            index=['precision', 'recall', 'fscore', 'support'],
                            columns=classes)

    return accuracy, conf_mat, prfs_mat
