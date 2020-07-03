import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def get_mnist_numpy():
    from torchvision.datasets import MNIST
    from sklearn.preprocessing import label_binarize

    trainset = MNIST('./data', train=True, download=True)
    X_train = trainset.data.numpy().reshape(60000, -1).astype(np.float) / 255
    # y_train = label_binarize(trainset.train_labels.numpy(), classes=range(10))
    y_train = trainset.train_labels.numpy()

    testset = MNIST('./data', train=False, download=True)
    X_test = testset.data.numpy().reshape(10000, -1).astype(np.float) / 255
    # y_test = label_binarize(testset.test_labels.numpy(), classes=range(10))
    y_test = testset.test_labels.numpy()

    return X_train, y_train, X_test, y_test


def shuffle_data(X, y):
    assert X.shape[0] == y.shape[0]
    rng = np.random.default_rng()
    permutation = rng.permutation(X.shape[0])
    return X[permutation, :], y[permutation, :]


def get_mnist_metrics(model,
                      mnist_class=0,
                      batch_size=1,
                      data_transform=None,
                      result_transform=None):
    from tqdm import tqdm

    if not data_transform:
        data_transform = lambda x, y: (x, y)
    if not result_transform:
        result_transform = lambda x: x

    # get MNIST data as numpy arrays
    X_train, y_train, X_test, y_test = get_mnist_numpy()
    # randomly shuffle data
    X_train, y_train = shuffle_data(X_train, y_train)
    X_test, y_test = shuffle_data(X_test, y_test)
    # choose the target class
    y_train = y_train[:, mnist_class]
    y_test = y_test[:, mnist_class]

    X_train, y_train = data_transform(X_train, y_train)
    X_test, _ = data_transform(X_test, y_test)

    num_batches = int(np.ceil(len(X_train) / batch_size))
    for i in tqdm(range(num_batches)):
        # set learning rate
        model.set_learning_rate(min(100 / (i + 1), 0.01))

        # get batch
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        X_batch = X_train[batch_start:batch_end]
        y_batch = y_train[batch_start:batch_end]

        # run forward with data
        _ = model.predict(X_batch.T, X_batch.T, y_batch)

    # perform inference on test set
    num_batches = int(np.ceil(len(X_test) / batch_size))
    outputs = []
    for i in tqdm(range(num_batches)):
        # get batch
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        X_batch = X_test[batch_start:batch_end]

        # run forward with data
        outputs.append(result_transform(model.predict(X_batch.T, X_batch.T)))

    outputs = np.hstack(outputs).flatten()

    # threshold outputs
    outputs = (outputs.flatten() > 0.5).astype(np.int)

    # define metrics
    classes = [f'not_{mnist_class}', f'is_{mnist_class}']
    accuracy = 100 * sum(y_test == outputs) / len(y_test)
    conf_mat = pd.DataFrame(confusion_matrix(y_test, outputs),
                            index=classes,
                            columns=classes)
    prfs_mat = pd.DataFrame(precision_recall_fscore_support(y_test, outputs),
                            index=['precision', 'recall', 'fscore', 'support'],
                            columns=classes)

    return accuracy, conf_mat, prfs_mat
