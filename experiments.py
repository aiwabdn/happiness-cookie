# %%
import numpy as np
import pandas as pd
from copy import deepcopy
from pygln.utils import evaluate_mnist
from pygln import GLN


# %%
def identity(x):
    return x


def paper_squash_inputs(x):
    return (x * (1 - 2 * 0.001)) + 0.001


def constant(x):
    return np.full(shape=x.shape, fill_value=0.5)


DEFAULT_PARAMS = {
    'backend': 'numpy',
    'layer_sizes': [2, 1],
    'input_size': 784,
    'context_map_size': 4,
    'classes': range(10),
    'base_predictor': None,
    'learning_rate': 0.01,
    'pred_clipping': 0.001,
    'weight_clipping': 5,
    'bias': True,
    'context_bias': True
}

TESTING_GRID = {
    'layer_sizes': [[2, 2, 1], [4, 4, 1], [8, 8, 1], [16, 16, 1], [32, 32, 1],
                    [64, 64, 1], [128, 128, 1], [1], [2, 1], [4, 2, 1],
                    [8, 4, 2, 1], [16, 8, 4, 2, 1]],
    'context_map_size': [1, 2, 4, 8],
    'base_predictor': [None, paper_squash_inputs, constant],
    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'pred_clipping': [1e-3, 1e-2, 1e-1],
    'weight_clipping': [2, 3, 4, 5, 10],
    'bias': [True, False],
    'context_bias': [True, False]
}

TEST_GRID = {
    'layer_sizes': [[2, 1], [4, 1]],
    'bias': [True, False],
}


def average_n_runs(num_epochs=3, num_runs=3, **kwargs):
    print('Evaluating {}'.format(str(kwargs)))
    run_outputs = []
    for run in range(num_runs):
        model = GLN(**kwargs)
        run_accuracies = evaluate_mnist(
            model,
            max_learning_rate=kwargs['learning_rate'],
            num_epochs=num_epochs)
        run_outputs.append(run_accuracies)
    run_outputs = np.vstack(run_outputs)
    print(run_outputs)
    mean_accuracies = np.mean(run_outputs, axis=0)
    print(mean_accuracies)
    mean_accuracies = dict(
        zip([f'acc_mean_{i}' for i in range(1, num_epochs + 1)],
            mean_accuracies))
    kwargs.update(mean_accuracies)
    std_accuracies = np.std(run_outputs, axis=0)
    std_accuracies = dict(
        zip([f'acc_std_{i}' for i in range(1, num_epochs + 1)],
            std_accuracies))
    kwargs.update(std_accuracies)
    kwargs['layer_sizes'] = str(kwargs['layer_sizes'])
    if kwargs['base_predictor']:
        kwargs['base_predictor'] = kwargs['base_predictor'].__name__
    _ = kwargs.pop('classes')
    return kwargs


def run_test_grid(test_grid, default_grid, epochs_per_run=3, runs_per_test=3):
    run_outputs = []
    for param_name, param_values in test_grid.items():
        for param_value in param_values:
            test_params = deepcopy(default_grid)
            test_params[param_name] = param_value
            run_output = average_n_runs(epochs_per_run, runs_per_test,
                                        **test_params)
            run_outputs.append(run_output)
    return pd.DataFrame(run_outputs)


# %%
run_test_grid(TEST_GRID, DEFAULT_PARAMS)
