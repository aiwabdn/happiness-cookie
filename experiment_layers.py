import numpy as np
import os

from pygln import GLN, utils


backend = 'pytorch'
num_runs = 5
num_epochs = 1
batch_size = 10
eval_batch_size = 100

depths = (0, 5)
sizes = (2, 10)


train_images, train_labels, test_images, test_labels = utils.get_mnist()


os.makedirs('results', exist_ok=True)
with open(f'results/{backend}_layers.csv', 'w') as file:
    file.write(f'{backend},' + ','.join(map(str, range(*sizes))) + '\n')

    # Depths
    for depth in range(*depths):
        accuracy_mean = list()
        accuracy_min = list()
        accuracy_max = list()

        # Sizes
        for size in range(*sizes):

            # Layer config
            layer_sizes = [int(2 ** size) for _ in range(depth)] + [1]

            # Multiple runs
            accuracies = list()
            for run in range(num_runs):

                # Model
                model = GLN(
                    backend=backend, layer_sizes=layer_sizes, input_size=train_images.shape[1],
                    context_map_size=4, num_classes=10, base_predictor=None, learning_rate=1e-4,
                    pred_clipping=1e-3, weight_clipping=5.0, bias=True, context_bias=True
                )

                # Training
                for n in range((num_epochs * train_images.shape[0]) // batch_size):
                    indices = np.arange(n * batch_size, (n + 1) * batch_size)
                    indices = indices % train_images.shape[0]
                    model.predict(train_images[indices], train_labels[indices])

                # Evaluation
                num_correct = 0
                for n in range(test_images.shape[0] // eval_batch_size):
                    indices = np.arange(n * eval_batch_size, (n + 1) * eval_batch_size)
                    prediction = model.predict(test_images[indices])
                    num_correct += np.count_nonzero(prediction == test_labels[indices])

                # Accuracy
                accuracy = num_correct / test_images.shape[0]
                accuracies.append(accuracy)
                print(layer_sizes, run, accuracy)

            # Compute mean/min/max accuracy over runs
            accuracy_mean.append(sum(accuracies) / len(accuracies))
            accuracy_min.append(min(accuracies))
            accuracy_max.append(max(accuracies))

            if depth == 0:
                break

        # Record data per depth
        file.write(f'{depth}-mean,' + ','.join(map(str, accuracy_mean)) + '\n'
                   f'{depth}-min,' + ','.join(map(str, accuracy_min)) + '\n'
                   f'{depth}-max,' + ','.join(map(str, accuracy_max)) + '\n')
