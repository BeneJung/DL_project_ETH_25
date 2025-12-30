import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# Run this file to generate the gaussians
# =============================================================================

dataset_size = 10000
dataset_folder = "./data/two_gaussians"
dataset_name = "dataset.pt"
dataset_path = dataset_folder + "/" + dataset_name

class TwoGaussians(Dataset):
    def __init__(self):
        self.dataset = torch.load(dataset_path)
        super().__init__()

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        dummy_label = idx   # Our code works with datasets where datapoints have labels, like MNIST
                            # We don't have labels here, but to be compatible with other code, we invent one
                            # We discard it anyway before passing it to the ML-model
                            # Using the index might be useful for debugging
        return (self.dataset[idx], dummy_label)

if __name__ == '__main__':
    first_gaussian_mean = [15, 15]
    first_gaussian_std = [[5, 0], [0, 5]]
    second_gaussian_mean = [35, 25]
    second_gaussian_std = [[10, 0], [0, 10]]
    bias_to_first_gaussian = 0.60

    def naive_gaussian_mixture():
        if np.random.rand() <= bias_to_first_gaussian:
            return np.round(
                np.random.multivariate_normal(first_gaussian_mean, first_gaussian_std)
            )
        else:
            return np.round(
                np.random.multivariate_normal(second_gaussian_mean, second_gaussian_std)
            )

    def generate_samples(num_samples):
        samples = np.ndarray(shape = (num_samples, 2, 1))
        samples_generated = 0
        while samples_generated < num_samples:
            x = naive_gaussian_mixture()
            if 0.0 <= x[0] <= 49.0 and 0.0 <= x[1] <= 49.0:
                samples[samples_generated, 0, 0] = x[0]
                samples[samples_generated, 1, 0] = x[1]
                samples_generated += 1
        return samples

    print("Creating dataset...")
    dataset = torch.tensor(generate_samples(dataset_size))

    print("Saving...")
    Path(dataset_folder).mkdir(parents=True, exist_ok=True)
    torch.save(dataset, dataset_path)

    dataset = TwoGaussians()
    print("Length of dataset: ", dataset.__len__())
    print("Random element: ", dataset.__getitem__(np.random.randint(dataset.__len__())))

    heatmap = np.zeros(shape = (50, 50))

    for datapoint, _ in dataset:
        x = datapoint[0, 0]
        y = datapoint[1, 0]
        heatmap[int(x), int(y)] += 1.0
    heatmap = heatmap / heatmap.max()

    plt.imshow(
        heatmap,
        cmap="viridis",
        origin="lower"
    )
    #plt.colorbar(label="Relative density")
    plt.show()












