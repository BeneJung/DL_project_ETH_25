
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    The following technique was invented by "Attention is all you need"
    This layer takes the current time (int) and returns a vector that still encodes a sense of time
    Unlike an int, this encoding generalises if the network tests for longer than any timestep it saw during learning
    Google "Sinusoidal Position Embeddings" for more information
"""
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10_000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

"""
    Neural network that learns the forward and backward process
    Note that it only takes "soft" inputs that are preprocessed such that they represent a probability distribution
    If, let's say, the vocab of the dataset is {0, 1, 2}, "hard" datapoints such as [2, 1]
    must be converted to [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]] beforehand
"""
class NeuralNetwork(nn.Module):

    def __init__(self, vocab_size = 50, num_dims_of_input = 2):
        super().__init__()
        self.time_size = 60
        self.vocab_size = vocab_size
        self.num_dims_of_input = num_dims_of_input

        self.time_net = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_size),
        )

        input_size = num_dims_of_input * vocab_size + self.time_size # Flattened "soft" datapoint + encoded time
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * vocab_size),
        )

    def forward(self, x, time):
        batch_size = x.shape[0]
        x_flattened = x.reshape(batch_size, -1)

        batched_time = torch.full([batch_size], float(time)) # A vector full of time
        time_emb = self.time_net(batched_time)
        input = torch.cat((x_flattened, time_emb), dim = -1)
        output = self.net(input)

        # Reshape: [batch, 2, vocab_size]
        logits = output.view(batch_size, self.num_dims_of_input, self.vocab_size)
        return logits