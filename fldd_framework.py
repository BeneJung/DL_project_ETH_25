
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import RelaxedOneHotCategorical

from random import randint

from gaussian_neutral_net import NeuralNetwork

class FLDD:
    def __init__(self, num_warmup_steps = 1000, num_reinforce_steps = 1000):
        self.vocab_size = 50

        self.num_warmup_steps = num_warmup_steps
        self.num_reinforce_steps = num_reinforce_steps

        self.total_num_diffuse_steps = 2
        self.forward_net = NeuralNetwork()
        self.backward_net = NeuralNetwork()
        self.optimizer = optim.Adam(
            params = list(self.forward_net.parameters()) + list(self.backward_net.parameters()),
            lr = 2e-4
        )

        # The following values (1 & 1e-3) are taken from the paper
        self.temperature = 1.0
        self.final_temperature = 1e-3
        self.temp_decay_rate = (self.final_temperature / self.temperature) ** (1.0 / self.num_warmup_steps)

    """
    Given a data point x and a time t, returns the probabilities q_φ(z_t | x) of the latent variable z_t as a tensor
    This tensor is shaped like x and z_t but with an additional dimension, where for all coordinates of z_t
    at entry i along that dimension it contains the probability q_φ(z_t = i | x)
    """
    def get_forward_marginals(self, x, time):
        if time == 0:
            return F.one_hot(x) # By definition z_t at time 0 equals x

        if time == self.total_num_diffuse_steps:
            uniform_prob = 1.0 / self.vocab_size
            return torch.full(x.shape, uniform_prob) # By definition z_t at the last timestep is uniformly diffused

        logits = self.forward_net(x, time)
        probabilities = F.softmax(logits, dim = -1)
        # TODO: Benedikt interpolates between these probabilities from the network and the boundary conditions
        # Should we do that here too?
        return probabilities

    """
    Given a latent state z_t at a time t, returns the probabilities p_θ(z_s | z_t) where s = t-1 as a tensor
    This tensor encodes them in the same way as get_forward_marginals()
    """
    def get_backward_probabilities(self, x, time):
        logits = self.backward_net(x, time)
        probabilities = F.softmax(logits, dim = -1)
        return probabilities

    def warmup_step(self, x):
        t = randint(1, self.total_num_diffuse_steps)
        s = t - 1

        u_t = self.get_forward_marginals(x, t)
        u_s = self.get_forward_marginals(x, s)

        # Note that z_t will be a "soft" sample, not a "hard" sample. It will not be a vector of discrete elements
        # but a probability distribution over all the discrete elements
        z_t = RelaxedOneHotCategorical(temperature = self.temperature, probs = u_t + 1e-8).rsample()

        if s == 0:
            u_s_given_t = F.one_hot(x, self.vocab_size).float()
        else:
            u_s_given_t = None # TODO: Implement Maximum Coupling

        v_s_given_t = self.get_backward_probabilities(z_t, t)

        # Prevent -∞ in the KL divergence
        safe_v_s_given_t = v_s_given_t.clamp(min = 1e-8)
        logits_v_s_given_t = torch.log(safe_v_s_given_t)
        kl_loss = F.kl_div(input = u_s_given_t, target = logits_v_s_given_t).mean()

"""
max_mean_discrepancy_calculator = SamplesLoss("gaussian")
        with torch.no_grad():
            test_dataset = test_loader.dataset
            test_dataset = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).squeeze()
            torch_samples = torch.from_numpy(samples)
            max_mean_discrepancy = max_mean_discrepancy_calculator(torch_samples.float(), test_dataset.float())
            print("MMD between model samples and dataset is: ", max_mean_discrepancy)
"""
