
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# NEURAL NETWORK
# =============================================================================

""" Sinusoidal timestep embeddings as used in diffusion models """
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

""" Neural network that learns the forward and backward process """
class NeuralNetwork(nn.Module):

    def __init__(self, vocab_size = 50):
        super().__init__()
        self.time_size = 60
        self.vocab_size = vocab_size
        self.time_net = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_size),
        )
        self.net = nn.Sequential(
            nn.Linear(2 + self.time_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * vocab_size),
        )

    def forward(self, x, time):
        """
        x: [batch, 2] discrete indices
        time: [batch] of the same number
        Returns: [batch, 2, vocab_size] logits
        """
        batch_size = x.shape[0]

        # transform x so that it is a one-hot vector of length 2*vocab_size
        x_onehot = F.one_hot(x.long(), self.vocab_size).float()
        x_input = x_onehot.reshape(batch_size, -1)

        time_repeated = torch.full([batch_size], time)
        time_emb = self.time_net(time_repeated)
        input = torch.cat((x_input, time_emb), dim = -1)
        output = self.net(input)

        # Reshape: [batch, 2, vocab_size]
        logits = output.view(batch_size, 2, self.vocab_size)

        return logits


# =============================================================================
# SIMPLIFIED FLDD (NO MAXIMUM COUPLING, JUST DIRECT)
# =============================================================================

class SimpleFLDD:
    """
    Extremely simplified FLDD to test basic behavior.
    NO Maximum Coupling - just use the forward marginals directly as posteriors.
    """

    def __init__(self, vocab_size=50, num_timesteps=2):
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.device = device

        self.forward_net = NeuralNetwork(vocab_size).to(device)
        self.reverse_net = NeuralNetwork(vocab_size).to(device)

        self.optimizer = torch.optim.AdamW(
            list(self.forward_net.parameters()) + list(self.reverse_net.parameters()),
            lr=1e-3
        )

    def get_forward_marginals(self, x, t):
        """Get q(z_t | x) - factorized over dimensions."""
        batch_size = x.shape[0]

        if t == 0:
            # Deterministic: z_0 = x
            probs = F.one_hot(x.long(), self.vocab_size).float()
            return probs

        elif t == self.num_timesteps:
            # Uniform prior
            probs = torch.ones(batch_size, 2, self.vocab_size, device=self.device)
            probs = probs / self.vocab_size
            return probs
        else:
            # Use network
            logits = self.forward_net(x, t)
            probs = F.softmax(logits, dim=-1)
            return probs

    def get_reverse_probs(self, z_t, t):
        """Get p(z_s | z_t) - factorized over dimensions."""
        logits = self.reverse_net(z_t, t)
        probs = F.softmax(logits, dim=-1)
        return probs

    def train_step(self, x):
        """Single training step - simplified."""
        self.optimizer.zero_grad()

        # Sample timestep
        t = np.random.randint(1, self.num_timesteps + 1)
        s = t - 1

        # Get forward marginals
        u_s = self.get_forward_marginals(x, s)  # [batch, 2, vocab]
        u_t = self.get_forward_marginals(x, t)  # [batch, 2, vocab]

        # Sample z_t from u_t
        dist_t = Categorical(probs=u_t + 1e-8)
        z_t = dist_t.sample()  # [batch, 2]

        # SIMPLIFIED POSTERIOR: Just use u_s directly (no Maximum Coupling)
        # This is WRONG but let's see if even this works
        u_s_given_t = u_s

        # Get reverse probs
        v_s_given_t = self.get_reverse_probs(z_t, t)

        # KL divergence: sum over dimensions and vocab
        kl = (u_s_given_t * (torch.log(u_s_given_t + 1e-8) - torch.log(v_s_given_t + 1e-8))).sum(dim=-1).sum(dim=-1)
        loss = kl.mean()

        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, num_samples=1000):
        """Generate samples."""
        self.forward_net.eval()
        self.reverse_net.eval()

        # Start from uniform
        z_t = torch.randint(0, self.vocab_size, (num_samples, 2), device=self.device)

        # Reverse diffusion
        for t in reversed(range(1, self.num_timesteps + 1)):
            probs = self.get_reverse_probs(z_t, t)
            dist = Categorical(probs=probs + 1e-8)
            z_t = dist.sample()

        self.forward_net.train()
        self.reverse_net.train()

        return z_t


# =============================================================================
# TESTING
# =============================================================================

def generate_data(num_samples = 10000):
    gaussian1_mean = np.array([15.0, 15.0])
    gaussian1_cov = np.array([[5, 0], [0, 5]])
    gaussian2_mean = np.array([35.0, 25.0])
    gaussian2_cov = np.array([[10, 0], [0, 10]])

    data = []
    while len(data) < num_samples:
        if np.random.rand() < 0.6:
            sample = np.random.multivariate_normal(gaussian1_mean, gaussian1_cov).round()
        else:
            sample = np.random.multivariate_normal(gaussian2_mean, gaussian2_cov).round()
        if 0 <= sample[0] < 50 and 0 <= sample[1] < 50:
            data.append(sample)

    return torch.tensor(np.array(data), dtype=torch.long)


def visualize_samples(samples, title="Samples"):
    heatmap = np.zeros((50, 50))
    samples_np = samples.cpu().numpy()

    for i in range(samples_np.shape[0]):
        x, y = int(samples_np[i, 0]), int(samples_np[i, 1])
        heatmap[x, y] += 1
    heatmap = heatmap / (heatmap.max() + 1e-8)

    plt.imshow(heatmap, cmap='viridis', origin='lower')
    plt.show()

def try_forward_process(model, data):
    """Test what the forward process learns."""
    model.forward_net.eval()

    # Pick a few data points
    test_points = data[:4].to(device)

    print("\n=== Forward Process Analysis ===")
    for i, point in enumerate(test_points):
        print(f"\nData point {i}: x={point[0].item()}, y={point[1].item()}")

        for t in range(model.num_timesteps + 1):
            probs = model.get_forward_marginals(point.unsqueeze(0), t)

            # Get top-3 most likely values for each dimension
            probs_np = probs[0].cpu().detach().numpy()

            for dim in range(2):
                top_vals = np.argsort(probs_np[dim])[-3:][::-1]
                top_probs = probs_np[dim][top_vals]
                print(f"  t={t}, dim={dim}: top values = {top_vals}, probs = {top_probs}")

    model.forward_net.train()


def main():
    data = generate_data(10000)
    #visualize_samples(data, "Training Data")

    model = SimpleFLDD(vocab_size=50, num_timesteps=2)
    print("\nTraining...")
    batch_size = 128
    num_epochs = 50

    for epoch in range(num_epochs):
        losses = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device)
            loss = model.train_step(batch)
            losses.append(loss)

        avg_loss = np.mean(losses)
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}")

    samples = model.sample(num_samples=1000)
    visualize_samples(samples, "Samples")

if __name__ == "__main__":
    main()