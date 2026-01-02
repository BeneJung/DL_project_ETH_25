import torch
import numpy as np
import time
import ot
import os
import matplotlib.pyplot as plt

from fldd import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# fix for certificate error
import os
import certifi

# Set environment variables to use certifi's bundle
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

class SinkhornTransportModel(nn.Module):
    """
    Implements a optimal transport plan following the method presented in "Sinkhorn Distances: 
    Lightspeed computation of optimal transportation distances" by Cuturi, 2015.
    Computes the transportation matrix with the Sinkhorn-Knopp-Algorithm.
    For distributions a = u_s, b = u_t, we aim to find P with P1 = a and P^T=b for which 
    argmin_{P >= 0} <P,C> - epsilon h(P) to compute u_(s|t)
    """
    C: torch.Tensor

    def __init__(self, vocab, lam=1, threshold=5e-7, C=None, learnable=False):
        # Default: C_{u,v} = 1_{u\neq v}
        super().__init__()
        print("Using Sinkhorn Transport")
        if C is not None:
            self.C = C
            assert self.C.shape[-2:] == (vocab, vocab)
        else:
            self.C: torch.Tensor = torch.ones((vocab, vocab)).fill_diagonal_(0)
        if learnable:
            self.C = torch.nn.Parameter(self.C)
        self.learnable = learnable
        self.vocab = vocab
        self.lam = lam
        self.threshold = threshold

    def compute_optimal_transport(self, a: torch.Tensor, b: torch.Tensor):
        """
        Computes the optimal transport matrix and Slinkhorn distance using the
        Sinkhorn-Knopp algorithm with self.C (of shape (K, K) ) as cost matrix

        Args:
          a: source distribution, of shape (..., K)
          b: target distribution, of shape (..., K)
        Returns:
          P: optimal transport matrix of shape (K, K), res[r, c] = q_(z_s = r, z_t = s)
          err: Error
        """
        # print(self.learnable, self.C)
        ndim = a.ndim
        eps = 1e-8
        P: torch.Tensor = torch.exp(-self.C / self.lam)
        P = P.to(device=device)
        P = P.view([1 for _ in range(ndim-1)]+[self.vocab, self.vocab])
        pdim = list(a.shape[:-1])+[1, 1]
        P = P.repeat(pdim)
        # make sure sum over last two dimension add up to 0
        P /= P.sum(dim=(-2, -1), keepdim=True) + eps

        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)
        b = b.reshape([*a.shape[:-2]]+[1, -1])  # to col vec

        err = 1
        i = 0
        while err > self.threshold:
            # different computation than in proposal and paper
            # but equivalent calculation
            row_ratio = a / (P.sum(dim=-1, keepdim=True) + eps)
            P *= row_ratio
            col_ratio = b / (P.sum(dim=-2, keepdim=True) + eps)
            P *= col_ratio

            err_row = torch.max(torch.abs(P.sum(dim=-1, keepdim=True) - a))
            err_col = torch.max(torch.abs(P.sum(dim=-2, keepdim=True) - b))

            err = torch.max(err_row, err_col)

            i += 1

            if i == 1500:
                print(
                    f"Warning: Sinkhorn Transport Model did not converge after 1500 iterations, error: {err} > {self.threshold}")
                break
        # Sanity check
        # a = a.squeeze(-1)
        # b = b.squeeze(-2)
        # print(torch.allclose(P.sum(dim=(-1)), a, rtol=1e-05, atol=1e-08))
        # print(torch.allclose(P.sum(dim=(-2)), b, rtol=1e-05, atol=1e-08))
        return P, err

    def compute_posterior(self, u_s: torch.Tensor, u_t: torch.Tensor, z_t: torch.Tensor):
        '''
        Compute posterior q(z_s | z_t) using Sinkhorn Transport Model

        :param u_s: Marginal parameters at time s [batch, H, W, vocab]
        :param u_t: Marginal parameters at time t [batch, H, W, vocab]
        :param z_t: Discrete samples at time t [batch, H, W]

        :out res: shape [batch, H, W, vocab], res[b,h,w,r] = q(z_s = r | z_t = z_t)
        '''
        eps = 1e-8

        a, b = u_s, u_t
        P, err = self.compute_optimal_transport(
            a, b)  # shape [batch, H, W, vocab, vocab]
        res = P / (P.sum(-2, keepdim=True) + eps)
        index = z_t.unsqueeze(-1).unsqueeze(-1).expand(
            list(a.shape[:-1]) + [self.vocab, 1])
        u_s_given_t = torch.gather(res, dim=-1, index=index).squeeze(-1)
        return u_s_given_t

    def compute_posterior_soft(self, u_s: torch.Tensor, u_t: torch.Tensor, z_t_soft: torch.Tensor):
        '''
        Compute posterior for soft/relaxed samples (Equation 14).
        Taken from MaximumCoupling class 

        :param u_s: Marginal parameters at time s [batch, H, W, vocab]
        :param u_t: Marginal parameters at time t [batch, H, W, vocab]
        :param z_t_soft: Soft samples at time t [batch, H, W, vocab]

        :out res: shape [batch, H, W, vocab], res[b,h,w,r] = q(z_s = r | z_t = z_t)
        '''
        batch_size, H, W, vocab = u_s.shape
        device = u_s.device

        # Initialize output
        posterior = torch.zeros_like(u_s, device=device)
        # Precompute optimal transport (the same matrix for all k)
        eps = 1e-8
        a, b = u_s, u_t
        P, err = self.compute_optimal_transport(
            a, b)  # shape [batch, H, W, vocab, vocab]
        res = P / (P.sum(-2, keepdim=True) + eps)

        # Weighted average over all possible discrete values k
        for k in range(vocab):
            # Create hard samples where z_t = k everywhere
            z_t_hard = torch.full((batch_size, H, W), k,
                                  dtype=torch.long, device=device)

            # Compute posterior assuming z_t = k
            index = z_t_hard.unsqueeze(-1).unsqueeze(-1).expand(
                list(a.shape[:-1]) + [self.vocab, 1])
            posterior_k =  torch.gather(res, dim=-1, index=index).squeeze(-1)

            # Weight by the soft probability of z_t being k
            weight = z_t_soft[:, :, :, k:k+1]  # [batch, H, W, 1]
            posterior = posterior + posterior_k * weight
        return posterior


def sinkhorn_testing():
    '''Local testing, can be ignored'''
    vocab = 3
    a = torch.Tensor([[0.2, 0.2, 0.6]])
    a = torch.rand(size=(16, 10, 10, vocab))
    # a = torch.rand(size=(vocab,))
    a /= a.sum(dim=-1, keepdim=True)

    b = torch.Tensor([[0.1, 0.8, 0.1]])
    b = torch.rand(size=(16, 10, 10, vocab))
    # b = torch.rand(size=(vocab,))
    b /= b.sum(dim=-1, keepdim=True)

    reg = 1
    start = time.time()
    tmodel = SinkhornTransportModel(vocab=vocab, lam=reg)
    print("before")
    z_t_soft = torch.randint(0, vocab, size=(16, 10, 10, vocab)).float()
    z_t_soft /= z_t_soft.sum(dim=-1, keepdim=True)
    # z_t = torch.Tensor([1]).long()
    transport_matrix, err = tmodel.compute_optimal_transport(
        a, b)

    pos = tmodel.compute_posterior_soft(a, b, z_t_soft)
    # print(z_t_soft[0, 0, 0], pos[0, 0, 0])
    end = time.time()
    elapsedtime = end-start
    print(f"Own Time: {elapsedtime} s")
    print("Own error:", err)
    exit()

    # start = time.time()
    # gamma : ArrayLike = ot.sinkhorn(a, b, C, reg, stopThr=10e-8)  # type: ignore
    # end = time.time()
    # elapsedtime = end-start

    # print(f"POT time: {elapsedtime} s")
    # err_row = torch.max(torch.abs(gamma.sum(dim=-1) - a))
    # err_col = torch.max(torch.abs(gamma.sum(dim=-2) - b))
    # err = torch.max(err_row, err_col)
    # print("POT error:", err)

    print(transport_matrix)
    # print(gamma)
    pass


def main():
    """Main training script taken from fldd file"""
    print("=" * 60)
    print("Forward-Learned Discrete Diffusion (FLDD) - Corrected")
    print("=" * 60)
    print(f"Device: {device}")

    # Create output directory
    os.makedirs('./outputs', exist_ok=True)

    # Prepare data
    # dataset_name = "MNIST"
    dataset_name = "TwoGaussians"
    train_loader, test_loader = prepare_data(
        dataset=dataset_name, batch_size=128)

    # Initialize model
    # Using T=10 as in paper experiments for few-step generation
    vocab_size, num_timesteps = (
        2, 10) if dataset_name == "MNIST" else (50, 10)
    if dataset_name == "MNIST":
        vocab_size = 2
        num_timesteps = 10
        hidden_dim = 128
        time_dim = 128
        forward_nn_model = FLDDNetwork(
            vocab_size, time_dim, hidden_dim).to(device)
        reverse_nn_model = FLDDNetwork(
            vocab_size, time_dim, hidden_dim).to(device)
    elif dataset_name == "TwoGaussians":
        vocab_size = 50
        num_timesteps = 2
        forward_nn_model = FLDDTwoGaussians(vocab_size=vocab_size).to(device)
        reverse_nn_model = FLDDTwoGaussians(vocab_size=vocab_size).to(device)
    else:
        raise Exception("Invalid or unknown dataset name")

    transport = "sinkhorn"
    # transport = "sinkhorn_learnable"
    model = FLDD(
        vocab_size=vocab_size,      # Binary images
        num_timesteps=num_timesteps,  # Few-step generation
        forward_net=forward_nn_model,
        reverse_net=reverse_nn_model,
        transportplan=transport,
        warmup_steps=1000
    )

    print(f"\nModel Configuration:")
    print(f"  - Timesteps (T): {model.num_timesteps}")
    print(f"  - Vocabulary size: {model.vocab_size}")
    print(f"  - Warmup steps: {model.warmup_steps:,}")
    print(f"  - REINFORCE steps: {model.reinforce_steps:,}")
    print(f"  - Total steps: {model.total_steps:,}")
    print(f"  - Learning rate: 2e-4 (AdamW)")
    print(f"  - Temperature: {model.temperature} → {model.min_temperature}")
    print(f"  - Transport Plan: {transport}")

    # Count parameters
    forward_params = sum(p.numel() for p in model.forward_net.parameters())
    reverse_params = sum(p.numel() for p in model.reverse_net.parameters())
    print(f"  - Forward network params: {forward_params:,}")
    print(f"  - Reverse network params: {reverse_params:,}")
    print("=" * 60)

    # Train model
    losses = train_model(model, train_loader, dataset_name, num_epochs=30)

    # Plot training curve
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    warmup_epochs = model.warmup_steps / len(train_loader)
    plt.axvline(x=warmup_epochs, color='r',
                linestyle='--', label='Warmup → REINFORCE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FLDD Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./outputs/training_curve_{transport}_{dataset_name}.png', dpi=150)
    plt.close()

    if dataset_name == "TwoGaussians":
        samples = model.sample(num_samples=1000, sample_shape=(
            2, 1)).squeeze(-1).cpu().numpy()
        heatmap = np.zeros(shape=(50, 50))
        for i in range(samples.shape[0]):
            x = samples[i, 0]
            y = samples[i, 1]
            if 0 <= x < 50 and 0 <= y < 50:
                heatmap[x, y] += 1.0
            else:
                print("Invalid Coordinates: " + str(x) + ", " + str(y))
        heatmap = heatmap / heatmap.max()
        plt.imshow(
            heatmap,
            cmap="viridis",
            origin="lower"
        )
        plt.savefig(f'./outputs/gaussian_samples_{transport}.png')
        plt.show()
        import sys
        sys.exit(0)

    # Visualize diffusion process
    visualize_diffusion_process(model, test_loader, dataset_name)

    # Generate final samples
    print("\nGenerating final samples...")
    samples = model.sample(num_samples=64)

    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].cpu().numpy(), cmap='gray')
        ax.axis('off')

    plt.suptitle(f'FLDD Final Samples (T={model.num_timesteps} steps)')
    plt.tight_layout()
    plt.savefig('./outputs/final_samples.png', dpi=150)
    plt.close()

    # Save model
    torch.save({
        'forward_net': model.forward_net.state_dict(),
        'reverse_net': model.reverse_net.state_dict(),
        'config': {
            'vocab_size': model.vocab_size,
            'num_timesteps': model.num_timesteps,
            'total_steps_trained': model.current_step
        }
    }, './outputs/fldd_mnist_final.pth')

    print(f"\n{'=' * 60}")
    print(f"✓ Training complete!")
    print(f"✓ Model saved to ./outputs/fldd_mnist_final.pth")
    print(f"✓ Total steps trained: {model.current_step:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
