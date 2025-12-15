"""
Forward-Learned Discrete Diffusion (FLDD) - Corrected Implementation for MNIST

This implementation follows the paper "Forward-Learned Discrete Diffusion: 
Learning How to Noise to Denoise Faster" (ICLR 2026 submission).

Key Concepts:
1. Learn the forward (noising) process so that the induced target distribution
   becomes factorized, matching what the reverse model can represent.
2. Use Maximum Coupling for posterior computation.
3. Train with Concrete relaxation warm-up followed by REINFORCE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, RelaxedOneHotCategorical
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal timestep embeddings as used in diffusion models.
    
    These embeddings encode the timestep t into a continuous vector space,
    allowing the network to understand "when" in the diffusion process it is.
    
    The formula is:
        PE(t, 2i) = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm and time embedding injection.
    
    Time conditioning is done via FiLM (Feature-wise Linear Modulation):
    the time embedding is projected to scale and shift parameters.
    """
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_dim, out_channels * 2)  # scale and shift
        
        # Skip connection if dimensions change
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Time conditioning via FiLM
        t_params = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = t_params.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        h = F.gelu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.gelu(h)
        
        return h + self.skip(x)


class FLDDNetwork(nn.Module):
    """
    U-Net style network for both forward and reverse processes.
    
    For the FORWARD network:
        - Input: x (original data) and t (timestep)
        - Output: parameters u_φ(x, t) for q_φ(z_t|x)
        
    For the REVERSE network:
        - Input: z_t (noisy state) and t (timestep)  
        - Output: parameters v_θ(z_t, t) for p_θ(z_s|z_t)
    
    Both networks have the same architecture as specified in the paper.
    """
    def __init__(self, vocab_size=2, time_dim=128, hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.time_dim = time_dim
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Initial convolution: one-hot input → hidden
        self.init_conv = nn.Conv2d(vocab_size, hidden_dim, 3, padding=1)
        
        # Encoder (downsampling path)
        self.enc1 = ResidualBlock(hidden_dim, hidden_dim, time_dim)
        self.down1 = nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)  # 28→14
        
        self.enc2 = ResidualBlock(hidden_dim, hidden_dim * 2, time_dim)
        self.down2 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 4, stride=2, padding=1)  # 14→7
        
        # Middle (bottleneck)
        self.mid1 = ResidualBlock(hidden_dim * 2, hidden_dim * 2, time_dim)
        self.mid2 = ResidualBlock(hidden_dim * 2, hidden_dim * 2, time_dim)
        
        # Decoder (upsampling path)
        self.up2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, 4, stride=2, padding=1)  # 7→14
        self.dec2 = ResidualBlock(hidden_dim * 4, hidden_dim, time_dim)  # *4 due to skip connection
        
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)  # 14→28
        self.dec1 = ResidualBlock(hidden_dim * 2, hidden_dim, time_dim)  # *2 due to skip connection
        
        # Output: hidden → vocab_size logits
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, vocab_size, 3, padding=1)
        )

    def forward(self, x, t):
        """
        Args:
            x: [batch, H, W] discrete indices OR [batch, vocab, H, W] one-hot/soft
            t: [batch] timesteps (integers or floats)
        Returns:
            logits: [batch, H, W, vocab_size] - unnormalized log probabilities
        """
        # Convert to one-hot if needed
        if x.dim() == 3:
            x_onehot = F.one_hot(x.long(), self.vocab_size).float()
            x_onehot = x_onehot.permute(0, 3, 1, 2)  # [B, vocab, H, W]
        else:
            x_onehot = x  # Already [B, vocab, H, W]
        
        # Time embedding
        t_emb = self.time_mlp(t.float())
        
        # Initial projection
        h = self.init_conv(x_onehot)
        
        # Encoder
        h1 = self.enc1(h, t_emb)
        h = self.down1(h1)
        
        h2 = self.enc2(h, t_emb)
        h = self.down2(h2)
        
        # Middle
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)
        
        # Decoder with skip connections
        h = self.up2(h)
        h = torch.cat([h, h2], dim=1)
        h = self.dec2(h, t_emb)
        
        h = self.up1(h)
        h = torch.cat([h, h1], dim=1)
        h = self.dec1(h, t_emb)
        
        # Output logits
        logits = self.out_conv(h)
        
        # Reshape to [batch, H, W, vocab]
        logits = logits.permute(0, 2, 3, 1)
        
        return logits


# =============================================================================
# MAXIMUM COUPLING (Equation 11)
# =============================================================================

class MaximumCoupling:
    """
    Implements Maximum Coupling for posterior computation.
    
    Maximum Coupling constructs a transport plan between two distributions
    that minimizes the probability of changing state. It's used to define
    q(z_s | z_t, x) given the marginals q(z_s|x) and q(z_t|x).
    
    From Equation 11 in the paper:
    For z_t = k, the posterior u_{s|t} is:
        u_{s|t}^j = min(u_s^k, u_t^k) / u_t^k           if j = k  (stay)
        u_{s|t}^j = max(0, u_t^k - u_s^k) × m_{s|t}^j   if j ≠ k  (move)
    
    where m_{s|t} = clamp(u_s - u_t, min=0) / ||clamp(u_s - u_t, min=0)||_1
    is the normalized distribution of "deficit" states (states that need
    more probability at time s than they have at time t).
    """
    
    @staticmethod
    def compute_posterior(u_s, u_t, z_t):
        """
        Compute q(z_s | z_t, x) using Maximum Coupling (Equation 11).
        
        Maximum Coupling constructs a transport plan between two distributions
        that minimizes the probability of changing state. It's used to define
        q(z_s | z_t, x) given the marginals q(z_s|x) and q(z_t|x).
        
        From Equation 11 in the paper:
        For z_t = k, the posterior u_{s|t} is:
            u_{s|t}^j = min(u_s^k, u_t^k) / u_t^k                    if j = k  (stay)
            u_{s|t}^j = max(0, u_t^k - u_s^k) / u_t^k × m_{s|t}^j    if j ≠ k  (move)
        
        where m_{s|t} = clamp(u_s - u_t, min=0) / ||clamp(u_s - u_t, min=0)||_1
        is the normalized distribution of "deficit" states.
        
        KEY INSIGHT: The move coefficient must be FRACTIONAL (divided by u_t^k),
        not absolute. This ensures:
            stay_prob + move_coeff = min(u_s^k, u_t^k)/u_t^k + max(0, u_t^k - u_s^k)/u_t^k = 1
        So the posterior sums to 1 WITHOUT any final normalization.
        
        Args:
            u_s: [batch, H, W, vocab] - marginal parameters at time s
            u_t: [batch, H, W, vocab] - marginal parameters at time t
            z_t: [batch, H, W] - discrete samples at time t
        
        Returns:
            u_s_given_t: [batch, H, W, vocab] - posterior parameters (sum to 1)
        """
        batch_size, H, W, vocab = u_s.shape
        device = u_s.device
        eps = 1e-8
        
        # =====================================================================
        # Step 1: Get the probability at the current state k = z_t
        # =====================================================================
        # We need u_t^k and u_s^k for each position, where k is the sampled state
        z_t_long = z_t.long().unsqueeze(-1)  # [batch, H, W, 1] - index for gather
        
        # Gather u_t[k] and u_s[k] for k = z_t at each spatial position
        # Result shape: [batch, H, W] after squeeze
        u_t_k = torch.gather(u_t, dim=-1, index=z_t_long).squeeze(-1)  # [batch, H, W]
        u_s_k = torch.gather(u_s, dim=-1, index=z_t_long).squeeze(-1)  # [batch, H, W]
        
        # =====================================================================
        # Step 2: Probability of STAYING in state k
        # =====================================================================
        # stay_prob = min(u_s^k, u_t^k) / u_t^k
        # 
        # Intuition: We can only "keep" probability mass at state k up to the 
        # minimum of what's available at t (u_t^k) and what's needed at s (u_s^k).
        # We divide by u_t^k because we're computing P(z_s=k | z_t=k).
        stay_prob = torch.minimum(u_s_k, u_t_k) / (u_t_k + eps)  # [batch, H, W]
        
        # =====================================================================
        # Step 3: Probability of MOVING away from state k (FRACTIONAL coefficient)
        # =====================================================================
        # move_coeff = max(0, u_t^k - u_s^k) / u_t^k
        #
        # This is the FRACTIONAL probability of moving, NOT the absolute excess.
        # Critical: Dividing by u_t^k ensures stay_prob + move_coeff = 1
        #
        # Intuition: If u_t^k > u_s^k, there's excess probability at k that must
        # be redistributed. The fraction that moves is (u_t^k - u_s^k) / u_t^k.
        move_coeff = torch.clamp((u_t_k - u_s_k) / (u_t_k + eps), min=0)  # [batch, H, W]
        
        # =====================================================================
        # Step 4: Compute deficit distribution m_{s|t}
        # =====================================================================
        # Deficit: states that need MORE probability at time s than at time t
        # m_{s|t} = clamp(u_s - u_t, min=0) / ||clamp(u_s - u_t, min=0)||_1
        #
        # This tells us WHERE to redistribute the excess probability.
        # States with u_s^j > u_t^j have a deficit and will receive probability.
        # Note: At state k (where there's excess), deficit^k = 0, so we never
        # "move to k" when moving away from k.
        deficit = torch.clamp(u_s - u_t, min=0)  # [batch, H, W, vocab]
        deficit_sum = deficit.sum(dim=-1, keepdim=True) + eps  # [batch, H, W, 1]
        m_s_t = deficit / deficit_sum  # Normalized deficit distribution [batch, H, W, vocab]
        
        # =====================================================================
        # Step 5: Compute move probabilities for each target state j
        # =====================================================================
        # move_prob^j = move_coeff × m_{s|t}^j  for all j
        #
        # We distribute the fractional move probability according to the deficit.
        # Note: move_prob^k = move_coeff × m_{s|t}^k = move_coeff × 0 = 0
        # (because deficit^k = 0 when there's excess at k)
        move_prob = move_coeff.unsqueeze(-1) * m_s_t  # [batch, H, W, vocab]
        
        # =====================================================================
        # Step 6: Combine stay and move probabilities
        # =====================================================================
        # Start with move probabilities (which are 0 at position k)
        u_s_given_t = move_prob.clone()
        
        # Create a mask that is 1 at position k and 0 elsewhere
        # Then use it to insert stay_prob at the correct position
        mask = torch.zeros_like(u_s_given_t).scatter_(-1, z_t_long, 1.0)
        
        # Replace position k with stay_prob, keep move_prob for other positions
        u_s_given_t = u_s_given_t * (1 - mask) + stay_prob.unsqueeze(-1) * mask
        
        # =====================================================================
        # Final result: NO NORMALIZATION NEEDED
        # =====================================================================
        # The posterior should sum to 1 exactly:
        #   sum_j u_{s|t}^j = stay_prob + sum_{j≠k} move_prob^j
        #                   = stay_prob + move_coeff × sum_j m_{s|t}^j
        #                   = stay_prob + move_coeff × 1
        #                   = stay_prob + move_coeff = 1
        #
        # We only clamp to eps for numerical stability (avoid exact zeros)
        return u_s_given_t.clamp(min=eps)
    
    @staticmethod
    def compute_posterior_soft(u_s, u_t, z_t_soft):
        """
        Compute posterior for soft/relaxed samples (Equation 14).
        
        This is used during the warm-up phase with Concrete distribution.
        The soft posterior is a weighted average of hard posteriors:
            q(z_s | z̄_t, x) = Σ_k z̄_t^k × q(z_s | z_t=k, x)
        
        Args:
            u_s: [batch, H, W, vocab] - marginal parameters at time s
            u_t: [batch, H, W, vocab] - marginal parameters at time t
            z_t_soft: [batch, H, W, vocab] - soft samples (sum to 1 over vocab)
        
        Returns:
            posterior: [batch, H, W, vocab] - soft posterior parameters
        """
        batch_size, H, W, vocab = u_s.shape
        device = u_s.device
        
        # Initialize output
        posterior = torch.zeros_like(u_s)
        
        # Weighted average over all possible discrete values k
        for k in range(vocab):
            # Create hard samples where z_t = k everywhere
            z_t_hard = torch.full((batch_size, H, W), k, dtype=torch.long, device=device)
            
            # Compute posterior assuming z_t = k
            posterior_k = MaximumCoupling.compute_posterior(u_s, u_t, z_t_hard)
            
            # Weight by the soft probability of z_t being k
            weight = z_t_soft[:, :, :, k:k+1]  # [batch, H, W, 1]
            posterior = posterior + posterior_k * weight
        
        return posterior


# =============================================================================
# FLDD MODEL
# =============================================================================

class FLDD:
    """
    Forward-Learned Discrete Diffusion model.
    
    This class implements the full FLDD framework:
    1. Learnable forward process q_φ(z_t|x) with Maximum Coupling posteriors
    2. Standard factorized reverse process p_θ(z_s|z_t)
    3. Two-phase training: Concrete warm-up → REINFORCE
    """
    
    def __init__(self, vocab_size=2, num_timesteps=10, hidden_dim=128, time_dim=128):
        """
        Args:
            vocab_size: Number of discrete values (2 for binary MNIST)
            num_timesteps: Number of diffusion steps T
            hidden_dim: Hidden dimension for networks
            time_dim: Dimension of time embeddings
        """
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.device = device
        
        # =====================================================================
        # Networks
        # =====================================================================
        # Forward network: takes x (data), outputs u_φ(x, t) for q_φ(z_t|x)
        self.forward_net = FLDDNetwork(vocab_size, time_dim, hidden_dim).to(device)
        
        # Reverse network: takes z_t, outputs v_θ(z_t, t) for p_θ(z_s|z_t)
        self.reverse_net = FLDDNetwork(vocab_size, time_dim, hidden_dim).to(device)
        
        # =====================================================================
        # Optimizer (AdamW with lr=2e-4 as in paper Appendix A)
        # =====================================================================
        self.optimizer = optim.AdamW(
            list(self.forward_net.parameters()) + list(self.reverse_net.parameters()),
            lr=2e-4
        )
        
        # =====================================================================
        # Training schedule (Section 3.3 and Appendix A)
        # =====================================================================
        # Paper: warm up for 10^5 steps, then REINFORCE for 100k iterations
        self.warmup_steps = 50000    # Reduced for faster experimentation
        self.reinforce_steps = 50000
        self.total_steps = self.warmup_steps + self.reinforce_steps
        self.current_step = 0
        
        # Temperature schedule: 1.0 → 1e-3 over warmup period
        self.temperature = 1.0
        self.min_temperature = 1e-3
        self.temp_decay_rate = (self.min_temperature / 1.0) ** (1.0 / self.warmup_steps)
        
        # For logging
        self.losses = []
        
    def get_forward_marginals(self, x, t):
        """
        Get q_φ(z_t | x) - the forward marginal distribution.
        
        Boundary conditions (enforced directly, not via interpolation):
        - t = 0: q(z_0|x) = δ(z_0 - x)  (deterministic = original data)
        - t = T: q(z_T|x) = p(z_T)      (uniform prior)
        
        For intermediate t, the network has FULL FREEDOM to learn any distribution.
        This is the key to FLDD: the forward process learns data-dependent
        corruption patterns that make the reverse task easier.
        
        Args:
            x: [batch, H, W] - original discrete data
            t: int - timestep (0 to T)
        
        Returns:
            probs: [batch, H, W, vocab] - probability distribution
        """
        batch_size = x.shape[0]
        
        if t == 0:
            # Boundary: q(z_0|x) = δ(z_0 - x)
            # One-hot encoding of x
            return F.one_hot(x.long(), self.vocab_size).float()
            
        elif t == self.num_timesteps:
            # Boundary: q(z_T|x) = p(z_T) = uniform
            return torch.ones(batch_size, x.shape[1], x.shape[2], 
                            self.vocab_size, device=self.device) / self.vocab_size
            
        else:
            # Intermediate: network has FULL CONTROL (no interpolation!)
            # This allows FLDD to learn non-trivial corruption patterns
            t_tensor = torch.full((batch_size,), t, device=self.device)
            logits = self.forward_net(x, t_tensor)
            return F.softmax(logits, dim=-1)
    
    def get_reverse_distribution(self, z_t, t):
        """
        Get p_θ(z_s | z_t) - the reverse distribution.
        
        This is the standard factorized categorical distribution
        parameterized by the reverse network.
        
        Args:
            z_t: [batch, H, W] discrete OR [batch, vocab, H, W] soft
            t: int - current timestep (sampling z_{t-1} given z_t)
        
        Returns:
            probs: [batch, H, W, vocab] - probability distribution
        """
        batch_size = z_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device)
        
        logits = self.reverse_net(z_t, t_tensor)
        probs = F.softmax(logits, dim=-1)
        
        return probs
    
    def compute_kl_divergence(self, p, q):
        """
        Compute KL(p || q) for the variational objective.
        
        KL(p||q) = Σ p(x) log(p(x)/q(x))
        
        Special handling:
        - When p(x) = 0, the term contributes 0 (not NaN)
        - Add small epsilon to q for numerical stability
        
        Args:
            p, q: [..., vocab] probability distributions
        
        Returns:
            kl: [...] KL divergence (summed over vocab dimension)
        """
        # Avoid log(0) and division by zero
        p_safe = p.clamp(min=1e-10)
        q_safe = q.clamp(min=1e-10)
        
        # KL divergence, masking out p=0 terms
        kl = p * (torch.log(p_safe) - torch.log(q_safe))
        
        # Where p is very small, set contribution to 0
        kl = torch.where(p < 1e-8, torch.zeros_like(kl), kl)
        
        return kl.sum(dim=-1)
    
    def warmup_step(self, x):
        """
        Relaxed optimization step using Concrete distribution (Section 3.3).
        
        During warm-up, we use the Gumbel-Softmax trick to get differentiable
        samples from the categorical distribution. This allows gradient flow
        through the sampling operation via reparameterization.
        
        As training progresses, the temperature decreases, making samples
        more discrete-like.
        """
        batch_size = x.shape[0]
        
        # Sample uniform timestep t ∈ {1, ..., T}
        t = torch.randint(1, self.num_timesteps + 1, (1,)).item()
        s = t - 1  # Previous timestep
        
        # Get forward marginals u_s and u_t
        u_s = self.get_forward_marginals(x, s)
        u_t = self.get_forward_marginals(x, t)
        
        # Sample z_t from Concrete/Gumbel-Softmax distribution
        # This gives differentiable "soft" samples
        dist = RelaxedOneHotCategorical(self.temperature, probs=u_t + 1e-8)
        z_t_soft = dist.rsample()  # [batch, H, W, vocab]
        
        # When s=0, u_s is one-hot (from get_forward_marginals), and the coupling
        # will correctly produce a one-hot posterior
        u_s_given_t = MaximumCoupling.compute_posterior_soft(u_s, u_t, z_t_soft)
        
        # Get reverse distribution p_θ(z_s | z_t)
        # Convert soft samples to [batch, vocab, H, W] for network input
        z_t_input = z_t_soft.permute(0, 3, 1, 2)
        v_s_given_t = self.get_reverse_distribution(z_t_input, t)
        
        # Compute KL divergence loss
        kl_loss = self.compute_kl_divergence(u_s_given_t, v_s_given_t).mean()
        
        # Prior loss: encourage q(z_T|x) ≈ uniform
        if t == self.num_timesteps:
            prior = torch.ones_like(u_t) / self.vocab_size
            prior_loss = self.compute_kl_divergence(u_t, prior).mean()
            kl_loss = kl_loss + 0.1 * prior_loss
        
        return kl_loss
    
    def reinforce_step(self, x):
        """
        REINFORCE training step with discrete samples (Algorithm 1).
        
        Since we cannot backpropagate through discrete samples, we use
        REINFORCE (policy gradient) to estimate gradients.
        
        From Equation 13:
        ∇_φ L = E_{q_φ(z_t|x)} [ ∇_φ log q_φ(z_t|x) × KL(...) ]
        
        The gradient flows through:
        1. Direct path: ∇_φ KL (through posterior parameters)
        2. REINFORCE path: log q_φ(z_t|x) × [KL]_sg (score function gradient)
        """
        batch_size = x.shape[0]
        
        # Sample timestep t ∈ {1, ..., T}
        t = torch.randint(1, self.num_timesteps + 1, (1,)).item()
        s = t - 1
        
        # Get forward marginals
        u_s = self.get_forward_marginals(x, s)
        u_t = self.get_forward_marginals(x, t)
        
        # Sample z_t ~ Cat(u_t) - discrete sample
        dist = Categorical(probs=u_t + 1e-8)
        z_t = dist.sample()  # [batch, H, W]
        
        # Log probability for REINFORCE
        # Sum over all spatial positions since we sample independently
        log_prob_z_t = dist.log_prob(z_t)  # [batch, H, W]
        log_prob_sum = log_prob_z_t.sum(dim=[1, 2])  # [batch]
        
        #compute posterior q(z_s | z_t, x)
        u_s_given_t = MaximumCoupling.compute_posterior(u_s, u_t, z_t)
        
        # Get reverse distribution p_θ(z_s | z_t)
        v_s_given_t = self.get_reverse_distribution(z_t, t)
        
        # Compute KL divergence
        kl_per_position = self.compute_kl_divergence(u_s_given_t, v_s_given_t)  # [batch, H, W]
        kl_per_sample = kl_per_position.sum(dim=[1, 2])  # [batch]
        kl_loss = kl_per_sample.mean()
        
        # REINFORCE gradient (Equation 13)
        # The gradient through sampling is: log_prob × [reward]_sg
        # Here "reward" is negative KL (we want to minimize KL)
        # But since we're minimizing, we use KL directly
        reinforce_loss = (log_prob_sum * kl_per_sample.detach()).mean()
        
        # Total loss
        total_loss = kl_loss + reinforce_loss
        
        # Prior regularization
        if t == self.num_timesteps:
            prior = torch.ones_like(u_t) / self.vocab_size
            prior_loss = self.compute_kl_divergence(u_t, prior).mean()
            total_loss = total_loss + 0.1 * prior_loss
        
        return total_loss
    
    def train_step(self, x):
        """
        Single training step following paper's procedure.
        
        Phase 1 (warm-up): Use Concrete relaxation with temperature annealing
        Phase 2 (REINFORCE): Use discrete samples with policy gradient
        """
        self.optimizer.zero_grad()
        
        if self.current_step < self.warmup_steps:
            # Warm-up phase with relaxed optimization
            loss = self.warmup_step(x)
            
            # Exponential temperature decay
            self.temperature = max(
                self.temperature * self.temp_decay_rate,
                self.min_temperature
            )
        else:
            # REINFORCE phase
            loss = self.reinforce_step(x)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(self.forward_net.parameters()) + list(self.reverse_net.parameters()),
            max_norm=1.0
        )
        
        self.optimizer.step()
        self.current_step += 1
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, num_samples=16, image_size=28):
        """
        Sampling procedure (Algorithm 2).
        
        Start from prior p(z_T) and iteratively apply reverse process
        to generate samples.
        
        Args:
            num_samples: Number of samples to generate
            image_size: Size of generated images
        
        Returns:
            samples: [num_samples, H, W] discrete samples
        """
        self.forward_net.eval()
        self.reverse_net.eval()
        
        # Sample z_T from prior p(z_T) = uniform
        z_t = torch.randint(0, self.vocab_size, (num_samples, image_size, image_size), 
                           device=self.device)
        
        # Reverse diffusion: t = T, T-1, ..., 1
        for t in reversed(range(1, self.num_timesteps + 1)):
            # Get reverse distribution p_θ(z_s | z_t)
            v_s_given_t = self.get_reverse_distribution(z_t, t)
            
            # Sample z_s ~ Cat(v_s|t)
            dist = Categorical(probs=v_s_given_t + 1e-8)
            z_t = dist.sample()
        
        self.forward_net.train()
        self.reverse_net.train()
        
        return z_t
    
    @torch.no_grad()
    def visualize_forward_process(self, x):
        """
        Visualize the learned forward dynamics.
        
        Shows how the forward process corrupts data over time.
        """
        self.forward_net.eval()
        
        batch_size = 1
        x_single = x[:batch_size]
        
        trajectories = [x_single.cpu()]
        
        for t in range(1, self.num_timesteps + 1):
            u_t = self.get_forward_marginals(x_single, t)
            dist = Categorical(probs=u_t + 1e-8)
            z_t = dist.sample()
            trajectories.append(z_t.cpu())
        
        self.forward_net.train()
        
        return trajectories


# =============================================================================
# DATA PREPARATION
# =============================================================================

# The following function is defined in this location, since on MacOS threads are spawned differently as on Linux
# I don't understand the details, but if we defined discretize(x) in some inside scope, the threads that Python
# spawns can't find it
def discretize(x):
    return (x > 0.5).float()

def prepare_data(batch_size=128):
    """
    Prepare binarized MNIST dataset.
    
    The images are binarized with threshold 0.5, resulting in
    vocab_size = 2 (black/white pixels).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(discretize)
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    is_pin_mem_available = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=is_pin_mem_available)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=is_pin_mem_available)
    
    return train_loader, test_loader


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_loader, num_epochs=None):
    """
    Training loop following paper specifications.
    """
    losses = []
    
    # Calculate epochs based on total steps
    steps_per_epoch = len(train_loader)
    if num_epochs is None:
        total_epochs = (model.total_steps + steps_per_epoch - 1) // steps_per_epoch
    else:
        total_epochs = num_epochs
    
    print(f"Training for {model.total_steps} total steps (~{total_epochs} epochs)")
    print(f"Warmup: {model.warmup_steps} steps, REINFORCE: {model.reinforce_steps} steps")
    
    for epoch in range(total_epochs):
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}')
        
        for batch_idx, (data, _) in enumerate(pbar):
            if model.current_step >= model.total_steps:
                break
            
            # Prepare data: [batch, 1, H, W] → [batch, H, W]
            data = data.squeeze(1).to(device)
            
            # Training step
            loss = model.train_step(data)
            epoch_losses.append(loss)
            
            # Update progress bar
            if batch_idx % 20 == 0:
                phase = 'warmup' if model.current_step < model.warmup_steps else 'REINFORCE'
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'temp': f'{model.temperature:.5f}',
                    'phase': phase,
                    'step': f'{model.current_step}/{model.total_steps}'
                })
        
        # Epoch statistics
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}')
        
        # Generate samples periodically
        if (epoch + 1) % 5 == 0 or epoch == 0:
            visualize_samples(model, epoch + 1)
        
        if model.current_step >= model.total_steps:
            break
    
    return losses


def visualize_samples(model, epoch, save_dir='./outputs'):
    """Generate and save sample images."""
    os.makedirs(save_dir, exist_ok=True)
    
    samples = model.sample(num_samples=16)
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].cpu().numpy(), cmap='gray')
        ax.axis('off')
    
    phase = 'warmup' if model.current_step < model.warmup_steps else 'reinforce'
    plt.suptitle(f'FLDD Samples - Epoch {epoch} ({phase}, step {model.current_step})')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/samples_epoch_{epoch}.png', dpi=150)
    plt.close()
    print(f"  → Saved samples to {save_dir}/samples_epoch_{epoch}.png")


def visualize_diffusion_process(model, test_loader, save_dir='./outputs'):
    """Visualize forward and reverse processes."""
    os.makedirs(save_dir, exist_ok=True)
    
    data, _ = next(iter(test_loader))
    x = data[0:1].squeeze(1).to(device)
    
    # Forward process visualization
    forward_traj = model.visualize_forward_process(x)
    
    # Reverse process visualization
    reverse_traj = []
    z_t = torch.randint(0, model.vocab_size, (1, 28, 28), device=device)
    reverse_traj.append(z_t.cpu())
    
    model.reverse_net.eval()
    with torch.no_grad():
        for t in reversed(range(1, model.num_timesteps + 1)):
            v_s_given_t = model.get_reverse_distribution(z_t, t)
            dist = Categorical(probs=v_s_given_t + 1e-8)
            z_t = dist.sample()
            reverse_traj.append(z_t.cpu())
    model.reverse_net.train()
    
    reverse_traj = list(reversed(reverse_traj))
    
    # Plot
    fig, axes = plt.subplots(2, model.num_timesteps + 1, figsize=(3 * (model.num_timesteps + 1), 6))
    
    for t in range(model.num_timesteps + 1):
        axes[0, t].imshow(forward_traj[t].squeeze().numpy(), cmap='gray')
        axes[0, t].set_title(f't={t}')
        axes[0, t].axis('off')
        
        axes[1, t].imshow(reverse_traj[t].squeeze().numpy(), cmap='gray')
        axes[1, t].axis('off')
    
    axes[0, 0].set_ylabel('Forward\n(learned)', rotation=0, size='large', ha='right')
    axes[1, 0].set_ylabel('Reverse\n(generation)', rotation=0, size='large', ha='right')
    
    plt.suptitle(f'FLDD: Learned Forward and Reverse Processes (T={model.num_timesteps})')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/diffusion_process.png', dpi=150)
    plt.close()
    print(f"  → Saved diffusion process visualization to {save_dir}/diffusion_process.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training script."""
    print("=" * 60)
    print("Forward-Learned Discrete Diffusion (FLDD) - Corrected")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs('./outputs', exist_ok=True)
    
    # Prepare data
    train_loader, test_loader = prepare_data(batch_size=128)
    
    # Initialize model
    # Using T=10 as in paper experiments for few-step generation
    model = FLDD(
        vocab_size=2,      # Binary images
        num_timesteps=10,  # Few-step generation
        hidden_dim=128,
        time_dim=128
    )
    
    print(f"\nModel Configuration:")
    print(f"  - Timesteps (T): {model.num_timesteps}")
    print(f"  - Vocabulary size: {model.vocab_size}")
    print(f"  - Warmup steps: {model.warmup_steps:,}")
    print(f"  - REINFORCE steps: {model.reinforce_steps:,}")
    print(f"  - Total steps: {model.total_steps:,}")
    print(f"  - Learning rate: 2e-4 (AdamW)")
    print(f"  - Temperature: {model.temperature} → {model.min_temperature}")
    
    # Count parameters
    forward_params = sum(p.numel() for p in model.forward_net.parameters())
    reverse_params = sum(p.numel() for p in model.reverse_net.parameters())
    print(f"  - Forward network params: {forward_params:,}")
    print(f"  - Reverse network params: {reverse_params:,}")
    print("=" * 60)
    
    # Train model
    losses = train_model(model, train_loader)
    
    # Plot training curve
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    warmup_epochs = model.warmup_steps / len(train_loader)
    plt.axvline(x=warmup_epochs, color='r', linestyle='--', label='Warmup → REINFORCE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FLDD Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./outputs/training_curve.png', dpi=150)
    plt.close()
    
    # Visualize diffusion process
    visualize_diffusion_process(model, test_loader)
    
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
