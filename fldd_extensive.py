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
import random
import json
from datetime import datetime
import wandb

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

SEED = 42


def set_seed(seed, deterministic=True):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # Faster but non-deterministic
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

# =============================================================================
# CONFIGURATION
# =============================================================================

PRESET = "paper"  # Changed default to show the new baseline

VOCAB_SIZE = 2
NUM_TIMESTEPS = 4
HIDDEN_DIM = 128
TIME_DIM = 128

PRETRAIN_STEPS = 40
WARMUP_STEPS = 1000
REINFORCE_STEPS = 1000
LEARNING_RATE = 2e-4

INITIAL_TEMPERATURE = 1.0
MIN_TEMPERATURE = 0.1
TEMPERATURE_SCHEDULE = "exponential"

# Forward modes:
# "80_20" - Original working (80% network + 20% interpolation)
# "paper" - Paper mode with boundary enforcement
# "standard" - FIXED forward (no learning) - standard diffusion baseline
# "paper_raw" - Raw paper mode (network only, no boundary enforcement)
FORWARD_MODE = "standard"

# Whether to train the forward network
TRAIN_FORWARD = True  # Set False for standard diffusion baseline

COUPLING_METHOD = "maximum_coupling"
SINKHORN_EPSILON = 0.1
SINKHORN_ITERATIONS = 20
LEARN_COST_MATRIX = False
SINKHORN_EPS_SCHEDULE = [1.0, 0.5, 0.1]

BASELINE_METHOD = "simple"
RLOO_K_SAMPLES = 4

GRADIENT_ESTIMATOR = "reinforce"
RELAXATION_METHOD = "concrete"

USE_PRETRAINING = True
USE_RECONSTRUCTION_LOSS = True
USE_AUXILIARY_CE_LOSS = False
AUXILIARY_CE_LAMBDA = 0.001
USE_SCORE_ENTROPY_LOSS = False

USE_TC_REGULARIZATION = False
TC_LAMBDA = 0.1
TC_METHOD = "simple"

# Boundary loss - enforces proper forward process behavior
USE_BOUNDARY_LOSS = False
BOUNDARY_LAMBDA = 1.0  # Weight for boundary loss
ENTROPY_MONO_LAMBDA = 0.5  # Weight for entropy monotonicity

SAMPLING_METHOD = "standard"
PC_CORRECTOR_STEPS = 1
NOISE_SCHEDULE = "fixed"

USE_EMA = False
EMA_DECAY = 0.9999
EMA_START_STEP = 100
USE_GRADIENT_CLIPPING = True
GRADIENT_CLIP_NORM = 1.0
USE_MIN_SNR_WEIGHTING = False
MIN_SNR_GAMMA = 5.0

# Performance options
USE_AMP = False  # Mixed precision - disable by default (can cause NaN on some setups)
USE_COMPILE = False  # torch.compile JIT (PyTorch 2.0+, can be slow to compile)
DETERMINISTIC = False  # Set True for reproducibility, False for speed
PRETRAIN_NOISE_LEVELS = 2  # Reduce from 4 for faster pretraining
NETWORK_SIZE = "normal"  # "normal" (128), "small" (64), "tiny" (32) - smaller = faster

LOG_EVERY = 5
VIS_EVERY = 5

# =============================================================================
# PRESETS
# =============================================================================

def apply_preset():
    global MIN_TEMPERATURE, FORWARD_MODE, COUPLING_METHOD, BASELINE_METHOD
    global RELAXATION_METHOD, USE_BOUNDARY_LOSS, BOUNDARY_LAMBDA, ENTROPY_MONO_LAMBDA
    global USE_TC_REGULARIZATION, TC_LAMBDA, USE_AUXILIARY_CE_LOSS, AUXILIARY_CE_LAMBDA
    global TEMPERATURE_SCHEDULE, TC_METHOD, SAMPLING_METHOD, USE_PRETRAINING, USE_RECONSTRUCTION_LOSS
    global RLOO_K_SAMPLES, SINKHORN_EPS_SCHEDULE, USE_EMA, USE_MIN_SNR_WEIGHTING, TRAIN_FORWARD
    global PRETRAIN_STEPS, WARMUP_STEPS, REINFORCE_STEPS, USE_AMP, USE_COMPILE, DETERMINISTIC, PRETRAIN_NOISE_LEVELS, NETWORK_SIZE

    if PRESET == "standard":
        # Standard diffusion baseline - D3PM style simple training
        # Fixed forward process, simple KL training, no phases
        MIN_TEMPERATURE = 0.1
        FORWARD_MODE = "standard"
        TRAIN_FORWARD = False
        COUPLING_METHOD = "maximum_coupling"
        BASELINE_METHOD = "simple"
        RELAXATION_METHOD = "concrete"
        USE_TC_REGULARIZATION = False
        USE_AUXILIARY_CE_LOSS = False
        USE_PRETRAINING = False
        USE_RECONSTRUCTION_LOSS = False
        TEMPERATURE_SCHEDULE = "exponential"
        SAMPLING_METHOD = "standard"
        USE_BOUNDARY_LOSS = False
        PRETRAIN_STEPS = 0
        WARMUP_STEPS = 0
        REINFORCE_STEPS = 2040

    elif PRESET == "paper":
        # Paper mode WITH proper boundary enforcement
        MIN_TEMPERATURE = 1e-3
        FORWARD_MODE = "paper"  # Uses boundary-enforced learned forward
        TRAIN_FORWARD = True
        COUPLING_METHOD = "maximum_coupling"
        BASELINE_METHOD = "simple"
        RELAXATION_METHOD = "concrete"
        USE_TC_REGULARIZATION = False
        USE_AUXILIARY_CE_LOSS = False
        TEMPERATURE_SCHEDULE = "exponential"
        SAMPLING_METHOD = "standard"
        USE_BOUNDARY_LOSS = True  # Critical for learned forward

    elif PRESET == "project":
        # Project extensions: Sinkhorn, TC regularization, binary relaxation
        MIN_TEMPERATURE = 1e-3
        FORWARD_MODE = "paper"
        TRAIN_FORWARD = True
        COUPLING_METHOD = "sinkhorn"
        BASELINE_METHOD = "optimal"
        RELAXATION_METHOD = "binary"
        USE_TC_REGULARIZATION = True
        TC_LAMBDA = 0.1
        TC_METHOD = "minibatch_weighted"
        USE_AUXILIARY_CE_LOSS = False
        TEMPERATURE_SCHEDULE = "exponential"
        SAMPLING_METHOD = "standard"
        USE_BOUNDARY_LOSS = True

    elif PRESET == "manual":
        # Manual mode: use global variable settings as-is (no changes)
        # Configure the global variables above to experiment with different settings
        pass

    else:
        raise ValueError(f"Unknown preset: {PRESET}. Use 'standard', 'paper', 'project', or 'manual'.")

    # Detect simple training mode: standard preset uses D3PM-style simple KL
    simple_training = (PRESET == "standard" or
                       (PRETRAIN_STEPS == 0 and WARMUP_STEPS == 0 and FORWARD_MODE == "standard"))

    return {
        'preset': PRESET,
        'seed': SEED,
        'pretrain': PRETRAIN_STEPS,
        'warmup': WARMUP_STEPS,
        'reinforce': REINFORCE_STEPS,
        'min_temp': MIN_TEMPERATURE,
        'temp_schedule': TEMPERATURE_SCHEDULE,
        'forward_mode': FORWARD_MODE,
        'train_forward': TRAIN_FORWARD,
        'coupling': COUPLING_METHOD,
        'baseline': BASELINE_METHOD,
        'rloo_k': RLOO_K_SAMPLES,
        'relaxation': RELAXATION_METHOD,
        'pretraining': USE_PRETRAINING,
        'recon_loss': USE_RECONSTRUCTION_LOSS,
        'aux_ce_loss': USE_AUXILIARY_CE_LOSS,
        'aux_ce_lambda': AUXILIARY_CE_LAMBDA,
        'score_entropy': USE_SCORE_ENTROPY_LOSS,
        'tc_reg': USE_TC_REGULARIZATION,
        'tc_lambda': TC_LAMBDA,
        'tc_method': TC_METHOD,
        'boundary_loss': USE_BOUNDARY_LOSS,
        'boundary_lambda': BOUNDARY_LAMBDA,
        'entropy_mono_lambda': ENTROPY_MONO_LAMBDA,
        'sampling': SAMPLING_METHOD,
        'pc_steps': PC_CORRECTOR_STEPS,
        'noise_schedule': NOISE_SCHEDULE,
        'use_ema': USE_EMA,
        'ema_decay': EMA_DECAY,
        'min_snr': USE_MIN_SNR_WEIGHTING,
        'min_snr_gamma': MIN_SNR_GAMMA,
        'sinkhorn_eps_schedule': SINKHORN_EPS_SCHEDULE,
        # Performance options
        'use_amp': USE_AMP,
        'use_compile': USE_COMPILE,
        'deterministic': DETERMINISTIC,
        'pretrain_noise_levels': PRETRAIN_NOISE_LEVELS,
        'network_size': NETWORK_SIZE,
        # D3PM-style simple training (no phases)
        'simple_training': simple_training,
    }

# =============================================================================
# ENVIRONMENT
# =============================================================================

class Env:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.batch_size = 256
            self.name = "CUDA"
            self.num_workers = 4  # Parallel data loading
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.batch_size = 128
            self.name = "MPS"
            self.num_workers = 2
        else:
            self.device = torch.device('cpu')
            self.batch_size = 64
            self.name = "CPU"
            self.num_workers = 2
        self.pin_memory = self.device.type == 'cuda'

env = Env()
device = env.device


# =============================================================================
# COUPLING METHODS 
# =============================================================================

class MaxCoupling:
    @staticmethod
    def posterior(u_s, u_t, z_t):

        eps = 1e-8
        u_s = torch.clamp(u_s, min=eps)
        u_t = torch.clamp(u_t, min=eps)
        u_s = u_s / u_s.sum(-1, keepdim=True)
        u_t = u_t / u_t.sum(-1, keepdim=True)

        idx = z_t.long().unsqueeze(-1)
        u_t_k = torch.gather(u_t, -1, idx).squeeze(-1)
        u_s_k = torch.gather(u_s, -1, idx).squeeze(-1)

        stay = torch.minimum(u_s_k, u_t_k) / (u_t_k + eps)
        move = torch.clamp((u_t_k - u_s_k) / (u_t_k + eps), min=0)

        deficit = torch.clamp(u_s - u_t, min=0)
        m = deficit / (deficit.sum(-1, keepdim=True) + eps)

        post = move.unsqueeze(-1) * m
        mask = torch.zeros_like(post).scatter_(-1, idx, 1.0)
        post = post * (1 - mask) + stay.unsqueeze(-1) * mask
        post = torch.clamp(post, min=eps)
        result = post / post.sum(-1, keepdim=True)
        return result
    
    @staticmethod
    def posterior_soft(u_s, u_t, z_soft):
        B, H, W, K = u_s.shape
        eps = 1e-8
        tau = 0.1  # Temperature for smooth approximations
        u_s = torch.clamp(u_s, min=eps)
        u_t = torch.clamp(u_t, min=eps)
        u_s = u_s / u_s.sum(-1, keepdim=True)
        u_t = u_t / u_t.sum(-1, keepdim=True)
        u_t_k_soft = (z_soft * u_t).sum(-1, keepdim=True) 
        u_s_k_soft = (z_soft * u_s).sum(-1, keepdim=True) 

        def smooth_min(a, b, tau=tau):
            stacked = torch.stack([-a/tau, -b/tau], dim=-1)
            return -tau * torch.logsumexp(stacked, dim=-1)

        def smooth_relu(x, tau=tau):
            return tau * F.softplus(x / tau)

        stay_soft = smooth_min(u_s_k_soft.squeeze(-1), u_t_k_soft.squeeze(-1)) / (u_t_k_soft.squeeze(-1) + eps)
        move_soft = smooth_relu(u_t_k_soft.squeeze(-1) - u_s_k_soft.squeeze(-1)) / (u_t_k_soft.squeeze(-1) + eps)
        deficit = smooth_relu(u_s - u_t)
        m = deficit / (deficit.sum(-1, keepdim=True) + eps)
        post = move_soft.unsqueeze(-1) * m
        post = post + stay_soft.unsqueeze(-1) * z_soft
        post = torch.clamp(post, min=eps)
        result = post / post.sum(-1, keepdim=True)
        return result


class Sinkhorn:
    @staticmethod
    def solve(a, b, C, eps=0.1, iters=20):
        a = torch.clamp(a, min=1e-8)
        b = torch.clamp(b, min=1e-8)
        a = a / a.sum(-1, keepdim=True)
        b = b / b.sum(-1, keepdim=True)
        log_K = -C / eps
        f, g = torch.zeros_like(a), torch.zeros_like(b)
        for i in range(iters):
            f = torch.log(a) - torch.logsumexp(log_K.unsqueeze(0) + g.unsqueeze(-2), -1)
            g = torch.log(b) - torch.logsumexp(log_K.T.unsqueeze(0) + f.unsqueeze(-2), -1)
        result = torch.clamp(torch.exp(f.unsqueeze(-1) + log_K.unsqueeze(0) + g.unsqueeze(-2)), min=1e-10)
        return result
    
    @staticmethod
    def posterior(u_s, u_t, z_t, C, eps=0.1, iters=20):
        B, H, W, K = u_s.shape
        P = Sinkhorn.solve(u_s.reshape(-1, K), u_t.reshape(-1, K), C, eps, iters).reshape(B, H, W, K, K)
        idx = z_t.long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, K, 1)
        P_col = torch.gather(P, -1, idx).squeeze(-1)
        u_t_z = torch.gather(u_t, -1, z_t.long().unsqueeze(-1)).squeeze(-1)
        post = P_col / (u_t_z.unsqueeze(-1) + 1e-8)
        post = torch.clamp(post, min=1e-8)
        result = post / post.sum(-1, keepdim=True)
        return result

    @staticmethod
    def posterior_soft(u_s, u_t, z_soft, C, eps=0.1, iters=20):
        B, H, W, K = u_s.shape
        P = Sinkhorn.solve(u_s.reshape(-1, K), u_t.reshape(-1, K), C, eps, iters).reshape(B, H, W, K, K)
        cond = P / (u_t.unsqueeze(-2) + 1e-8)
        post = torch.einsum('...ik,...k->...i', cond, z_soft / (z_soft.sum(-1, keepdim=True) + 1e-8))
        post = torch.clamp(post, min=1e-8)
        result = post / post.sum(-1, keepdim=True)
        return result


class LogSinkhorn:
    @staticmethod
    def solve(a, b, C, eps_schedule=[1.0, 0.5, 0.1], iters_per_eps=20):
        a = torch.clamp(a, min=1e-8)
        b = torch.clamp(b, min=1e-8)
        a = a / a.sum(-1, keepdim=True)
        b = b / b.sum(-1, keepdim=True)

        log_a = torch.log(a)
        log_b = torch.log(b)
        f = torch.zeros_like(a)
        g = torch.zeros_like(b)

        for eps in eps_schedule:
            log_K = -C / eps
            for _ in range(iters_per_eps):
                f_new = eps * (log_a - torch.logsumexp((g.unsqueeze(-2) + log_K.T.unsqueeze(0)), dim=-1))
                g_new = eps * (log_b - torch.logsumexp((f_new.unsqueeze(-1) + log_K.unsqueeze(0)), dim=-2))
                f, g = f_new, g_new

        log_P = (f.unsqueeze(-1) + log_K.unsqueeze(0) + g.unsqueeze(-2))
        P = torch.clamp(torch.exp(log_P), min=1e-10)
        return P
    
    @staticmethod
    def posterior(u_s, u_t, z_t, C, eps_schedule=[1.0, 0.5, 0.1], iters_per_eps=20):
        B, H, W, K = u_s.shape
        P = LogSinkhorn.solve(u_s.reshape(-1, K), u_t.reshape(-1, K), C, eps_schedule, iters_per_eps).reshape(B, H, W, K, K)
        idx = z_t.long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, K, 1)
        P_col = torch.gather(P, -1, idx).squeeze(-1)
        u_t_z = torch.gather(u_t, -1, z_t.long().unsqueeze(-1)).squeeze(-1)
        post = P_col / (u_t_z.unsqueeze(-1) + 1e-8)
        post = torch.clamp(post, min=1e-8)
        result = post / post.sum(-1, keepdim=True)
        return result

    @staticmethod
    def posterior_soft(u_s, u_t, z_soft, C, eps_schedule=[1.0, 0.5, 0.1], iters_per_eps=20):
        B, H, W, K = u_s.shape
        P = LogSinkhorn.solve(u_s.reshape(-1, K), u_t.reshape(-1, K), C, eps_schedule, iters_per_eps).reshape(B, H, W, K, K)
        cond = P / (u_t.unsqueeze(-2) + 1e-8)
        post = torch.einsum('...ik,...k->...i', cond, z_soft / (z_soft.sum(-1, keepdim=True) + 1e-8))
        post = torch.clamp(post, min=1e-8)
        result = post / post.sum(-1, keepdim=True)
        return result


# =============================================================================
# KL DECOMPOSITION & TC ESTIMATION
# =============================================================================

class KLDecomp:
    @staticmethod
    def tc_simple(posterior):
        B, H, W, K = posterior.shape
        D = H * W
        post_flat = posterior.reshape(B, D, K)
        q_agg = post_flat.mean(0)  
        log_post = torch.log(post_flat + 1e-10)
        log_agg = torch.log(q_agg + 1e-10)
        tc = (post_flat * (log_post - log_agg)).sum(-1).sum(-1).mean() / D
        return tc
    
    @staticmethod
    def tc_minibatch_weighted(z_s_samples, forward_marginals_s, batch_size):
        B, H, W, K = forward_marginals_s.shape
        D = H * W
        z_flat = z_s_samples.reshape(B, D) 
        u_flat = forward_marginals_s.reshape(B, D, K)  
        z_expanded = z_flat.unsqueeze(1).expand(B, B, D)  
        log_probs = torch.log(torch.gather(u_flat.unsqueeze(0).expand(B, B, D, K), -1,   z_expanded.unsqueeze(-1).long()).squeeze(-1) + 1e-10) 
        log_q_joint = torch.logsumexp(log_probs.sum(-1), dim=1) - math.log(B)  # [B]
        log_q_marginal = torch.logsumexp(log_probs, dim=1) - math.log(B)  # [B, D]
        log_q_prod = log_q_marginal.sum(-1)  # [B]
        tc = torch.clamp((log_q_joint - log_q_prod).mean() / D, min=0.0)
        return tc
    
    @staticmethod
    def factorized_kl(post, rev):
        B, H, W, K = post.shape
        num_pixels = H * W
        p = torch.clamp(post, min=1e-8)
        q = torch.clamp(rev, min=1e-8)
        p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
        kl = p * (torch.log(p) - torch.log(q))
        kl = torch.where(p < 1e-7, torch.zeros_like(kl), kl)
        kl = torch.clamp(kl, min=0, max=100)
        result = kl.sum(-1).sum([1, 2]).mean() / num_pixels
        return result

    @staticmethod
    def kl_per_sample(post, rev):
        B, H, W, K = post.shape
        num_pixels = H * W
        p = torch.clamp(post, min=1e-8)
        q = torch.clamp(rev, min=1e-8)
        p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
        kl = p * (torch.log(p) - torch.log(q))
        kl = torch.where(p < 1e-7, torch.zeros_like(kl), kl)
        kl = torch.clamp(kl, min=0, max=100)
        result = kl.sum(-1).sum([1, 2]) / num_pixels
        return result


# =============================================================================
# BASELINES FOR VARIANCE REDUCTION
# =============================================================================

class SimpleBaseline:
    def __init__(self):
        self.v = 0.0

    def update(self, lp, kl):
        old_v = self.v
        self.v = 0.99 * self.v + 0.01 * kl.mean().item()
        return self.v


class OptimalBaseline:
    def __init__(self):
        self.n, self.d = 0.0, 1e-8

    def update(self, lp, kl):
        w = (lp - lp.mean())**2 + 1e-8
        self.n = 0.99 * self.n + 0.01 * (w * kl).sum().item()
        self.d = 0.99 * self.d + 0.01 * w.sum().item()
        result = self.n / self.d
        return result


class RLOOBaseline:
    def __init__(self, k_samples=4, normalize=False):
        self.k = k_samples
        self.normalize = normalize

    def compute_gradient(self, rewards, log_probs):
        reward_sum = rewards.sum(dim=-1, keepdim=True)
        baseline = (reward_sum - rewards) / (self.k - 1)
        advantages = rewards - baseline
        if self.normalize:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std
        loss = (advantages.detach() * log_probs).mean()
        return loss


# =============================================================================
# RELAXATION METHODS
# =============================================================================

class BinaryReparam:
    @staticmethod
    def sample(p, t=0.1):

        eps = 1e-6
        # Clamp and normalize to ensure valid probabilities
        p_clamped = torch.clamp(p, min=eps, max=1-eps)
        p_clamped = p_clamped / p_clamped.sum(-1, keepdim=True)

        # Uses p[..., 0] and p[..., 1] for proper gradient flow
        p0, p1 = p_clamped[..., 0], p_clamped[..., 1]
        log_odds = torch.log(p1) - torch.log(p0)
        u = torch.rand_like(log_odds)
        u = torch.clamp(u, min=eps, max=1-eps)
        logistic_sample = torch.log(u) - torch.log(1 - u)
        s = torch.sigmoid((log_odds + logistic_sample) / t)
        soft = torch.stack([1-s, s], -1)
        result = soft - p_clamped.detach() + p_clamped
        return result


class StraightThroughGumbel:
    @staticmethod
    def sample(logits, temperature=1.0):
        gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        soft = F.softmax((logits + gumbels) / temperature, dim=-1)
        hard = F.one_hot(soft.argmax(dim=-1), num_classes=logits.shape[-1]).float()
        result = hard - soft.detach() + soft
        return result


class StraightThroughEstimator:
    @staticmethod
    def sample(probs):
        hard = F.one_hot(probs.argmax(dim=-1), num_classes=probs.shape[-1]).float()
        result = hard - probs.detach() + probs
        return result


# =============================================================================
# EMA MODEL
# =============================================================================

class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# NETWORK
# =============================================================================

class SinEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
    
    def forward(self, t):
        h = self.d // 2
        e = torch.exp(torch.arange(h, device=t.device) * -math.log(10000) / (h - 1))
        e = t[:, None] * e[None, :]
        return torch.cat([e.sin(), e.cos()], -1)


class ResBlk(nn.Module):
    def __init__(self, ci, co, td):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, padding=1)
        self.c2 = nn.Conv2d(co, co, 3, padding=1)
        self.n1 = nn.GroupNorm(8, co)
        self.n2 = nn.GroupNorm(8, co)
        self.tp = nn.Linear(td, co * 2)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
    
    def forward(self, x, t):
        h = F.gelu(self.n1(self.c1(x)))
        s, sh = self.tp(t)[:, :, None, None].chunk(2, 1)
        h = F.gelu(self.n2(self.c2(h * (1 + s) + sh)))
        return h + self.skip(x)


class Net(nn.Module):
    def __init__(self, K, td, hd):
        super().__init__()
        self.K = K
        self.te = nn.Sequential(SinEmb(td), nn.Linear(td, td * 4), nn.GELU(), nn.Linear(td * 4, td))
        self.inc = nn.Conv2d(K, hd, 3, padding=1)
        self.e1, self.d1 = ResBlk(hd, hd, td), nn.Conv2d(hd, hd, 4, 2, 1)
        self.e2, self.d2 = ResBlk(hd, hd * 2, td), nn.Conv2d(hd * 2, hd * 2, 4, 2, 1)
        self.m1, self.m2 = ResBlk(hd * 2, hd * 2, td), ResBlk(hd * 2, hd * 2, td)
        self.u2, self.dc2 = nn.ConvTranspose2d(hd * 2, hd * 2, 4, 2, 1), ResBlk(hd * 4, hd, td)
        self.u1, self.dc1 = nn.ConvTranspose2d(hd, hd, 4, 2, 1), ResBlk(hd * 2, hd, td)
        self.out = nn.Sequential(nn.GroupNorm(8, hd), nn.GELU(), nn.Conv2d(hd, K, 3, padding=1))
        num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x, t):
        if x.dim() == 3:
            x = F.one_hot(x.long(), self.K).float().permute(0, 3, 1, 2)
        te = self.te(t.float())
        h = self.inc(x)
        h1 = self.e1(h, te)
        h = self.d1(h1)
        h2 = self.e2(h, te)
        h = self.d2(h2)
        h = self.m2(self.m1(h, te), te)
        h = self.dc2(torch.cat([self.u2(h), h2], 1), te)
        h = self.dc1(torch.cat([self.u1(h), h1], 1), te)
        out = self.out(h).permute(0, 2, 3, 1)
        return out


# =============================================================================
# FLDD MODEL
# =============================================================================

class FLDD:
    def __init__(self, cfg):

        self.cfg = cfg
        net_size = cfg.get('network_size', 'normal')
        hidden_dim = {'normal': 128, 'small': 64, 'tiny': 32}.get(net_size, 128)
        time_dim = {'normal': 128, 'small': 64, 'tiny': 32}.get(net_size, 128)

        self.fwd = Net(VOCAB_SIZE, time_dim, hidden_dim).to(device)
        self.rev = Net(VOCAB_SIZE, time_dim, hidden_dim).to(device)

        # Optional torch.compile for speedup (PyTorch 2.0+)
        if cfg.get('use_compile', False) and hasattr(torch, 'compile'):
            print("Compiling networks with torch.compile (this may take a minute)...")
            self.fwd = torch.compile(self.fwd)
            self.rev = torch.compile(self.rev)

        # AMP scaler for mixed precision (with conservative settings to avoid NaN)
        self.use_amp = cfg.get('use_amp', False) and device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(
                'cuda',
                init_scale=2**10,  # Lower initial scale (default is 2**16)
                growth_interval=2000,  # Less aggressive scaling
            )
        else:
            self.scaler = None

        # Cost matrix for Sinkhorn methods
        self.C = None
        coupling = cfg.get('coupling', 'maximum_coupling')
        if coupling in ["sinkhorn", "log_sinkhorn", "sinkhorn_divergence"]:
            c = torch.ones(VOCAB_SIZE, VOCAB_SIZE, device=device) - torch.eye(VOCAB_SIZE, device=device)
            self.C = nn.Parameter(c) if LEARN_COST_MATRIX else c

        if cfg.get('train_forward', True):
            params = list(self.fwd.parameters()) + list(self.rev.parameters())
        else:
            # Standard diffusion baseline: only train reverse
            params = list(self.rev.parameters())
            # Freeze forward network
            for p in self.fwd.parameters():
                p.requires_grad = False

        if coupling in ["sinkhorn", "log_sinkhorn", "sinkhorn_divergence"] and LEARN_COST_MATRIX:
            params.append(self.C)

        self.opt = optim.AdamW(params, lr=LEARNING_RATE)

        # Initialize baseline
        baseline = cfg.get('baseline', 'simple')
        if baseline == "optimal":
            self.baseline = OptimalBaseline()
        elif baseline in ["rloo", "rloo_normalized"]:
            self.baseline = RLOOBaseline(k_samples=cfg.get('rloo_k', 4), normalize=(baseline == "rloo_normalized"))
        else:
            self.baseline = SimpleBaseline()

        # EMA
        use_ema = cfg.get('use_ema', False)
        ema_decay = cfg.get('ema_decay', 0.9999)
        self.ema_fwd = EMA(self.fwd, ema_decay) if use_ema and cfg.get('train_forward', True) else None
        self.ema_rev = EMA(self.rev, ema_decay) if use_ema else None

        self.step = 0
        self.temp = INITIAL_TEMPERATURE

        if cfg.get('temp_schedule', 'exponential') == "exponential":
            self.decay = (cfg.get('min_temp', 0.1) / INITIAL_TEMPERATURE) ** (1.0 / max(cfg.get('warmup', 5000), 1))
        else:
            self.decay = None
        
    
    @property
    def total(self):
        return self.cfg.get('pretrain', 1000) + self.cfg.get('warmup', 5000) + self.cfg.get('reinforce', 5000)

    @property
    def phase(self):
        # Simple training mode has no phases
        if self.cfg.get('simple_training', False):
            return 'simple'
        pretrain = self.cfg.get('pretrain', 1000)
        warmup = self.cfg.get('warmup', 5000)
        if self.step < pretrain:
            return 'pre'
        if self.step < pretrain + warmup:
            return 'warm'
        return 'RL'

    def get_temperature(self):
        min_temp = self.cfg.get('min_temp', 0.1)
        if self.phase != 'warm':
            return min_temp
        pretrain = self.cfg.get('pretrain', 1000)
        warmup = self.cfg.get('warmup', 5000)
        warmup_progress = (self.step - pretrain) / max(warmup, 1)
        temp_schedule = self.cfg.get('temp_schedule', 'exponential')
        if temp_schedule == "exponential":
            return max(self.temp, min_temp)
        elif temp_schedule == "cosine":
            return min_temp + 0.5 * (INITIAL_TEMPERATURE - min_temp) * (1 + math.cos(math.pi * warmup_progress))
        elif temp_schedule == "linear":
            return INITIAL_TEMPERATURE - warmup_progress * (INITIAL_TEMPERATURE - min_temp)
        else:
            return self.temp
    
    def fwd_marg(self, x, t):
        B = x.shape[0]
        x_oh = F.one_hot(x.long(), VOCAB_SIZE).float()
        uni = torch.ones(B, x.shape[1], x.shape[2], VOCAB_SIZE, device=device) / VOCAB_SIZE

        # Boundary conditions (same for all modes)
        if t == 0:
            return x_oh
        if t == NUM_TIMESTEPS:
            return uni

        a = t / NUM_TIMESTEPS  # Interpolation factor

        # === STANDARD MODE: Fixed forward (no learning) ===
        forward_mode = self.cfg.get('forward_mode', 'paper')
        if forward_mode == "standard":
            # Simple linear interpolation - D3PM style
            p = (1 - a) * x_oh + a * uni
            p = torch.clamp(p, min=1e-8)
            result = p / p.sum(-1, keepdim=True)
            return result

        # === LEARNED FORWARD MODES ===
        tt = torch.full((B,), t, device=device)
        logits = torch.clamp(self.fwd(x, tt), -20, 20)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        net_p = F.softmax(logits, -1)

        if forward_mode == "80_20":
            # 80% network + 20% interpolation
            interp = (1 - a) * x_oh + a * uni
            p = 0.8 * net_p + 0.2 * interp

        elif forward_mode == "paper":
            # with BOUNDED network influence to prevent forward collapse
            # The network can adjust the linear schedule, but only by a limited amount
            # This architecturally prevents the degenerate "instant corruption" solution
            base = (1 - a) * x_oh + a * uni
            lambda_max = self.cfg.get('forward_lambda_max', 0.3)
            scale = lambda_max * 4.0 * a * (1 - a) 
            p = (1 - scale) * base + scale * net_p
            p = torch.clamp(p, min=1e-6)

        elif forward_mode == "constrained":
            # Hard boundary interpolation to prevent forward collapse
            linear_base = (1 - a) * x_oh + a * uni
            lambda_blend = self.cfg.get('constrained_lambda', 0.3)
            network_target = (1 - a) * net_p + a * uni
            p = (1 - lambda_blend) * linear_base + lambda_blend * network_target
            p = torch.clamp(p, min=1e-6)

        elif forward_mode == "paper_raw":
            p = net_p

        else:
            raise ValueError(f"Unknown forward mode: {forward_mode}")

        p = torch.clamp(p, min=1e-8)
        result = p / p.sum(-1, keepdim=True)
        return result

    def fwd_marg_batched(self, x, t, s):
        B = x.shape[0]
        x_oh = F.one_hot(x.long(), VOCAB_SIZE).float()
        uni = torch.ones(B, x.shape[1], x.shape[2], VOCAB_SIZE, device=device) / VOCAB_SIZE

        # Handle boundary conditions
        if s == 0:
            u_s = x_oh
        elif s == NUM_TIMESTEPS:
            u_s = uni
        else:
            u_s = None  # Will compute via network

        if t == 0:
            u_t = x_oh
        elif t == NUM_TIMESTEPS:
            u_t = uni
        else:
            u_t = None  # Will compute via network

        # If both are boundary conditions, return early
        if u_t is not None and u_s is not None:
            return u_t, u_s

        # === STANDARD MODE: No network needed ===
        forward_mode = self.cfg.get('forward_mode', 'paper')
        if forward_mode == "standard":
            if u_t is None:
                a_t = t / NUM_TIMESTEPS
                p_t = (1 - a_t) * x_oh + a_t * uni
                u_t = torch.clamp(p_t, min=1e-8)
                u_t = u_t / u_t.sum(-1, keepdim=True)
            if u_s is None:
                a_s = s / NUM_TIMESTEPS
                p_s = (1 - a_s) * x_oh + a_s * uni
                u_s = torch.clamp(p_s, min=1e-8)
                u_s = u_s / u_s.sum(-1, keepdim=True)
            return u_t, u_s

        # === LEARNED FORWARD: Batch both timesteps in one forward pass ===
        timesteps_to_compute = []
        if u_t is None:
            timesteps_to_compute.append(t)
        if u_s is None:
            timesteps_to_compute.append(s)

        if len(timesteps_to_compute) == 2:
            # Batch: duplicate x and process both timesteps together
            # Ensure x is long for one-hot encoding in network
            x_batched = torch.cat([x, x], dim=0)  # [2B, H, W]
            tt_batched = torch.cat([
                torch.full((B,), timesteps_to_compute[0], device=device),
                torch.full((B,), timesteps_to_compute[1], device=device)
            ])
            logits_batched = torch.clamp(self.fwd(x_batched, tt_batched), -20, 20)
            # Replace any NaN/Inf with 0 before softmax
            logits_batched = torch.nan_to_num(logits_batched, nan=0.0, posinf=20.0, neginf=-20.0)
            net_p_batched = F.softmax(logits_batched, -1)
            net_p_t, net_p_s = net_p_batched[:B], net_p_batched[B:]
        elif len(timesteps_to_compute) == 1:
            tt = torch.full((B,), timesteps_to_compute[0], device=device)
            logits = torch.clamp(self.fwd(x, tt), -20, 20)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            net_p_single = F.softmax(logits, -1)
            if u_t is None:
                net_p_t = net_p_single
                net_p_s = None
            else:
                net_p_t = None
                net_p_s = net_p_single
        else:
            net_p_t, net_p_s = None, None

        # Apply forward mode transformations
        def apply_mode(net_p, timestep):
            a = timestep / NUM_TIMESTEPS
            if forward_mode == "80_20":
                interp = (1 - a) * x_oh + a * uni
                p = 0.8 * net_p + 0.2 * interp
            elif forward_mode == "paper":
                base = (1 - a) * x_oh + a * uni
                lambda_max = self.cfg.get('forward_lambda_max', 0.3)
                scale = lambda_max * 4.0 * a * (1 - a)  # Capped to prevent collapse
                p = (1 - scale) * base + scale * net_p
                p = torch.clamp(p, min=1e-6)
            elif forward_mode == "paper_raw":
                p = net_p
            else:
                raise ValueError(f"Unknown forward mode: {forward_mode}")
            p = torch.clamp(p, min=1e-8)
            return p / p.sum(-1, keepdim=True)

        if u_t is None:
            u_t = apply_mode(net_p_t, t)
        if u_s is None:
            u_s = apply_mode(net_p_s, s)

        return u_t, u_s
    
    def rev_dist(self, z, t):
        B = z.shape[0]
        tt = torch.full((B,), t, device=device)
        logits = torch.clamp(self.rev(z, tt), -20, 20)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        p = torch.clamp(F.softmax(logits, -1), min=1e-8)
        result = p / p.sum(-1, keepdim=True)
        return result
    
    def post(self, u_s, u_t, z):
        coupling = self.cfg.get('coupling', 'maximum_coupling')
        if coupling == "sinkhorn":
            return Sinkhorn.posterior(u_s, u_t, z, self.C, SINKHORN_EPSILON, SINKHORN_ITERATIONS)
        elif coupling == "log_sinkhorn":
            return LogSinkhorn.posterior(u_s, u_t, z, self.C, self.cfg.get('sinkhorn_eps_schedule', [1.0, 0.5, 0.1]))
        else:
            return MaxCoupling.posterior(u_s, u_t, z)

    def post_soft(self, u_s, u_t, z_soft):
        coupling = self.cfg.get('coupling', 'maximum_coupling')
        if coupling == "sinkhorn":
            return Sinkhorn.posterior_soft(u_s, u_t, z_soft, self.C, SINKHORN_EPSILON, SINKHORN_ITERATIONS)
        elif coupling == "log_sinkhorn":
            return LogSinkhorn.posterior_soft(u_s, u_t, z_soft, self.C, self.cfg.get('sinkhorn_eps_schedule', [1.0, 0.5, 0.1]))
        else:
            return MaxCoupling.posterior_soft(u_s, u_t, z_soft)

    def get_relaxed_sample(self, u_t, temperature):
        relaxation = self.cfg.get('relaxation', 'concrete')
        if relaxation == "binary" and VOCAB_SIZE == 2:
            result = BinaryReparam.sample(u_t, temperature)
        elif relaxation == "st_gumbel":
            logits = torch.log(u_t + 1e-8)
            result = StraightThroughGumbel.sample(logits, temperature)
        elif relaxation == "straight_through":
            result = StraightThroughEstimator.sample(u_t)
        else:
            u_safe = torch.clamp(u_t, min=1e-6)
            u_safe = u_safe / u_safe.sum(-1, keepdim=True)
            result = RelaxedOneHotCategorical(temperature, probs=u_safe).rsample()
        return result

    def compute_tc(self, posterior, u_s, x, z_s=None):
        """Compute TC using configured method."""
        if self.cfg.get('tc_method', 'simple') == "minibatch_weighted" and z_s is not None:
            return KLDecomp.tc_minibatch_weighted(z_s, u_s, x.shape[0])
        else:
            return KLDecomp.tc_simple(posterior)
    
    def train_step_simple(self, x):
        # D3PM-style simple training: just KL loss, no phases.

        self.opt.zero_grad()
        t = torch.randint(1, NUM_TIMESTEPS + 1, (1,)).item()
        s = t - 1
        u_t, u_s = self.fwd_marg_batched(x, t, s)
        u_safe = torch.clamp(u_t, min=1e-6)
        u_safe = u_safe / u_safe.sum(-1, keepdim=True)
        z_t = Categorical(probs=u_safe).sample()

        if s == 0:
            # At t=1, posterior is just the data
            posterior = F.one_hot(x.long(), VOCAB_SIZE).float()
        else:
            posterior = self.post(u_s, u_t, z_t)

        rev_p = self.rev_dist(z_t, t)

        kl_loss = KLDecomp.factorized_kl(posterior, rev_p)
        loss = kl_loss

        if t == NUM_TIMESTEPS:
            prior = torch.ones_like(u_t) / VOCAB_SIZE
            prior_kl = KLDecomp.factorized_kl(u_t, prior)
            loss = loss + 0.1 * prior_kl

        # Skip NaN losses
        if torch.isnan(loss) or torch.isinf(loss):
            self.step += 1
            return {'loss': 0.0, 'phase': 'simple', 'timestep': t, 'nan_skip': True}

        loss.backward()
        if USE_GRADIENT_CLIPPING:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.rev.parameters()), GRADIENT_CLIP_NORM
            )
        self.opt.step()

        self.step += 1

        return {
            'loss': loss.item(),
            'kl_loss': kl_loss.item(),
            'phase': 'simple',
            'timestep': t,
        }

    def train_step(self, x):
        if self.cfg.get('simple_training', False):
            return self.train_step_simple(x)

        self.opt.zero_grad()
        t = torch.randint(1, NUM_TIMESTEPS + 1, (1,)).item()
        s = t - 1
        phase = self.phase

        u_t, u_s = self.fwd_marg_batched(x, t, s)

        # Check for NaN and handle gracefully
        if torch.isnan(u_t).any() or torch.isnan(u_s).any():
            # Fallback to non-batched computation
            u_t = self.fwd_marg(x, t)
            u_s = self.fwd_marg(x, s)
            # If still NaN, skip this step
            if torch.isnan(u_t).any() or torch.isnan(u_s).any():
                self.step += 1
                return {'loss': 0.0, 'phase': self.phase, 'timestep': t, 'nan_skip': True}

        metrics = {}

        # Min-SNR weighting
        snr_weight = 1.0
        if self.cfg.get('min_snr', False) and phase != 'pre':
            snr = (NUM_TIMESTEPS - t + 1) / t
            snr_weight = min(snr, self.cfg.get('min_snr_gamma', 5.0)) / (snr + 1e-8)

        # === PRETRAIN ===
        if phase == 'pre':
            B = x.shape[0]
            x_oh = F.one_hot(x.long(), VOCAB_SIZE).float()
            uni = torch.ones_like(x_oh) / VOCAB_SIZE

            # Configurable noise levels (2 is faster, 4 is original)
            n_levels = self.cfg.get('pretrain_noise_levels', 4)
            noise_levels = [0.2, 0.6] if n_levels == 2 else [0.1, 0.3, 0.5, 0.7]
            N = len(noise_levels)

            # Batch all noise levels together for ONE network forward pass
            z_noisy_list = []
            t_noise_list = []
            for noise_level in noise_levels:
                noisy_dist = (1 - noise_level) * x_oh + noise_level * uni
                noisy_dist = noisy_dist / noisy_dist.sum(-1, keepdim=True)
                z_noisy_list.append(Categorical(probs=noisy_dist).sample())
                t_noise_list.append(max(1, int(noise_level * NUM_TIMESTEPS)))

            z_batched = torch.cat(z_noisy_list, dim=0)
            tt_batched = torch.cat([torch.full((B,), t, device=device) for t in t_noise_list])

            logits = torch.clamp(self.rev(z_batched, tt_batched), -20, 20)
            rev_p_batched = F.softmax(logits, -1)
            rev_p_batched = torch.clamp(rev_p_batched, min=1e-8)
            rev_p_batched = rev_p_batched / rev_p_batched.sum(-1, keepdim=True)

            # Compute CE loss for each noise level
            x_targets = x.reshape(-1).long().repeat(N)
            loss = F.cross_entropy(rev_p_batched.reshape(-1, VOCAB_SIZE), x_targets)

        # === WARMUP ===
        elif phase == 'warm':
            tmp = self.get_temperature()
            z_soft = self.get_relaxed_sample(u_t, tmp)

            if s == 0:
                posterior = F.one_hot(x.long(), VOCAB_SIZE).float()
            else:
                posterior = self.post_soft(u_s, u_t, z_soft)

            rev_p = self.rev_dist(z_soft.permute(0, 3, 1, 2), t)

            kl_loss = KLDecomp.factorized_kl(posterior, rev_p)
            loss = kl_loss * snr_weight
            metrics['kl_loss'] = kl_loss.item()

            # TC regularization
            if self.cfg.get('tc_reg', False):
                tc = self.compute_tc(posterior, u_s, x)
                loss = loss + self.cfg.get('tc_lambda', 0.1) * tc
                metrics['tc'] = tc.item()

            # Reconstruction loss
            if self.cfg.get('recon_loss', True) and s == 0:
                x_oh = F.one_hot(x.long(), VOCAB_SIZE).float()
                recon_loss = KLDecomp.factorized_kl(x_oh, rev_p)
                loss = loss + recon_loss
                metrics['recon_loss'] = recon_loss.item()

            # Auxiliary CE loss
            if self.cfg.get('aux_ce_loss', False):
                aux_ce = F.cross_entropy(rev_p.reshape(-1, VOCAB_SIZE), x.reshape(-1).long())
                loss = loss + self.cfg.get('aux_ce_lambda', 0.001) * aux_ce
                metrics['aux_ce'] = aux_ce.item()

            if self.cfg.get('temp_schedule', 'exponential') == "exponential":
                self.temp = max(self.temp * self.decay, self.cfg.get('min_temp', 0.1))
            metrics['temp'] = tmp
        
        # === REINFORCE ===
        else:
            if isinstance(self.baseline, RLOOBaseline):
                K = self.cfg.get('rloo_k', 4)
                u_safe = torch.clamp(u_t, min=1e-6)
                u_safe = u_safe / u_safe.sum(-1, keepdim=True)
                dist = Categorical(probs=u_safe)
                # Normalize log_prob to per-pixel scale to match KL normalization
                num_pixels = x.shape[1] * x.shape[2]

                rewards, log_probs = [], []
                for k in range(K):
                    z_t = dist.sample()
                    lp = dist.log_prob(z_t).sum([1, 2]) / num_pixels
                    if s == 0:
                        posterior = F.one_hot(x.long(), VOCAB_SIZE).float()
                    else:
                        posterior = self.post(u_s, u_t, z_t)
                    rev_p = self.rev_dist(z_t, t)
                    kl = KLDecomp.kl_per_sample(posterior, rev_p)
                    rewards.append(-kl)
                    log_probs.append(lp)

                rewards = torch.stack(rewards, dim=-1)
                log_probs = torch.stack(log_probs, dim=-1)
                reinforce_loss = self.baseline.compute_gradient(rewards, log_probs)
                kl_loss = -rewards.mean()
                loss = kl_loss + reinforce_loss
                metrics['kl_loss'] = kl_loss.item()
                metrics['reinforce_loss'] = reinforce_loss.item()
            else:
                u_safe = torch.clamp(u_t, min=1e-6)
                u_safe = u_safe / u_safe.sum(-1, keepdim=True)
                dist = Categorical(probs=u_safe)
                z_t = dist.sample()
                # Normalize log_prob to per-pixel scale to match KL normalization
                num_pixels = x.shape[1] * x.shape[2]
                lp = dist.log_prob(z_t).sum([1, 2]) / num_pixels

                if s == 0:
                    posterior = F.one_hot(x.long(), VOCAB_SIZE).float()
                else:
                    posterior = self.post(u_s, u_t, z_t)
                rev_p = self.rev_dist(z_t, t)

                kl = KLDecomp.kl_per_sample(posterior, rev_p)
                kl_loss = kl.mean()
                baseline = self.baseline.update(lp, kl)
                advantage = kl.detach() - baseline
                reinforce_loss = (lp * advantage).mean()
                loss = kl_loss + reinforce_loss
                metrics['kl_loss'] = kl_loss.item()
                metrics['reinforce_loss'] = reinforce_loss.item()

            loss = loss * snr_weight
            
            # TC regularization in RL phase
            if self.cfg.get('tc_reg', False):
                u_safe = torch.clamp(u_t, min=1e-6)
                u_safe = u_safe / u_safe.sum(-1, keepdim=True)
                z_t = Categorical(probs=u_safe).sample()
                z_s = Categorical(probs=torch.clamp(u_s, min=1e-6)).sample() if s > 0 else x.long()
                posterior = F.one_hot(x.long(), VOCAB_SIZE).float() if s == 0 else self.post(u_s, u_t, z_t)
                tc = self.compute_tc(posterior, u_s, x, z_s)
                loss = loss + self.cfg.get('tc_lambda', 0.1) * tc
                metrics['tc'] = tc.item()

            # Reconstruction loss
            if self.cfg.get('recon_loss', True) and s == 0:
                x_oh = F.one_hot(x.long(), VOCAB_SIZE).float()
                recon = KLDecomp.kl_per_sample(x_oh, rev_p).mean()
                loss = loss + recon

            # Auxiliary CE loss
            if self.cfg.get('aux_ce_loss', False):
                aux_ce = F.cross_entropy(rev_p.reshape(-1, VOCAB_SIZE), x.reshape(-1).long())
                loss = loss + self.cfg.get('aux_ce_lambda', 0.001) * aux_ce
                metrics['aux_ce'] = aux_ce.item()

        # Prior loss
        if t == NUM_TIMESTEPS:
            prior = torch.ones_like(u_t) / VOCAB_SIZE
            prior_kl = KLDecomp.factorized_kl(u_t, prior)
            prior_weight = 1.0 if self.cfg.get('preset', 'paper') in ["paper", "working"] else 0.1
            loss = loss + prior_weight * prior_kl
            metrics['prior_kl'] = prior_kl.item()

        # Boundary loss - enforce proper forward process behavior
        if self.cfg.get('boundary_loss', False) and self.cfg.get('train_forward', True) and self.cfg.get('forward_mode', 'paper') != 'standard':
            x_oh = F.one_hot(x.long(), VOCAB_SIZE).float()
            uni = torch.ones_like(x_oh) / VOCAB_SIZE
            a = t / NUM_TIMESTEPS  # How far along we are (0=data, 1=noise)

            # Expected interpolation at this timestep
            expected = (1 - a) * x_oh + a * uni

            # Boundary deviation: how far is u_t from expected interpolation? Weight more heavily near boundaries where we have strong priors
            boundary_weight = 4.0 * a * (1 - a)  # Peaks at 0.5, zero at boundaries
            boundary_weight = 1.0 - boundary_weight  # Invert: high at boundaries, low in middle

            # MSE loss between u_t and expected interpolation
            boundary_dev = ((u_t - expected) ** 2).mean()
            boundary_loss = self.cfg.get('boundary_lambda', 1.0) * boundary_weight * boundary_dev
            loss = loss + boundary_loss
            metrics['boundary_loss'] = boundary_loss.item()

            # Entropy monotonicity: H(u_t) should be >= H(u_s)
            H_t = -(u_t * torch.log(u_t + 1e-10)).sum(-1).mean()
            H_s = -(u_s * torch.log(u_s + 1e-10)).sum(-1).mean()
            # Penalize if entropy decreases (H_t < H_s)
            entropy_violation = F.relu(H_s - H_t)  # Only penalize if H_t < H_s
            entropy_loss = self.cfg.get('entropy_mono_lambda', 0.5) * entropy_violation
            loss = loss + entropy_loss
            metrics['entropy_t'] = H_t.item()
            metrics['entropy_s'] = H_s.item()
            metrics['entropy_loss'] = entropy_loss.item()

        if torch.isnan(loss) or torch.isinf(loss):
            self.step += 1
            return {'loss': 0.0, 'phase': phase, 'timestep': t}

        # Backward pass with optional AMP scaling
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if USE_GRADIENT_CLIPPING:
                self.scaler.unscale_(self.opt)
                params_to_clip = list(self.rev.parameters())
                if self.cfg.get('train_forward', True):
                    params_to_clip += list(self.fwd.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, GRADIENT_CLIP_NORM)
                metrics['grad_norm'] = grad_norm.item()
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            if USE_GRADIENT_CLIPPING:
                params_to_clip = list(self.rev.parameters())
                if self.cfg.get('train_forward', True):
                    params_to_clip += list(self.fwd.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, GRADIENT_CLIP_NORM)
                metrics['grad_norm'] = grad_norm.item()
            self.opt.step()

        # EMA update
        if self.cfg.get('use_ema', False) and self.step >= EMA_START_STEP:
            if self.ema_fwd:
                self.ema_fwd.update()
            if self.ema_rev:
                self.ema_rev.update()

        self.step += 1
        metrics['loss'] = loss.item()
        metrics['phase'] = phase
        metrics['timestep'] = t
        
        return metrics
    
    @torch.no_grad()
    def sample(self, n=16, sz=28, use_ema=True):
        
        if use_ema and self.cfg.get('use_ema', False) and self.ema_rev is not None:
            self.ema_rev.apply_shadow()

        self.fwd.eval()
        self.rev.eval()

        z = torch.randint(0, VOCAB_SIZE, (n, sz, sz), device=device)

        for t in reversed(range(1, NUM_TIMESTEPS + 1)):
            rev_p = self.rev_dist(z, t)
            z = Categorical(probs=rev_p).sample()
            # Count how many pixels are 0 vs 1
            num_zeros = (z == 0).sum().item()
            num_ones = (z == 1).sum().item()

        self.fwd.train()
        self.rev.train()

        if use_ema and self.cfg.get('use_ema', False) and self.ema_rev is not None:
            self.ema_rev.restore()

        
        return z


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_forward_process(model, x_samples, n_rows=4):
    model.fwd.eval()
    x = x_samples[:n_rows]
    
    fig, axes = plt.subplots(n_rows, NUM_TIMESTEPS + 1, figsize=(1.5 * (NUM_TIMESTEPS + 1), 1.5 * n_rows))
    
    for t in range(NUM_TIMESTEPS + 1):
        u_t = model.fwd_marg(x, t)
        if t == 0:
            z_t = x
        else:
            z_t = Categorical(probs=u_t.clamp(min=1e-6)).sample()
        
        for i in range(n_rows):
            axes[i, t].imshow(z_t[i].cpu().numpy(), cmap='gray', vmin=0, vmax=VOCAB_SIZE - 1)
            axes[i, t].axis('off')
            if i == 0:
                axes[i, t].set_title(f't={t}', fontsize=8)
    
    plt.suptitle(f'Forward: Data → Noise ({model.cfg["forward_mode"]} mode)', fontsize=10)
    plt.tight_layout()
    model.fwd.train()
    return fig


def visualize_backward_process(model, n_rows=4):
    model.fwd.eval()
    model.rev.eval()
    
    fig, axes = plt.subplots(n_rows, NUM_TIMESTEPS + 1, figsize=(1.5 * (NUM_TIMESTEPS + 1), 1.5 * n_rows))
    
    z = torch.randint(0, VOCAB_SIZE, (n_rows, 28, 28), device=device)
    
    for i in range(n_rows):
        axes[i, 0].imshow(z[i].cpu().numpy(), cmap='gray', vmin=0, vmax=VOCAB_SIZE - 1)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title(f't={NUM_TIMESTEPS}', fontsize=8)
    
    for col, t in enumerate(reversed(range(1, NUM_TIMESTEPS + 1)), start=1):
        z = Categorical(probs=model.rev_dist(z, t)).sample()
        for i in range(n_rows):
            axes[i, col].imshow(z[i].cpu().numpy(), cmap='gray', vmin=0, vmax=VOCAB_SIZE - 1)
            axes[i, col].axis('off')
            if i == 0:
                axes[i, col].set_title(f't={t - 1}', fontsize=8)
    
    plt.suptitle('Backward: Noise → Data', fontsize=10)
    plt.tight_layout()
    model.fwd.train()
    model.rev.train()
    return fig


# =============================================================================
# TRAINING LOGGER
# =============================================================================

class TrainingLogger:
    """Logs training metrics to JSON for later comparison."""

    def __init__(self, run_name, cfg, output_dir='./outputs'):
        self.run_name = run_name
        self.output_dir = output_dir
        self.start_time = datetime.now()

        self.log = {
            "run_name": run_name,
            "preset": cfg.get('preset', 'unknown'),
            "config": cfg,
            "start_time": self.start_time.isoformat(),
            "metrics": [],
            "sample_quality": [],
            "checkpoints": [],
        }

        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, f"run_{run_name}.json")

    def log_step(self, step, phase, metrics):
        """Log metrics for a training step."""
        entry = {
            "step": step,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
               for k, v in metrics.items()}
        }
        self.log["metrics"].append(entry)

    def log_sample_quality(self, step, samples):
        """Log sample quality metrics."""
        samples_np = samples.cpu().numpy() if hasattr(samples, 'cpu') else samples
        ones_ratio = float(np.mean(samples_np))
        pixel_var = float(np.var(samples_np, axis=0).mean())
        unique = len(np.unique(samples_np.reshape(samples_np.shape[0], -1), axis=0))

        entry = {
            "step": step,
            "ones_ratio": ones_ratio,
            "pixel_variance": pixel_var,
            "unique_samples": unique,
            "diversity": unique / samples_np.shape[0],
        }
        self.log["sample_quality"].append(entry)
        return entry

    def log_checkpoint(self, step, epoch, avg_loss):
        """Log a checkpoint."""
        self.log["checkpoints"].append({
            "step": step,
            "epoch": epoch,
            "avg_loss": float(avg_loss),
            "timestamp": datetime.now().isoformat(),
        })

    def save(self):
        """Save log to JSON file."""
        self.log["end_time"] = datetime.now().isoformat()
        self.log["duration_seconds"] = (datetime.now() - self.start_time).total_seconds()

        with open(self.log_path, 'w') as f:
            json.dump(self.log, f, indent=2, default=str)
        return self.log_path

    def print_summary(self):
        """Print training summary."""
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY: {self.run_name}")
        print(f"{'='*60}")
        print(f"  Preset: {self.log['preset']}")
        print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"  Total steps logged: {len(self.log['metrics'])}")

        if self.log["metrics"]:
            final_loss = self.log["metrics"][-1].get("loss", "N/A")
            print(f"  Final loss: {final_loss:.4f}" if isinstance(final_loss, float) else f"  Final loss: {final_loss}")

        if self.log["sample_quality"]:
            last_sq = self.log["sample_quality"][-1]
            print(f"  Final sample quality:")
            print(f"    ones_ratio: {last_sq['ones_ratio']:.3f}")
            print(f"    pixel_variance: {last_sq['pixel_variance']:.3f}")
            print(f"    diversity: {last_sq['diversity']:.3f}")

        print(f"  Log saved to: {self.log_path}")
        print(f"{'='*60}\n")


# =============================================================================
# DATA & TRAINING
# =============================================================================

def get_data(seed=None):
    tr = transforms.Compose([transforms.ToTensor(), lambda x: (x > 0.5).float()])
    ds = datasets.MNIST('./data', train=True, download=True, transform=tr)
    
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    
    return DataLoader(ds, env.batch_size, shuffle=True, num_workers=env.num_workers, 
                      pin_memory=env.pin_memory, generator=generator,
                      worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id) if seed else None)


def train():
    cfg = apply_preset()
    set_seed(cfg['seed'], deterministic=cfg.get('deterministic', True))

    # Create run name with timestamp
    run_name = f"{cfg['preset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = f'./outputs/{run_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize logger
    logger = TrainingLogger(run_name, cfg, output_dir)

    print(f"\n{'='*60}")
    print(f"STARTING TRAINING RUN: {run_name}")
    print(f"{'='*60}")
    print(f"  Preset: {cfg['preset']}")
    print(f"  Forward mode: {cfg['forward_mode']}")
    print(f"  Train forward: {cfg['train_forward']}")
    print(f"  Coupling: {cfg['coupling']}")
    print(f"  Device: {env.name} | Batch size: {env.batch_size}")
    print(f"  Network: {cfg.get('network_size', 'normal')} | AMP: {cfg.get('use_amp', False)}")
    print(f"  Compile: {cfg.get('use_compile', False)} | Deterministic: {cfg.get('deterministic', True)}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}\n")

    loader = get_data(seed=cfg['seed'])
    model = FLDD(cfg)

    ref_batch = next(iter(loader))[0].squeeze(1).to(device)
    epochs = (model.total + len(loader) - 1) // len(loader)

    # Progress tracking
    LOG_INTERVAL = 50  # Log to JSON every N steps
    SAMPLE_INTERVAL = 500  # Generate samples every N steps
    last_sample_step = 0

    print(f"Training for {model.total} steps ({epochs} epochs)")
    if cfg.get('simple_training', False):
        print(f"  Mode: D3PM-style simple KL training (no phases)")
    else:
        print(f"  Phases: pretrain={cfg['pretrain']}, warmup={cfg['warmup']}, reinforce={cfg['reinforce']}")

    for ep in range(epochs):
        losses = []
        pbar = tqdm(loader, desc=f'Ep {ep + 1}/{epochs}',
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

        for batch_idx, (x, _) in enumerate(pbar):
            if model.step >= model.total:
                break

            x = x.squeeze(1).to(device)
            m = model.train_step(x)
            loss = m.get('loss', 0)
            losses.append(loss)

            # Update progress bar
            if model.step % LOG_EVERY == 0:
                progress_pct = 100 * model.step / model.total
                d = {
                    'loss': f"{loss:.3f}",
                    'phase': m.get('phase', '')[:4],
                    'prog': f"{progress_pct:.0f}%",
                }
                if 'temp' in m:
                    d['T'] = f"{m['temp']:.2f}"
                if 'tc' in m:
                    d['tc'] = f"{m['tc']:.3f}"
                pbar.set_postfix(d)

            # Log to JSON periodically
            if model.step % LOG_INTERVAL == 0:
                logger.log_step(model.step, model.phase, m)

            # Generate and evaluate samples periodically
            if model.step - last_sample_step >= SAMPLE_INTERVAL:
                with torch.no_grad():
                    samples = model.sample(32, use_ema=False)
                    sq = logger.log_sample_quality(model.step, samples)
                    print(f"\n  [Step {model.step}] Sample quality: ones={sq['ones_ratio']:.3f}, var={sq['pixel_variance']:.3f}")

                    # Save sample image
                    fig, ax = plt.subplots(4, 8, figsize=(8, 4))
                    for i, a in enumerate(ax.flat):
                        a.imshow(samples[i].cpu().numpy(), cmap='gray')
                        a.axis('off')
                    plt.suptitle(f"Step {model.step} | ones={sq['ones_ratio']:.3f}")
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/samples_step{model.step}.png', dpi=100)
                    plt.close()

                last_sample_step = model.step

        # End of epoch
        avg_loss = np.mean(losses)
        logger.log_checkpoint(model.step, ep + 1, avg_loss)
        print(f"\n  Epoch {ep + 1}/{epochs} complete: avg_loss={avg_loss:.4f}, step={model.step}/{model.total}")

        # Visualization at epoch end
        if (ep + 1) % VIS_EVERY == 0 or ep == 0:
            fig_fwd = visualize_forward_process(model, ref_batch, n_rows=4)
            plt.savefig(f'{output_dir}/forward_ep{ep + 1}.png', dpi=100)
            plt.close()

            fig_bwd = visualize_backward_process(model, n_rows=4)
            plt.savefig(f'{output_dir}/backward_ep{ep + 1}.png', dpi=100)
            plt.close()

        # Save intermediate log
        logger.save()

        if model.step >= model.total:
            break

    # Final outputs
    print("\nGenerating final samples...")
    s = model.sample(64)
    final_sq = logger.log_sample_quality(model.step, s)

    fig, ax = plt.subplots(8, 8, figsize=(8, 8))
    for i, a in enumerate(ax.flat):
        a.imshow(s[i].cpu().numpy(), cmap='gray')
        a.axis('off')
    plt.suptitle(f"{cfg['preset']} | ones={final_sq['ones_ratio']:.3f}, var={final_sq['pixel_variance']:.3f}")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/final_samples.png', dpi=150)
    plt.close()

    # Save model
    torch.save({
        'fwd': model.fwd.state_dict(),
        'rev': model.rev.state_dict(),
        'cfg': cfg,
        'step': model.step,
    }, f'{output_dir}/model.pth')

    # Final log save and summary
    logger.save()
    logger.print_summary()

    print(f"All outputs saved to: {output_dir}/")
    return logger
    


if __name__ == "__main__":
    train()
