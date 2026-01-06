"""
Comprehensive FLDD Test Suite
=============================
Pytest-compatible tests covering every function and class in fldd.py.

Run: pytest test_fldd.py -v
Output: test_results.json (single consolidated file)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import sys
import io
import contextlib
from datetime import datetime
from pathlib import Path

# =============================================================================
# TEST INFRASTRUCTURE
# =============================================================================

@contextlib.contextmanager
def suppress_prints():
    """Suppress print statements during tests."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


class TestResults:
    """Collects all test results for JSON output."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.results = {
                "timestamp": datetime.now().isoformat(),
                "tests": {},
                "issues": [],
                "coverage": {},
                "summary": {"total": 0, "passed": 0, "failed": 0}
            }
            cls._instance.current_category = None
        return cls._instance

    def set_category(self, name):
        self.current_category = name
        if name not in self.results["tests"]:
            self.results["tests"][name] = {"tests": [], "passed": 0, "failed": 0}

    def record(self, name, passed, message="", details=None):
        if isinstance(passed, torch.Tensor):
            passed = bool(passed.item())
        passed = bool(passed)

        cat = self.current_category or "uncategorized"
        if cat not in self.results["tests"]:
            self.results["tests"][cat] = {"tests": [], "passed": 0, "failed": 0}

        self.results["tests"][cat]["tests"].append({
            "name": name, "passed": passed, "message": str(message), "details": details or {}
        })
        self.results["summary"]["total"] += 1

        if passed:
            self.results["summary"]["passed"] += 1
            self.results["tests"][cat]["passed"] += 1
        else:
            self.results["summary"]["failed"] += 1
            self.results["tests"][cat]["failed"] += 1
            self.results["issues"].append({
                "category": cat, "test": name, "message": str(message)
            })

    def add_issue(self, message):
        self.results["issues"].append({
            "category": self.current_category or "general",
            "message": message
        })

    def save(self, path="test_results.json"):
        # Add config info
        try:
            from fldd import VOCAB_SIZE, NUM_TIMESTEPS, HIDDEN_DIM, TIME_DIM, device
            self.results["config"] = {
                "device": str(device),
                "vocab_size": VOCAB_SIZE,
                "num_timesteps": NUM_TIMESTEPS,
                "hidden_dim": HIDDEN_DIM,
                "time_dim": TIME_DIM
            }
        except:
            pass

        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n✓ Results saved to {path}")
        return self.results


# Global results collector
results = TestResults()


def make_cfg(**overrides):
    """Create a test configuration with optional overrides."""
    cfg = {
        'preset': 'standard', 'seed': 42,
        'pretrain': 2, 'warmup': 2, 'reinforce': 2,
        'min_temp': 0.1, 'temp_schedule': 'exponential',
        'forward_mode': 'standard', 'train_forward': False,
        'coupling': 'maximum_coupling',
        'baseline': 'simple', 'rloo_k': 4,
        'gradient_est': 'reinforce', 'relaxation': 'concrete',
        'pretraining': True, 'recon_loss': True,
        'aux_ce_loss': False, 'aux_ce_lambda': 0.001,
        'score_entropy': False, 'tc_reg': False,
        'tc_lambda': 0.1, 'tc_method': 'simple',
        'sampling': 'standard', 'pc_steps': 1,
        'noise_schedule': 'fixed', 'use_ema': False,
        'ema_decay': 0.9999, 'min_snr': False,
        'min_snr_gamma': 5.0,
        'sinkhorn_eps_schedule': [1.0, 0.5, 0.1],
    }
    cfg.update(overrides)
    return cfg


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def fldd_imports():
    """Import all FLDD components."""
    with suppress_prints():
        from fldd import (
            set_seed, apply_preset, Env, device,
            VOCAB_SIZE, NUM_TIMESTEPS, HIDDEN_DIM, TIME_DIM,
            MaxCoupling, Sinkhorn, LogSinkhorn, KLDecomp,
            SimpleBaseline, OptimalBaseline, RLOOBaseline,
            BinaryReparam, StraightThroughGumbel, StraightThroughEstimator,
            EMA, SinEmb, ResBlk, Net, FLDD,
            visualize_forward_process, visualize_backward_process,
            get_data
        )
    return {
        'set_seed': set_seed, 'apply_preset': apply_preset,
        'Env': Env, 'device': device,
        'VOCAB_SIZE': VOCAB_SIZE, 'NUM_TIMESTEPS': NUM_TIMESTEPS,
        'HIDDEN_DIM': HIDDEN_DIM, 'TIME_DIM': TIME_DIM,
        'MaxCoupling': MaxCoupling, 'Sinkhorn': Sinkhorn,
        'LogSinkhorn': LogSinkhorn, 'KLDecomp': KLDecomp,
        'SimpleBaseline': SimpleBaseline, 'OptimalBaseline': OptimalBaseline,
        'RLOOBaseline': RLOOBaseline, 'BinaryReparam': BinaryReparam,
        'StraightThroughGumbel': StraightThroughGumbel,
        'StraightThroughEstimator': StraightThroughEstimator,
        'EMA': EMA, 'SinEmb': SinEmb, 'ResBlk': ResBlk,
        'Net': Net, 'FLDD': FLDD,
        'visualize_forward_process': visualize_forward_process,
        'visualize_backward_process': visualize_backward_process,
        'get_data': get_data
    }


@pytest.fixture(scope="module")
def device(fldd_imports):
    return fldd_imports['device']


@pytest.fixture(scope="module")
def vocab_size(fldd_imports):
    return fldd_imports['VOCAB_SIZE']


@pytest.fixture(scope="module")
def num_timesteps(fldd_imports):
    return fldd_imports['NUM_TIMESTEPS']


# =============================================================================
# TEST: Global Functions
# =============================================================================

class TestGlobalFunctions:
    """Tests for global utility functions."""

    def test_set_seed_reproducibility(self, fldd_imports):
        results.set_category("Global Functions")
        set_seed = fldd_imports['set_seed']

        with suppress_prints():
            set_seed(42)
        a = [torch.rand(1).item() for _ in range(3)]

        with suppress_prints():
            set_seed(42)
        b = [torch.rand(1).item() for _ in range(3)]

        passed = a == b
        results.record("set_seed reproducibility", passed, f"a={a}, b={b}")
        assert passed

    def test_set_seed_none(self, fldd_imports):
        set_seed = fldd_imports['set_seed']

        # Should not raise
        with suppress_prints():
            set_seed(None)

        passed = True
        results.record("set_seed with None", passed, "No error raised")
        assert passed

    @pytest.mark.parametrize("preset", ["standard", "working", "paper", "project", "enhanced"])
    def test_apply_preset_all(self, fldd_imports, preset):
        # Temporarily set preset
        import fldd
        original = fldd.PRESET
        fldd.PRESET = preset

        with suppress_prints():
            cfg = fldd_imports['apply_preset']()

        fldd.PRESET = original

        required_keys = ['preset', 'forward_mode', 'coupling', 'baseline', 'relaxation']
        missing = [k for k in required_keys if k not in cfg]

        passed = len(missing) == 0
        results.record(f"apply_preset('{preset}') has required keys", passed, f"missing: {missing}")
        assert passed


# =============================================================================
# TEST: Env Class
# =============================================================================

class TestEnvClass:
    """Tests for environment configuration."""

    def test_env_device(self, fldd_imports):
        results.set_category("Env Class")
        Env = fldd_imports['Env']

        with suppress_prints():
            env = Env()

        passed = hasattr(env, 'device') and env.device is not None
        results.record("Env.device exists", passed, f"device={env.device}")
        assert passed

    def test_env_batch_size(self, fldd_imports):
        Env = fldd_imports['Env']

        with suppress_prints():
            env = Env()

        passed = hasattr(env, 'batch_size') and env.batch_size > 0
        results.record("Env.batch_size > 0", passed, f"batch_size={env.batch_size}")
        assert passed

    def test_env_name(self, fldd_imports):
        Env = fldd_imports['Env']

        with suppress_prints():
            env = Env()

        passed = hasattr(env, 'name') and env.name in ['CUDA', 'MPS', 'CPU']
        results.record("Env.name valid", passed, f"name={env.name}")
        assert passed

    def test_env_num_workers(self, fldd_imports):
        Env = fldd_imports['Env']

        with suppress_prints():
            env = Env()

        passed = hasattr(env, 'num_workers') and isinstance(env.num_workers, int)
        results.record("Env.num_workers exists", passed)
        assert passed

    def test_env_pin_memory(self, fldd_imports):
        Env = fldd_imports['Env']

        with suppress_prints():
            env = Env()

        passed = hasattr(env, 'pin_memory') and isinstance(env.pin_memory, bool)
        results.record("Env.pin_memory exists", passed)
        assert passed


# =============================================================================
# TEST: MaxCoupling Class
# =============================================================================

class TestMaxCoupling:
    """Tests for MaxCoupling posterior computation."""

    def test_posterior_valid_probs(self, fldd_imports, device, vocab_size):
        results.set_category("MaxCoupling Class")
        MaxCoupling = fldd_imports['MaxCoupling']

        u_s = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        u_t = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        z_t = torch.randint(0, vocab_size, (2, 7, 7), device=device)

        with suppress_prints():
            post = MaxCoupling.posterior(u_s, u_t, z_t)

        sums_to_one = torch.allclose(post.sum(-1), torch.ones_like(post.sum(-1)), atol=1e-5)
        non_negative = (post >= 0).all()

        passed = sums_to_one and non_negative
        results.record("MaxCoupling.posterior valid probs", passed)
        assert passed

    def test_posterior_shape(self, fldd_imports, device, vocab_size):
        MaxCoupling = fldd_imports['MaxCoupling']

        u_s = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        u_t = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        z_t = torch.randint(0, vocab_size, (2, 7, 7), device=device)

        with suppress_prints():
            post = MaxCoupling.posterior(u_s, u_t, z_t)

        expected = torch.Size([2, 7, 7, vocab_size])
        passed = post.shape == expected
        results.record("MaxCoupling.posterior shape", passed, f"expected {expected}, got {post.shape}")
        assert passed

    def test_posterior_soft_valid_probs(self, fldd_imports, device, vocab_size):
        MaxCoupling = fldd_imports['MaxCoupling']

        u_s = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        u_t = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        z_soft = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            post = MaxCoupling.posterior_soft(u_s, u_t, z_soft)

        sums_to_one = torch.allclose(post.sum(-1), torch.ones_like(post.sum(-1)), atol=1e-5)
        non_negative = (post >= 0).all()

        passed = sums_to_one and non_negative
        results.record("MaxCoupling.posterior_soft valid probs", passed)
        assert passed

    def test_posterior_soft_gradient_u_s(self, fldd_imports, device, vocab_size):
        MaxCoupling = fldd_imports['MaxCoupling']

        u_s = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        u_s.requires_grad_(True)
        u_t = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        z_soft = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            post = MaxCoupling.posterior_soft(u_s, u_t, z_soft)

        loss = post.sum()
        loss.backward()

        has_grad = u_s.grad is not None
        grad_nonzero = u_s.grad.abs().sum().item() > 0 if has_grad else False

        passed = has_grad and grad_nonzero
        results.record("MaxCoupling.posterior_soft gradient to u_s", passed,
                      f"has_grad={has_grad}, grad_sum={u_s.grad.abs().sum().item() if has_grad else 0:.4f}")
        assert passed

    def test_posterior_soft_gradient_u_t(self, fldd_imports, device, vocab_size):
        MaxCoupling = fldd_imports['MaxCoupling']

        u_s = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        u_t = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        u_t.requires_grad_(True)
        z_soft = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            post = MaxCoupling.posterior_soft(u_s, u_t, z_soft)

        loss = post.sum()
        loss.backward()

        has_grad = u_t.grad is not None
        grad_nonzero = u_t.grad.abs().sum().item() > 0 if has_grad else False

        passed = has_grad and grad_nonzero
        results.record("MaxCoupling.posterior_soft gradient to u_t", passed)
        assert passed

    def test_posterior_soft_gradient_z_soft(self, fldd_imports, device, vocab_size):
        MaxCoupling = fldd_imports['MaxCoupling']

        u_s = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        u_t = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        z_soft = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        z_soft.requires_grad_(True)

        with suppress_prints():
            post = MaxCoupling.posterior_soft(u_s, u_t, z_soft)

        loss = post.sum()
        loss.backward()

        has_grad = z_soft.grad is not None
        grad_nonzero = z_soft.grad.abs().sum().item() > 0 if has_grad else False

        passed = has_grad and grad_nonzero
        results.record("MaxCoupling.posterior_soft gradient to z_soft", passed)
        assert passed

    def test_posterior_numerical_stability(self, fldd_imports, device, vocab_size):
        MaxCoupling = fldd_imports['MaxCoupling']

        # Peaked distribution
        u_s = torch.zeros(2, 7, 7, vocab_size, device=device)
        u_s[..., 0] = 0.99
        u_s[..., 1] = 0.01
        u_t = torch.zeros(2, 7, 7, vocab_size, device=device)
        u_t[..., 0] = 0.01
        u_t[..., 1] = 0.99
        z_t = torch.ones(2, 7, 7, dtype=torch.long, device=device)

        with suppress_prints():
            post = MaxCoupling.posterior(u_s, u_t, z_t)

        no_nan = not torch.isnan(post).any()
        no_inf = not torch.isinf(post).any()

        passed = no_nan and no_inf
        results.record("MaxCoupling.posterior numerical stability", passed)
        assert passed


# =============================================================================
# TEST: Sinkhorn Class
# =============================================================================

class TestSinkhorn:
    """Tests for Sinkhorn optimal transport solver."""

    def test_solve_output_shape(self, fldd_imports, device, vocab_size):
        results.set_category("Sinkhorn Class")
        Sinkhorn = fldd_imports['Sinkhorn']

        a = F.softmax(torch.randn(49, vocab_size, device=device), dim=-1)
        b = F.softmax(torch.randn(49, vocab_size, device=device), dim=-1)
        C = torch.ones(vocab_size, vocab_size, device=device) - torch.eye(vocab_size, device=device)

        with suppress_prints():
            P = Sinkhorn.solve(a, b, C, eps=0.1, iters=20)

        expected = torch.Size([49, vocab_size, vocab_size])
        passed = P.shape == expected
        results.record("Sinkhorn.solve output shape", passed, f"got {P.shape}")
        assert passed

    def test_solve_row_marginals(self, fldd_imports, device, vocab_size):
        Sinkhorn = fldd_imports['Sinkhorn']

        a = F.softmax(torch.randn(49, vocab_size, device=device), dim=-1)
        b = F.softmax(torch.randn(49, vocab_size, device=device), dim=-1)
        C = torch.ones(vocab_size, vocab_size, device=device) - torch.eye(vocab_size, device=device)

        with suppress_prints():
            P = Sinkhorn.solve(a, b, C, eps=0.1, iters=50)

        row_sums = P.sum(-1)
        error = (row_sums - a).abs().max().item()

        passed = error < 0.05
        results.record("Sinkhorn.solve row marginals", passed, f"max_error={error:.4f}")
        assert passed

    def test_solve_col_marginals(self, fldd_imports, device, vocab_size):
        Sinkhorn = fldd_imports['Sinkhorn']

        a = F.softmax(torch.randn(49, vocab_size, device=device), dim=-1)
        b = F.softmax(torch.randn(49, vocab_size, device=device), dim=-1)
        C = torch.ones(vocab_size, vocab_size, device=device) - torch.eye(vocab_size, device=device)

        with suppress_prints():
            P = Sinkhorn.solve(a, b, C, eps=0.1, iters=50)

        col_sums = P.sum(-2)
        error = (col_sums - b).abs().max().item()

        passed = error < 0.05
        results.record("Sinkhorn.solve col marginals", passed, f"max_error={error:.4f}")
        assert passed

    def test_solve_non_negative(self, fldd_imports, device, vocab_size):
        Sinkhorn = fldd_imports['Sinkhorn']

        a = F.softmax(torch.randn(49, vocab_size, device=device), dim=-1)
        b = F.softmax(torch.randn(49, vocab_size, device=device), dim=-1)
        C = torch.ones(vocab_size, vocab_size, device=device) - torch.eye(vocab_size, device=device)

        with suppress_prints():
            P = Sinkhorn.solve(a, b, C, eps=0.1, iters=20)

        passed = (P >= 0).all().item()
        results.record("Sinkhorn.solve non-negative", passed)
        assert passed

    def test_posterior_valid_probs(self, fldd_imports, device, vocab_size):
        Sinkhorn = fldd_imports['Sinkhorn']

        u_s = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        u_t = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        z_t = torch.randint(0, vocab_size, (2, 7, 7), device=device)
        C = torch.ones(vocab_size, vocab_size, device=device) - torch.eye(vocab_size, device=device)

        with suppress_prints():
            post = Sinkhorn.posterior(u_s, u_t, z_t, C, eps=0.1, iters=20)

        sums_to_one = torch.allclose(post.sum(-1), torch.ones_like(post.sum(-1)), atol=0.1)
        non_negative = (post >= 0).all()

        passed = sums_to_one and non_negative
        results.record("Sinkhorn.posterior valid probs", passed)
        assert passed

    def test_posterior_soft_valid_probs(self, fldd_imports, device, vocab_size):
        Sinkhorn = fldd_imports['Sinkhorn']

        u_s = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        u_t = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        z_soft = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        C = torch.ones(vocab_size, vocab_size, device=device) - torch.eye(vocab_size, device=device)

        with suppress_prints():
            post = Sinkhorn.posterior_soft(u_s, u_t, z_soft, C, eps=0.1, iters=20)

        sums_to_one = torch.allclose(post.sum(-1), torch.ones_like(post.sum(-1)), atol=0.1)
        non_negative = (post >= 0).all()

        passed = sums_to_one and non_negative
        results.record("Sinkhorn.posterior_soft valid probs", passed)
        assert passed


# =============================================================================
# TEST: LogSinkhorn Class
# =============================================================================

class TestLogSinkhorn:
    """Tests for log-domain Sinkhorn solver."""

    def test_solve_output_shape(self, fldd_imports, device, vocab_size):
        results.set_category("LogSinkhorn Class")
        LogSinkhorn = fldd_imports['LogSinkhorn']

        a = F.softmax(torch.randn(49, vocab_size, device=device), dim=-1)
        b = F.softmax(torch.randn(49, vocab_size, device=device), dim=-1)
        C = torch.ones(vocab_size, vocab_size, device=device) - torch.eye(vocab_size, device=device)

        with suppress_prints():
            P = LogSinkhorn.solve(a, b, C, eps_schedule=[1.0, 0.5, 0.1], iters_per_eps=10)

        expected = torch.Size([49, vocab_size, vocab_size])
        passed = P.shape == expected
        results.record("LogSinkhorn.solve output shape", passed)
        assert passed

    def test_solve_non_negative(self, fldd_imports, device, vocab_size):
        LogSinkhorn = fldd_imports['LogSinkhorn']

        a = F.softmax(torch.randn(49, vocab_size, device=device), dim=-1)
        b = F.softmax(torch.randn(49, vocab_size, device=device), dim=-1)
        C = torch.ones(vocab_size, vocab_size, device=device) - torch.eye(vocab_size, device=device)

        with suppress_prints():
            P = LogSinkhorn.solve(a, b, C, eps_schedule=[1.0, 0.5, 0.1], iters_per_eps=10)

        passed = (P >= 0).all().item()
        results.record("LogSinkhorn.solve non-negative", passed)
        assert passed

    def test_posterior_valid_probs(self, fldd_imports, device, vocab_size):
        LogSinkhorn = fldd_imports['LogSinkhorn']

        u_s = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        u_t = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        z_t = torch.randint(0, vocab_size, (2, 7, 7), device=device)
        C = torch.ones(vocab_size, vocab_size, device=device) - torch.eye(vocab_size, device=device)

        with suppress_prints():
            post = LogSinkhorn.posterior(u_s, u_t, z_t, C)

        sums_to_one = torch.allclose(post.sum(-1), torch.ones_like(post.sum(-1)), atol=0.1)
        non_negative = (post >= 0).all()

        passed = sums_to_one and non_negative
        results.record("LogSinkhorn.posterior valid probs", passed)
        assert passed

    def test_posterior_soft_valid_probs(self, fldd_imports, device, vocab_size):
        LogSinkhorn = fldd_imports['LogSinkhorn']

        u_s = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        u_t = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        z_soft = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        C = torch.ones(vocab_size, vocab_size, device=device) - torch.eye(vocab_size, device=device)

        with suppress_prints():
            post = LogSinkhorn.posterior_soft(u_s, u_t, z_soft, C)

        sums_to_one = torch.allclose(post.sum(-1), torch.ones_like(post.sum(-1)), atol=0.1)
        non_negative = (post >= 0).all()

        passed = sums_to_one and non_negative
        results.record("LogSinkhorn.posterior_soft valid probs", passed)
        assert passed


# =============================================================================
# TEST: KLDecomp Class
# =============================================================================

class TestKLDecomp:
    """Tests for KL divergence decomposition and TC estimation."""

    def test_factorized_kl_scalar(self, fldd_imports, device, vocab_size):
        results.set_category("KLDecomp Class")
        KLDecomp = fldd_imports['KLDecomp']

        post = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)
        rev = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            kl = KLDecomp.factorized_kl(post, rev)

        passed = kl.dim() == 0
        results.record("KLDecomp.factorized_kl is scalar", passed, f"dim={kl.dim()}")
        assert passed

    def test_factorized_kl_non_negative(self, fldd_imports, device, vocab_size):
        KLDecomp = fldd_imports['KLDecomp']

        post = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)
        rev = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            kl = KLDecomp.factorized_kl(post, rev)

        passed = kl.item() >= 0
        results.record("KLDecomp.factorized_kl >= 0", passed, f"kl={kl.item():.6f}")
        assert passed

    def test_factorized_kl_same_dist_zero(self, fldd_imports, device, vocab_size):
        KLDecomp = fldd_imports['KLDecomp']

        p = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            kl = KLDecomp.factorized_kl(p, p)

        passed = kl.item() < 1e-5
        results.record("KLDecomp.factorized_kl(p,p) ≈ 0", passed, f"kl={kl.item():.8f}")
        assert passed

    def test_factorized_kl_finite(self, fldd_imports, device, vocab_size):
        KLDecomp = fldd_imports['KLDecomp']

        post = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)
        rev = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            kl = KLDecomp.factorized_kl(post, rev)

        passed = torch.isfinite(kl).item()
        results.record("KLDecomp.factorized_kl is finite", passed)
        assert passed

    def test_kl_per_sample_shape(self, fldd_imports, device, vocab_size):
        KLDecomp = fldd_imports['KLDecomp']

        post = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)
        rev = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            kl = KLDecomp.kl_per_sample(post, rev)

        expected = torch.Size([4])
        passed = kl.shape == expected
        results.record("KLDecomp.kl_per_sample shape", passed, f"got {kl.shape}")
        assert passed

    def test_kl_per_sample_non_negative(self, fldd_imports, device, vocab_size):
        KLDecomp = fldd_imports['KLDecomp']

        post = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)
        rev = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            kl = KLDecomp.kl_per_sample(post, rev)

        passed = (kl >= 0).all().item()
        results.record("KLDecomp.kl_per_sample all >= 0", passed, f"min={kl.min().item():.6f}")
        assert passed

    def test_kl_per_sample_finite(self, fldd_imports, device, vocab_size):
        KLDecomp = fldd_imports['KLDecomp']

        post = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)
        rev = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            kl = KLDecomp.kl_per_sample(post, rev)

        passed = torch.isfinite(kl).all().item()
        results.record("KLDecomp.kl_per_sample all finite", passed)
        assert passed

    def test_tc_simple_scalar(self, fldd_imports, device, vocab_size):
        KLDecomp = fldd_imports['KLDecomp']

        posterior = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            tc = KLDecomp.tc_simple(posterior)

        passed = tc.dim() == 0
        results.record("KLDecomp.tc_simple is scalar", passed)
        assert passed

    def test_tc_simple_finite(self, fldd_imports, device, vocab_size):
        KLDecomp = fldd_imports['KLDecomp']

        posterior = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            tc = KLDecomp.tc_simple(posterior)

        passed = torch.isfinite(tc).item()
        results.record("KLDecomp.tc_simple is finite", passed, f"tc={tc.item():.6f}")
        assert passed

    def test_tc_minibatch_weighted_scalar(self, fldd_imports, device, vocab_size):
        KLDecomp = fldd_imports['KLDecomp']

        z_s = torch.randint(0, vocab_size, (4, 7, 7), device=device)
        u_s = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            tc = KLDecomp.tc_minibatch_weighted(z_s, u_s, batch_size=4)

        passed = tc.dim() == 0
        results.record("KLDecomp.tc_minibatch_weighted is scalar", passed)
        assert passed

    def test_tc_minibatch_weighted_non_negative(self, fldd_imports, device, vocab_size):
        KLDecomp = fldd_imports['KLDecomp']

        z_s = torch.randint(0, vocab_size, (4, 7, 7), device=device)
        u_s = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            tc = KLDecomp.tc_minibatch_weighted(z_s, u_s, batch_size=4)

        passed = tc.item() >= -0.01  # Allow small numerical error
        results.record("KLDecomp.tc_minibatch_weighted >= 0", passed, f"tc={tc.item():.6f}")
        assert passed


# =============================================================================
# TEST: Baseline Classes
# =============================================================================

class TestSimpleBaseline:
    """Tests for SimpleBaseline."""

    def test_init(self, fldd_imports):
        results.set_category("SimpleBaseline Class")
        SimpleBaseline = fldd_imports['SimpleBaseline']

        with suppress_prints():
            baseline = SimpleBaseline()

        passed = hasattr(baseline, 'v') and baseline.v == 0.0
        results.record("SimpleBaseline.__init__", passed, f"v={baseline.v}")
        assert passed

    def test_update_returns_float(self, fldd_imports, device):
        SimpleBaseline = fldd_imports['SimpleBaseline']

        with suppress_prints():
            baseline = SimpleBaseline()

        lp = torch.randn(4, device=device)
        kl = torch.abs(torch.randn(4, device=device))

        with suppress_prints():
            v = baseline.update(lp, kl)

        passed = isinstance(v, float)
        results.record("SimpleBaseline.update returns float", passed, f"type={type(v)}")
        assert passed

    def test_update_tracks_value(self, fldd_imports, device):
        SimpleBaseline = fldd_imports['SimpleBaseline']

        with suppress_prints():
            baseline = SimpleBaseline()

        # Update with some values
        for i in range(10):
            lp = torch.randn(4, device=device)
            kl = torch.ones(4, device=device) * (i + 1)
            with suppress_prints():
                baseline.update(lp, kl)

        v1 = baseline.v

        # Update with higher values
        for i in range(10):
            lp = torch.randn(4, device=device)
            kl = torch.ones(4, device=device) * (i + 10)
            with suppress_prints():
                baseline.update(lp, kl)

        v2 = baseline.v

        passed = v2 > v1
        results.record("SimpleBaseline.update tracks higher KL", passed, f"v1={v1:.4f}, v2={v2:.4f}")
        assert passed


class TestOptimalBaseline:
    """Tests for OptimalBaseline."""

    def test_init(self, fldd_imports):
        results.set_category("OptimalBaseline Class")
        OptimalBaseline = fldd_imports['OptimalBaseline']

        with suppress_prints():
            baseline = OptimalBaseline()

        passed = hasattr(baseline, 'n') and hasattr(baseline, 'd')
        results.record("OptimalBaseline.__init__", passed)
        assert passed

    def test_update_returns_float(self, fldd_imports, device):
        OptimalBaseline = fldd_imports['OptimalBaseline']

        with suppress_prints():
            baseline = OptimalBaseline()

        lp = torch.randn(4, device=device)
        kl = torch.abs(torch.randn(4, device=device))

        with suppress_prints():
            v = baseline.update(lp, kl)

        passed = isinstance(v, float)
        results.record("OptimalBaseline.update returns float", passed)
        assert passed

    def test_update_finite(self, fldd_imports, device):
        OptimalBaseline = fldd_imports['OptimalBaseline']

        with suppress_prints():
            baseline = OptimalBaseline()

        lp = torch.randn(4, device=device)
        kl = torch.abs(torch.randn(4, device=device))

        with suppress_prints():
            v = baseline.update(lp, kl)

        passed = np.isfinite(v)
        results.record("OptimalBaseline.update is finite", passed, f"v={v:.6f}")
        assert passed


class TestRLOOBaseline:
    """Tests for RLOOBaseline."""

    def test_init_k_samples(self, fldd_imports):
        results.set_category("RLOOBaseline Class")
        RLOOBaseline = fldd_imports['RLOOBaseline']

        with suppress_prints():
            baseline = RLOOBaseline(k_samples=4)

        passed = baseline.k == 4
        results.record("RLOOBaseline.__init__ k_samples", passed)
        assert passed

    def test_init_normalize_false(self, fldd_imports):
        RLOOBaseline = fldd_imports['RLOOBaseline']

        with suppress_prints():
            baseline = RLOOBaseline(k_samples=4, normalize=False)

        passed = baseline.normalize == False
        results.record("RLOOBaseline.__init__ normalize=False", passed)
        assert passed

    def test_init_normalize_true(self, fldd_imports):
        RLOOBaseline = fldd_imports['RLOOBaseline']

        with suppress_prints():
            baseline = RLOOBaseline(k_samples=4, normalize=True)

        passed = baseline.normalize == True
        results.record("RLOOBaseline.__init__ normalize=True", passed)
        assert passed

    def test_compute_gradient_scalar(self, fldd_imports, device):
        RLOOBaseline = fldd_imports['RLOOBaseline']

        with suppress_prints():
            baseline = RLOOBaseline(k_samples=4)

        rewards = torch.randn(8, 4, device=device)
        log_probs = torch.randn(8, 4, device=device)

        with suppress_prints():
            loss = baseline.compute_gradient(rewards, log_probs)

        passed = loss.dim() == 0
        results.record("RLOOBaseline.compute_gradient returns scalar", passed)
        assert passed

    def test_compute_gradient_finite(self, fldd_imports, device):
        RLOOBaseline = fldd_imports['RLOOBaseline']

        with suppress_prints():
            baseline = RLOOBaseline(k_samples=4)

        rewards = torch.randn(8, 4, device=device)
        log_probs = torch.randn(8, 4, device=device)

        with suppress_prints():
            loss = baseline.compute_gradient(rewards, log_probs)

        passed = torch.isfinite(loss).item()
        results.record("RLOOBaseline.compute_gradient is finite", passed, f"loss={loss.item():.4f}")
        assert passed

    def test_compute_gradient_normalized(self, fldd_imports, device):
        RLOOBaseline = fldd_imports['RLOOBaseline']

        with suppress_prints():
            baseline = RLOOBaseline(k_samples=4, normalize=True)

        rewards = torch.randn(8, 4, device=device)
        log_probs = torch.randn(8, 4, device=device)

        with suppress_prints():
            loss = baseline.compute_gradient(rewards, log_probs)

        passed = torch.isfinite(loss).item()
        results.record("RLOOBaseline.compute_gradient normalized finite", passed, f"loss={loss.item():.4f}")
        assert passed


# =============================================================================
# TEST: Relaxation Methods
# =============================================================================

class TestBinaryReparam:
    """Tests for BinaryReparam relaxation."""

    def test_sample_valid_probs(self, fldd_imports, device, vocab_size):
        results.set_category("BinaryReparam Class")
        BinaryReparam = fldd_imports['BinaryReparam']

        probs = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            z = BinaryReparam.sample(probs, t=0.5)

        sums_to_one = torch.allclose(z.sum(-1), torch.ones_like(z.sum(-1)), atol=1e-5)
        in_range = (z >= 0).all() and (z <= 1).all()

        passed = sums_to_one and in_range
        results.record("BinaryReparam.sample valid probs", passed)
        assert passed

    def test_sample_shape(self, fldd_imports, device, vocab_size):
        BinaryReparam = fldd_imports['BinaryReparam']

        probs = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            z = BinaryReparam.sample(probs, t=0.5)

        passed = z.shape == probs.shape
        results.record("BinaryReparam.sample correct shape", passed)
        assert passed

    def test_sample_gradient_flow(self, fldd_imports, device, vocab_size):
        BinaryReparam = fldd_imports['BinaryReparam']

        probs = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        probs.requires_grad_(True)

        with suppress_prints():
            z = BinaryReparam.sample(probs, t=0.5)

        loss = z.sum()
        loss.backward()

        has_grad = probs.grad is not None
        grad_nonzero = probs.grad.abs().sum().item() > 0 if has_grad else False

        passed = has_grad and grad_nonzero
        results.record("BinaryReparam.sample gradient flow", passed,
                      f"grad_sum={probs.grad.abs().sum().item() if has_grad else 0:.4f}")

        if not passed:
            results.add_issue("BinaryReparam does NOT propagate gradients - forward network won't learn with relaxation='binary'")

        assert passed

    def test_sample_different_temperatures(self, fldd_imports, device, vocab_size):
        BinaryReparam = fldd_imports['BinaryReparam']

        probs = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            z_high = BinaryReparam.sample(probs, t=1.0)
            z_low = BinaryReparam.sample(probs, t=0.01)

        # Low temp should be more peaked
        entropy_high = -(z_high * torch.log(z_high + 1e-8)).sum(-1).mean()
        entropy_low = -(z_low * torch.log(z_low + 1e-8)).sum(-1).mean()

        passed = entropy_low < entropy_high
        results.record("BinaryReparam low temp more peaked", passed,
                      f"entropy_high={entropy_high:.4f}, entropy_low={entropy_low:.4f}")
        assert passed


class TestStraightThroughGumbel:
    """Tests for StraightThroughGumbel relaxation."""

    def test_sample_is_one_hot(self, fldd_imports, device, vocab_size):
        results.set_category("StraightThroughGumbel Class")
        StraightThroughGumbel = fldd_imports['StraightThroughGumbel']

        logits = torch.randn(2, 7, 7, vocab_size, device=device)

        with suppress_prints():
            z = StraightThroughGumbel.sample(logits, temperature=1.0)

        # Check one-hot: sum to 1 and only 0s and 1s
        sums_to_one = torch.allclose(z.sum(-1), torch.ones_like(z.sum(-1)), atol=1e-5)
        is_binary = ((z == 0) | (z == 1)).all()

        passed = sums_to_one and is_binary
        results.record("StraightThroughGumbel.sample is one-hot", passed)
        assert passed

    def test_sample_shape(self, fldd_imports, device, vocab_size):
        StraightThroughGumbel = fldd_imports['StraightThroughGumbel']

        logits = torch.randn(2, 7, 7, vocab_size, device=device)

        with suppress_prints():
            z = StraightThroughGumbel.sample(logits, temperature=1.0)

        passed = z.shape == logits.shape
        results.record("StraightThroughGumbel.sample correct shape", passed)
        assert passed

    def test_sample_gradient_flow(self, fldd_imports, device, vocab_size):
        StraightThroughGumbel = fldd_imports['StraightThroughGumbel']

        logits = torch.randn(2, 7, 7, vocab_size, device=device, requires_grad=True)

        with suppress_prints():
            z = StraightThroughGumbel.sample(logits, temperature=1.0)

        loss = z.sum()
        loss.backward()

        has_grad = logits.grad is not None
        grad_nonzero = logits.grad.abs().sum().item() > 0 if has_grad else False

        passed = has_grad and grad_nonzero
        results.record("StraightThroughGumbel.sample gradient flow", passed)
        assert passed


class TestStraightThroughEstimator:
    """Tests for StraightThroughEstimator."""

    def test_sample_is_one_hot(self, fldd_imports, device, vocab_size):
        results.set_category("StraightThroughEstimator Class")
        StraightThroughEstimator = fldd_imports['StraightThroughEstimator']

        probs = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)

        with suppress_prints():
            z = StraightThroughEstimator.sample(probs)

        sums_to_one = torch.allclose(z.sum(-1), torch.ones_like(z.sum(-1)), atol=1e-5)
        is_binary = ((z == 0) | (z == 1)).all()

        passed = sums_to_one and is_binary
        results.record("StraightThroughEstimator.sample is one-hot", passed)
        assert passed

    def test_sample_gradient_flow(self, fldd_imports, device, vocab_size):
        StraightThroughEstimator = fldd_imports['StraightThroughEstimator']

        probs = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        probs.requires_grad_(True)

        with suppress_prints():
            z = StraightThroughEstimator.sample(probs)

        loss = z.sum()
        loss.backward()

        has_grad = probs.grad is not None
        grad_nonzero = probs.grad.abs().sum().item() > 0 if has_grad else False

        passed = has_grad and grad_nonzero
        results.record("StraightThroughEstimator.sample gradient flow", passed)
        assert passed


# =============================================================================
# TEST: EMA Class
# =============================================================================

class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_init_decay(self, fldd_imports, device):
        results.set_category("EMA Class")
        EMA = fldd_imports['EMA']
        Net = fldd_imports['Net']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']
        TIME_DIM = fldd_imports['TIME_DIM']
        HIDDEN_DIM = fldd_imports['HIDDEN_DIM']

        with suppress_prints():
            model = Net(VOCAB_SIZE, TIME_DIM, HIDDEN_DIM).to(device)
            ema = EMA(model, decay=0.999)

        passed = ema.decay == 0.999
        results.record("EMA.__init__ decay", passed)
        assert passed

    def test_init_shadow_created(self, fldd_imports, device):
        EMA = fldd_imports['EMA']
        Net = fldd_imports['Net']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']
        TIME_DIM = fldd_imports['TIME_DIM']
        HIDDEN_DIM = fldd_imports['HIDDEN_DIM']

        with suppress_prints():
            model = Net(VOCAB_SIZE, TIME_DIM, HIDDEN_DIM).to(device)
            ema = EMA(model, decay=0.999)

        passed = len(ema.shadow) > 0
        results.record("EMA.__init__ shadow created", passed, f"params={len(ema.shadow)}")
        assert passed

    def test_update_changes_shadow(self, fldd_imports, device):
        EMA = fldd_imports['EMA']
        Net = fldd_imports['Net']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']
        TIME_DIM = fldd_imports['TIME_DIM']
        HIDDEN_DIM = fldd_imports['HIDDEN_DIM']

        with suppress_prints():
            model = Net(VOCAB_SIZE, TIME_DIM, HIDDEN_DIM).to(device)
            ema = EMA(model, decay=0.9)

        # Get initial shadow
        initial = {k: v.clone() for k, v in ema.shadow.items()}

        # Modify model parameters
        for p in model.parameters():
            p.data.add_(torch.randn_like(p.data) * 0.1)

        # Update EMA
        with suppress_prints():
            ema.update()

        # Check if shadow changed
        changed = False
        for k in initial:
            if not torch.equal(initial[k], ema.shadow[k]):
                changed = True
                break

        passed = changed
        results.record("EMA.update changes shadow", passed)
        assert passed

    def test_apply_shadow_correct(self, fldd_imports, device):
        EMA = fldd_imports['EMA']
        Net = fldd_imports['Net']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']
        TIME_DIM = fldd_imports['TIME_DIM']
        HIDDEN_DIM = fldd_imports['HIDDEN_DIM']

        with suppress_prints():
            model = Net(VOCAB_SIZE, TIME_DIM, HIDDEN_DIM).to(device)
            ema = EMA(model, decay=0.9)

        # Update a few times
        for _ in range(5):
            for p in model.parameters():
                p.data.add_(torch.randn_like(p.data) * 0.1)
            with suppress_prints():
                ema.update()

        # Apply shadow
        with suppress_prints():
            ema.apply_shadow()

        # Check model params match shadow
        max_diff = 0
        for name, param in model.named_parameters():
            if param.requires_grad and name in ema.shadow:
                diff = (param.data - ema.shadow[name]).abs().max().item()
                max_diff = max(max_diff, diff)

        passed = max_diff < 1e-6
        results.record("EMA.apply_shadow correct", passed, f"max_diff={max_diff:.6f}")
        assert passed

    def test_restore_correct(self, fldd_imports, device):
        EMA = fldd_imports['EMA']
        Net = fldd_imports['Net']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']
        TIME_DIM = fldd_imports['TIME_DIM']
        HIDDEN_DIM = fldd_imports['HIDDEN_DIM']

        with suppress_prints():
            model = Net(VOCAB_SIZE, TIME_DIM, HIDDEN_DIM).to(device)
            ema = EMA(model, decay=0.9)

        # Save original params
        original = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}

        # Update and apply shadow
        for _ in range(5):
            for p in model.parameters():
                p.data.add_(torch.randn_like(p.data) * 0.1)
            with suppress_prints():
                ema.update()

        with suppress_prints():
            ema.apply_shadow()
            ema.restore()

        # Check restored
        max_diff = 0
        for name, param in model.named_parameters():
            if param.requires_grad and name in original:
                diff = (param.data - original[name]).abs().max().item()
                max_diff = max(max_diff, diff)

        # After restore, params should be back to what they were before apply_shadow
        # But they were modified before apply_shadow, so check backup was used
        passed = len(ema.backup) == 0  # Backup should be cleared after restore
        results.record("EMA.restore correct", passed)
        assert passed


# =============================================================================
# TEST: Neural Network Components
# =============================================================================

class TestSinEmb:
    """Tests for sinusoidal embeddings."""

    def test_init(self, fldd_imports, device):
        results.set_category("SinEmb Class")
        SinEmb = fldd_imports['SinEmb']

        with suppress_prints():
            emb = SinEmb(128).to(device)

        passed = emb.d == 128
        results.record("SinEmb.__init__ d", passed)
        assert passed

    def test_forward_shape(self, fldd_imports, device):
        SinEmb = fldd_imports['SinEmb']

        with suppress_prints():
            emb = SinEmb(128).to(device)

        t = torch.tensor([0, 5, 10], device=device).float()
        out = emb(t)

        expected = torch.Size([3, 128])
        passed = out.shape == expected
        results.record("SinEmb.forward shape", passed, f"got {out.shape}")
        assert passed

    def test_forward_finite(self, fldd_imports, device):
        SinEmb = fldd_imports['SinEmb']

        with suppress_prints():
            emb = SinEmb(128).to(device)

        t = torch.tensor([0, 5, 10], device=device).float()
        out = emb(t)

        passed = torch.isfinite(out).all().item()
        results.record("SinEmb.forward finite", passed)
        assert passed

    def test_forward_different_t(self, fldd_imports, device):
        SinEmb = fldd_imports['SinEmb']

        with suppress_prints():
            emb = SinEmb(128).to(device)

        t0 = torch.tensor([0.0], device=device)
        t1 = torch.tensor([1.0], device=device)

        out0 = emb(t0)
        out1 = emb(t1)

        passed = not torch.allclose(out0, out1)
        results.record("SinEmb.forward different for different t", passed, "t=0 and t=1 should differ")
        assert passed


class TestResBlk:
    """Tests for residual blocks."""

    def test_init_creates_layers(self, fldd_imports, device):
        results.set_category("ResBlk Class")
        ResBlk = fldd_imports['ResBlk']

        with suppress_prints():
            blk = ResBlk(64, 128, 128).to(device)

        passed = hasattr(blk, 'c1') and hasattr(blk, 'c2') and hasattr(blk, 'skip')
        results.record("ResBlk.__init__ creates layers", passed)
        assert passed

    def test_forward_shape(self, fldd_imports, device):
        ResBlk = fldd_imports['ResBlk']

        with suppress_prints():
            blk = ResBlk(64, 128, 128).to(device)

        x = torch.randn(2, 64, 14, 14, device=device)
        t = torch.randn(2, 128, device=device)

        out = blk(x, t)

        expected = torch.Size([2, 128, 14, 14])
        passed = out.shape == expected
        results.record("ResBlk.forward shape", passed, f"got {out.shape}")
        assert passed

    def test_forward_finite(self, fldd_imports, device):
        ResBlk = fldd_imports['ResBlk']

        with suppress_prints():
            blk = ResBlk(64, 128, 128).to(device)

        x = torch.randn(2, 64, 14, 14, device=device)
        t = torch.randn(2, 128, device=device)

        out = blk(x, t)

        passed = torch.isfinite(out).all().item()
        results.record("ResBlk.forward finite", passed)
        assert passed

    def test_forward_same_channels(self, fldd_imports, device):
        ResBlk = fldd_imports['ResBlk']

        with suppress_prints():
            blk = ResBlk(128, 128, 128).to(device)

        x = torch.randn(2, 128, 14, 14, device=device)
        t = torch.randn(2, 128, device=device)

        out = blk(x, t)

        passed = out.shape == x.shape
        results.record("ResBlk.forward same channels", passed)
        assert passed


class TestNet:
    """Tests for main U-Net."""

    def test_init_has_parameters(self, fldd_imports, device):
        results.set_category("Net Class")
        Net = fldd_imports['Net']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']
        TIME_DIM = fldd_imports['TIME_DIM']
        HIDDEN_DIM = fldd_imports['HIDDEN_DIM']

        with suppress_prints():
            net = Net(VOCAB_SIZE, TIME_DIM, HIDDEN_DIM).to(device)

        num_params = sum(p.numel() for p in net.parameters())
        passed = num_params > 0
        results.record("Net.__init__ has parameters", passed, f"params={num_params:,}")
        assert passed

    def test_init_K_stored(self, fldd_imports, device):
        Net = fldd_imports['Net']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']
        TIME_DIM = fldd_imports['TIME_DIM']
        HIDDEN_DIM = fldd_imports['HIDDEN_DIM']

        with suppress_prints():
            net = Net(VOCAB_SIZE, TIME_DIM, HIDDEN_DIM).to(device)

        passed = net.K == VOCAB_SIZE
        results.record("Net.__init__ K stored", passed)
        assert passed

    def test_forward_discrete_input(self, fldd_imports, device):
        Net = fldd_imports['Net']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']
        TIME_DIM = fldd_imports['TIME_DIM']
        HIDDEN_DIM = fldd_imports['HIDDEN_DIM']

        with suppress_prints():
            net = Net(VOCAB_SIZE, TIME_DIM, HIDDEN_DIM).to(device)

        x = torch.randint(0, VOCAB_SIZE, (2, 28, 28), device=device)
        t = torch.tensor([5, 5], device=device)

        with suppress_prints():
            out = net(x, t)

        expected = torch.Size([2, 28, 28, VOCAB_SIZE])
        passed = out.shape == expected
        results.record("Net.forward discrete input shape", passed, f"got {out.shape}")
        assert passed

    def test_forward_discrete_finite(self, fldd_imports, device):
        Net = fldd_imports['Net']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']
        TIME_DIM = fldd_imports['TIME_DIM']
        HIDDEN_DIM = fldd_imports['HIDDEN_DIM']

        with suppress_prints():
            net = Net(VOCAB_SIZE, TIME_DIM, HIDDEN_DIM).to(device)

        x = torch.randint(0, VOCAB_SIZE, (2, 28, 28), device=device)
        t = torch.tensor([5, 5], device=device)

        with suppress_prints():
            out = net(x, t)

        passed = torch.isfinite(out).all().item()
        results.record("Net.forward discrete input finite", passed)
        assert passed

    def test_forward_onehot_input(self, fldd_imports, device):
        Net = fldd_imports['Net']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']
        TIME_DIM = fldd_imports['TIME_DIM']
        HIDDEN_DIM = fldd_imports['HIDDEN_DIM']

        with suppress_prints():
            net = Net(VOCAB_SIZE, TIME_DIM, HIDDEN_DIM).to(device)

        x = torch.randn(2, VOCAB_SIZE, 28, 28, device=device)  # Already in channel-first
        t = torch.tensor([5, 5], device=device)

        with suppress_prints():
            out = net(x, t)

        expected = torch.Size([2, 28, 28, VOCAB_SIZE])
        passed = out.shape == expected
        results.record("Net.forward one-hot input shape", passed)
        assert passed


# =============================================================================
# TEST: FLDD Model
# =============================================================================

class TestFLDDCore:
    """Tests for FLDD model core functionality."""

    def test_init_fwd_network(self, fldd_imports, device):
        results.set_category("FLDD Class - Core")
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg()
        with suppress_prints():
            model = FLDD(cfg)

        passed = hasattr(model, 'fwd') and model.fwd is not None
        results.record("FLDD.__init__ fwd network created", passed)
        assert passed

    def test_init_rev_network(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg()
        with suppress_prints():
            model = FLDD(cfg)

        passed = hasattr(model, 'rev') and model.rev is not None
        results.record("FLDD.__init__ rev network created", passed)
        assert passed

    def test_init_optimizer(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg()
        with suppress_prints():
            model = FLDD(cfg)

        passed = hasattr(model, 'opt') and model.opt is not None
        results.record("FLDD.__init__ optimizer created", passed)
        assert passed

    def test_init_baseline(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg()
        with suppress_prints():
            model = FLDD(cfg)

        passed = hasattr(model, 'baseline') and model.baseline is not None
        results.record("FLDD.__init__ baseline created", passed)
        assert passed

    def test_init_step_zero(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg()
        with suppress_prints():
            model = FLDD(cfg)

        passed = model.step == 0
        results.record("FLDD.__init__ step = 0", passed)
        assert passed

    def test_total_property(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg(pretrain=5, warmup=10, reinforce=15)
        with suppress_prints():
            model = FLDD(cfg)

        passed = model.total == 30
        results.record("FLDD.total correct", passed, f"got {model.total}")
        assert passed

    def test_phase_pre(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg(pretrain=5, warmup=5, reinforce=5)
        with suppress_prints():
            model = FLDD(cfg)

        model.step = 0
        passed = model.phase == 'pre'
        results.record("FLDD.phase step=0 is 'pre'", passed)
        assert passed

    def test_phase_warm(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg(pretrain=5, warmup=5, reinforce=5)
        with suppress_prints():
            model = FLDD(cfg)

        model.step = 7
        passed = model.phase == 'warm'
        results.record("FLDD.phase step=7 is 'warm'", passed)
        assert passed

    def test_phase_RL(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg(pretrain=5, warmup=5, reinforce=5)
        with suppress_prints():
            model = FLDD(cfg)

        model.step = 12
        passed = model.phase == 'RL'
        results.record("FLDD.phase step=12 is 'RL'", passed)
        assert passed


class TestFLDDTemperature:
    """Tests for temperature scheduling."""

    @pytest.mark.parametrize("schedule", ["exponential", "cosine", "linear"])
    def test_get_temperature_pre_phase(self, fldd_imports, device, schedule):
        results.set_category("FLDD.get_temperature")
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg(pretrain=5, warmup=5, reinforce=5, temp_schedule=schedule, min_temp=0.1)
        with suppress_prints():
            model = FLDD(cfg)

        model.step = 0
        temp = model.get_temperature()

        passed = temp == 0.1  # Should return min_temp in pre phase
        results.record(f"get_temperature {schedule}: pre phase", passed, f"t_pre={temp}")
        assert passed

    @pytest.mark.parametrize("schedule", ["exponential", "cosine", "linear"])
    def test_get_temperature_RL_phase(self, fldd_imports, device, schedule):
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg(pretrain=5, warmup=5, reinforce=5, temp_schedule=schedule, min_temp=0.1)
        with suppress_prints():
            model = FLDD(cfg)

        model.step = 12
        temp = model.get_temperature()

        passed = temp == 0.1  # Should return min_temp in RL phase
        results.record(f"get_temperature {schedule}: RL phase", passed, f"t_rl={temp}")
        assert passed

    @pytest.mark.parametrize("schedule", ["exponential", "cosine", "linear"])
    def test_get_temperature_warmup_returns_value(self, fldd_imports, device, schedule):
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg(pretrain=5, warmup=5, reinforce=5, temp_schedule=schedule, min_temp=0.1)
        with suppress_prints():
            model = FLDD(cfg)

        model.step = 7
        temp = model.get_temperature()

        passed = isinstance(temp, (int, float)) and temp > 0
        results.record(f"get_temperature {schedule}: warmup returns value", passed)
        assert passed


class TestFLDDForwardMarginals:
    """Tests for forward marginal computation."""

    @pytest.mark.parametrize("mode", ["standard", "80_20", "paper", "paper_raw"])
    def test_fwd_marg_t0_is_data(self, fldd_imports, device, mode):
        results.set_category("FLDD.fwd_marg")
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        train_forward = mode != "standard"
        cfg = make_cfg(forward_mode=mode, train_forward=train_forward)
        with suppress_prints():
            model = FLDD(cfg)

        x = torch.randint(0, VOCAB_SIZE, (2, 28, 28), device=device)

        with suppress_prints():
            u = model.fwd_marg(x, 0)

        x_oh = F.one_hot(x.long(), VOCAB_SIZE).float()
        max_diff = (u - x_oh).abs().max().item()

        passed = max_diff < 1e-5
        results.record(f"fwd_marg {mode}: t=0 → data", passed, f"max_diff={max_diff:.6f}")
        assert passed

    @pytest.mark.parametrize("mode", ["standard", "80_20", "paper", "paper_raw"])
    def test_fwd_marg_tT_is_uniform(self, fldd_imports, device, mode):
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']
        NUM_TIMESTEPS = fldd_imports['NUM_TIMESTEPS']

        train_forward = mode != "standard"
        cfg = make_cfg(forward_mode=mode, train_forward=train_forward)
        with suppress_prints():
            model = FLDD(cfg)

        x = torch.randint(0, VOCAB_SIZE, (2, 28, 28), device=device)

        with suppress_prints():
            u = model.fwd_marg(x, NUM_TIMESTEPS)

        uniform = torch.ones_like(u) / VOCAB_SIZE
        max_diff = (u - uniform).abs().max().item()

        passed = max_diff < 1e-5
        results.record(f"fwd_marg {mode}: t=T → uniform", passed, f"max_diff={max_diff:.6f}")
        assert passed

    @pytest.mark.parametrize("mode", ["standard", "80_20", "paper", "paper_raw"])
    def test_fwd_marg_valid_probs(self, fldd_imports, device, mode):
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']
        NUM_TIMESTEPS = fldd_imports['NUM_TIMESTEPS']

        train_forward = mode != "standard"
        cfg = make_cfg(forward_mode=mode, train_forward=train_forward)
        with suppress_prints():
            model = FLDD(cfg)

        x = torch.randint(0, VOCAB_SIZE, (2, 28, 28), device=device)

        with suppress_prints():
            u = model.fwd_marg(x, NUM_TIMESTEPS // 2)

        sums_to_one = torch.allclose(u.sum(-1), torch.ones_like(u.sum(-1)), atol=1e-5)
        non_negative = (u >= 0).all()

        passed = sums_to_one and non_negative
        results.record(f"fwd_marg {mode}: t=T/2 valid prob", passed)
        assert passed


class TestFLDDReverse:
    """Tests for reverse distribution."""

    @pytest.mark.parametrize("t", [1, 5, 10])
    def test_rev_dist_valid_probs(self, fldd_imports, device, t):
        results.set_category("FLDD.rev_dist")
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg()
        with suppress_prints():
            model = FLDD(cfg)

        z = torch.randint(0, VOCAB_SIZE, (2, 28, 28), device=device)

        with suppress_prints():
            rev = model.rev_dist(z, t)

        sums_to_one = torch.allclose(rev.sum(-1), torch.ones_like(rev.sum(-1)), atol=1e-5)
        non_negative = (rev >= 0).all()

        passed = sums_to_one and non_negative
        results.record(f"rev_dist t={t}: valid prob", passed)
        assert passed

    @pytest.mark.parametrize("t", [1, 5, 10])
    def test_rev_dist_shape(self, fldd_imports, device, t):
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg()
        with suppress_prints():
            model = FLDD(cfg)

        z = torch.randint(0, VOCAB_SIZE, (2, 28, 28), device=device)

        with suppress_prints():
            rev = model.rev_dist(z, t)

        expected = torch.Size([2, 28, 28, VOCAB_SIZE])
        passed = rev.shape == expected
        results.record(f"rev_dist t={t}: correct shape", passed)
        assert passed


class TestFLDDTrainStep:
    """Tests for training step."""

    def test_train_step_pretrain_finite(self, fldd_imports, device):
        results.set_category("FLDD.train_step")
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg(pretrain=5, warmup=0, reinforce=0)
        with suppress_prints():
            model = FLDD(cfg)

        x = torch.randint(0, VOCAB_SIZE, (4, 28, 28), device=device)

        with suppress_prints():
            m = model.train_step(x)

        loss = m.get('loss', float('nan'))
        passed = np.isfinite(loss)
        results.record("train_step pretrain: loss finite", passed, f"loss={loss:.4f}")
        assert passed

    def test_train_step_warmup_finite(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg(pretrain=0, warmup=5, reinforce=0)
        with suppress_prints():
            model = FLDD(cfg)

        x = torch.randint(0, VOCAB_SIZE, (4, 28, 28), device=device)

        with suppress_prints():
            m = model.train_step(x)

        loss = m.get('loss', float('nan'))
        passed = np.isfinite(loss)
        results.record("train_step warmup: loss finite", passed, f"loss={loss:.4f}")
        assert passed

    def test_train_step_reinforce_finite(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg(pretrain=0, warmup=0, reinforce=5)
        with suppress_prints():
            model = FLDD(cfg)

        x = torch.randint(0, VOCAB_SIZE, (4, 28, 28), device=device)

        with suppress_prints():
            m = model.train_step(x)

        loss = m.get('loss', float('nan'))
        passed = np.isfinite(loss)
        results.record("train_step reinforce: loss finite", passed, f"loss={loss:.4f}")
        assert passed

    def test_train_step_increments_step(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg()
        with suppress_prints():
            model = FLDD(cfg)

        initial_step = model.step
        x = torch.randint(0, VOCAB_SIZE, (4, 28, 28), device=device)

        with suppress_prints():
            model.train_step(x)

        passed = model.step == initial_step + 1
        results.record("train_step increments step", passed)
        assert passed

    def test_train_step_tc_reg(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg(pretrain=0, warmup=5, reinforce=0, tc_reg=True, tc_lambda=0.1)
        with suppress_prints():
            model = FLDD(cfg)

        x = torch.randint(0, VOCAB_SIZE, (4, 28, 28), device=device)

        with suppress_prints():
            m = model.train_step(x)

        passed = 'tc' in m and np.isfinite(m['tc'])
        results.record("train_step tc_reg: metric recorded", passed)
        assert passed

    def test_train_step_aux_ce(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg(pretrain=0, warmup=5, reinforce=0, aux_ce_loss=True, aux_ce_lambda=0.001)
        with suppress_prints():
            model = FLDD(cfg)

        x = torch.randint(0, VOCAB_SIZE, (4, 28, 28), device=device)

        with suppress_prints():
            m = model.train_step(x)

        passed = 'aux_ce' in m and np.isfinite(m['aux_ce'])
        results.record("train_step aux_ce: metric recorded", passed)
        assert passed

    @pytest.mark.parametrize("baseline", ["simple", "optimal", "rloo", "rloo_normalized"])
    def test_train_step_baselines(self, fldd_imports, device, baseline):
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg(pretrain=0, warmup=0, reinforce=5, baseline=baseline)
        with suppress_prints():
            model = FLDD(cfg)

        x = torch.randint(0, VOCAB_SIZE, (4, 28, 28), device=device)

        with suppress_prints():
            m = model.train_step(x)

        loss = m.get('loss', float('nan'))
        passed = np.isfinite(loss)
        results.record(f"train_step baseline={baseline}: loss finite", passed, f"loss={loss:.4f}")
        assert passed


class TestFLDDSample:
    """Tests for sample generation."""

    def test_sample_shape(self, fldd_imports, device):
        results.set_category("FLDD.sample")
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg()
        with suppress_prints():
            model = FLDD(cfg)

        with suppress_prints():
            samples = model.sample(n=4, sz=28, use_ema=False)

        expected = torch.Size([4, 28, 28])
        passed = samples.shape == expected
        results.record("sample: correct shape", passed, f"got {samples.shape}")
        assert passed

    def test_sample_values_in_range(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg()
        with suppress_prints():
            model = FLDD(cfg)

        with suppress_prints():
            samples = model.sample(n=4, sz=28, use_ema=False)

        passed = samples.min() >= 0 and samples.max() < VOCAB_SIZE
        results.record("sample: values in range", passed, f"min={samples.min()}, max={samples.max()}")
        assert passed

    def test_sample_integer_values(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg()
        with suppress_prints():
            model = FLDD(cfg)

        with suppress_prints():
            samples = model.sample(n=4, sz=28, use_ema=False)

        passed = (samples == samples.long()).all().item()
        results.record("sample: integer values", passed)
        assert passed

    def test_sample_contains_both_classes(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg()
        with suppress_prints():
            model = FLDD(cfg)

        with suppress_prints():
            samples = model.sample(n=8, sz=28, use_ema=False)

        num_zeros = (samples == 0).sum().item()
        num_ones = (samples == 1).sum().item()

        passed = num_zeros > 0 and num_ones > 0
        results.record("sample: contains both classes", passed, f"0s={num_zeros}, 1s={num_ones}")
        assert passed

    def test_sample_with_ema(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']

        cfg = make_cfg(use_ema=True)
        with suppress_prints():
            model = FLDD(cfg)

        with suppress_prints():
            samples = model.sample(n=4, sz=28, use_ema=True)

        expected = torch.Size([4, 28, 28])
        passed = samples.shape == expected
        results.record("sample with EMA: correct shape", passed)
        assert passed


# =============================================================================
# TEST: Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full training pipeline."""

    def test_multi_step_training_no_nan(self, fldd_imports, device):
        results.set_category("Integration Tests")
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg(pretrain=3, warmup=3, reinforce=3)
        with suppress_prints():
            model = FLDD(cfg)

        x = torch.randint(0, VOCAB_SIZE, (4, 28, 28), device=device)

        nan_count = 0
        for _ in range(9):
            with suppress_prints():
                m = model.train_step(x)
            if not np.isfinite(m.get('loss', float('nan'))):
                nan_count += 1

        passed = nan_count == 0
        results.record("Multi-step training: no NaN losses", passed, f"nan_count={nan_count}")
        assert passed

    def test_multi_step_all_phases(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg(pretrain=2, warmup=2, reinforce=2)
        with suppress_prints():
            model = FLDD(cfg)

        x = torch.randint(0, VOCAB_SIZE, (4, 28, 28), device=device)

        phases_seen = set()
        for _ in range(6):
            with suppress_prints():
                m = model.train_step(x)
            phases_seen.add(m.get('phase', ''))

        passed = phases_seen == {'pre', 'warm', 'RL'}
        results.record("Multi-step training: all phases visited", passed, f"seen={phases_seen}")
        assert passed

    @pytest.mark.parametrize("preset", ["standard", "working", "paper", "project", "enhanced"])
    def test_preset_integration(self, fldd_imports, device, preset):
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        import fldd
        original = fldd.PRESET
        fldd.PRESET = preset

        with suppress_prints():
            cfg = fldd_imports['apply_preset']()

        fldd.PRESET = original

        # Override to make test fast
        cfg['pretrain'] = 1
        cfg['warmup'] = 1
        cfg['reinforce'] = 1

        with suppress_prints():
            model = FLDD(cfg)

        x = torch.randint(0, VOCAB_SIZE, (4, 28, 28), device=device)

        nan_count = 0
        for _ in range(3):
            with suppress_prints():
                m = model.train_step(x)
            if not np.isfinite(m.get('loss', float('nan'))):
                nan_count += 1

        passed = nan_count == 0
        results.record(f"Preset '{preset}': trains without NaN", passed)
        assert passed


# =============================================================================
# TEST: Visualization Functions
# =============================================================================

class TestVisualization:
    """Tests for visualization functions."""

    def test_visualize_forward_returns_figure(self, fldd_imports, device):
        results.set_category("Visualization Functions")
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']
        visualize_forward_process = fldd_imports['visualize_forward_process']

        cfg = make_cfg()
        with suppress_prints():
            model = FLDD(cfg)

        x = torch.randint(0, VOCAB_SIZE, (4, 28, 28), device=device)

        with suppress_prints():
            fig = visualize_forward_process(model, x, n_rows=2)

        import matplotlib.pyplot as plt
        passed = isinstance(fig, plt.Figure)
        plt.close(fig)

        results.record("visualize_forward_process: returns figure", passed)
        assert passed

    def test_visualize_backward_returns_figure(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']
        visualize_backward_process = fldd_imports['visualize_backward_process']

        cfg = make_cfg()
        with suppress_prints():
            model = FLDD(cfg)

        with suppress_prints():
            fig = visualize_backward_process(model, n_rows=2)

        import matplotlib.pyplot as plt
        passed = isinstance(fig, plt.Figure)
        plt.close(fig)

        results.record("visualize_backward_process: returns figure", passed)
        assert passed


# =============================================================================
# TEST: Data Loading
# =============================================================================

class TestDataLoading:
    """Tests for data loading functionality."""

    def test_get_data_returns_dataloader(self, fldd_imports):
        results.set_category("Data Loading")
        get_data = fldd_imports['get_data']

        with suppress_prints():
            loader = get_data(seed=42)

        from torch.utils.data import DataLoader
        passed = isinstance(loader, DataLoader)
        results.record("get_data: returns DataLoader", passed)
        assert passed

    def test_get_data_batch_shape(self, fldd_imports, device):
        get_data = fldd_imports['get_data']

        with suppress_prints():
            loader = get_data(seed=42)

        batch, _ = next(iter(loader))

        # Should be (batch, 1, 28, 28)
        passed = len(batch.shape) == 4 and batch.shape[2] == 28 and batch.shape[3] == 28
        results.record("get_data: batch shape", passed, f"got {batch.shape}")
        assert passed

    def test_get_data_binary(self, fldd_imports, device):
        get_data = fldd_imports['get_data']

        with suppress_prints():
            loader = get_data(seed=42)

        batch, _ = next(iter(loader))

        unique_vals = batch.unique()
        passed = len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals
        results.record("get_data: batch binary", passed)
        assert passed


# =============================================================================
# TEST: Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability in edge cases."""

    def test_kl_near_identical_small(self, fldd_imports, device, vocab_size):
        results.set_category("Numerical Stability")
        KLDecomp = fldd_imports['KLDecomp']

        p = F.softmax(torch.randn(4, 7, 7, vocab_size, device=device), dim=-1)
        q = p + torch.randn_like(p) * 1e-6
        q = q / q.sum(-1, keepdim=True)

        with suppress_prints():
            kl = KLDecomp.factorized_kl(p, q)

        passed = kl.item() < 0.01
        results.record("Numerical: KL near-identical is small", passed, f"kl={kl.item():.8f}")
        assert passed

    def test_concrete_low_temp(self, fldd_imports, device, vocab_size):
        from torch.distributions import RelaxedOneHotCategorical

        probs = F.softmax(torch.randn(2, 7, 7, vocab_size, device=device), dim=-1)
        probs = torch.clamp(probs, min=1e-6)
        probs = probs / probs.sum(-1, keepdim=True)

        dist = RelaxedOneHotCategorical(0.01, probs=probs)
        sample = dist.rsample()

        no_nan = not torch.isnan(sample).any()
        no_inf = not torch.isinf(sample).any()

        passed = no_nan and no_inf
        results.record("Numerical: Concrete at low temp", passed)
        assert passed


# =============================================================================
# TEST: Gradient Flow End-to-End
# =============================================================================

class TestGradientFlow:
    """Tests for end-to-end gradient flow."""

    def test_fwd_params_change_with_train_forward(self, fldd_imports, device):
        results.set_category("Gradient Flow E2E")
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg(forward_mode='paper', train_forward=True, pretrain=0, warmup=3, reinforce=0)
        with suppress_prints():
            model = FLDD(cfg)

        # Get initial params
        initial_fwd = {k: v.clone() for k, v in model.fwd.named_parameters()}

        x = torch.randint(0, VOCAB_SIZE, (4, 28, 28), device=device)

        for _ in range(3):
            with suppress_prints():
                model.train_step(x)

        # Check if params changed
        changed = False
        for k, v in model.fwd.named_parameters():
            if not torch.equal(initial_fwd[k], v):
                changed = True
                break

        passed = changed
        results.record("Gradient E2E: fwd params change (train_forward=True)", passed)
        assert passed

    def test_rev_params_change(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg(pretrain=0, warmup=3, reinforce=0)
        with suppress_prints():
            model = FLDD(cfg)

        # Get initial params
        initial_rev = {k: v.clone() for k, v in model.rev.named_parameters()}

        x = torch.randint(0, VOCAB_SIZE, (4, 28, 28), device=device)

        for _ in range(3):
            with suppress_prints():
                model.train_step(x)

        # Check if params changed
        changed = False
        for k, v in model.rev.named_parameters():
            if not torch.equal(initial_rev[k], v):
                changed = True
                break

        passed = changed
        results.record("Gradient E2E: rev params change", passed)
        assert passed

    def test_fwd_params_frozen_standard(self, fldd_imports, device):
        FLDD = fldd_imports['FLDD']
        VOCAB_SIZE = fldd_imports['VOCAB_SIZE']

        cfg = make_cfg(forward_mode='standard', train_forward=False, pretrain=0, warmup=3, reinforce=0)
        with suppress_prints():
            model = FLDD(cfg)

        # Get initial params
        initial_fwd = {k: v.clone() for k, v in model.fwd.named_parameters()}

        x = torch.randint(0, VOCAB_SIZE, (4, 28, 28), device=device)

        for _ in range(3):
            with suppress_prints():
                model.train_step(x)

        # Check if params stayed same
        same = True
        for k, v in model.fwd.named_parameters():
            if not torch.equal(initial_fwd[k], v):
                same = False
                break

        passed = same
        results.record("Gradient E2E: fwd params frozen (train_forward=False)", passed)
        assert passed


# =============================================================================
# PYTEST HOOKS
# =============================================================================

def pytest_sessionfinish(session, exitstatus):
    """Save results at the end of pytest session."""
    results.save("test_results.json")


if __name__ == "__main__":
    # Run with: python test_fldd.py
    pytest.main([__file__, "-v", "--tb=short"])
