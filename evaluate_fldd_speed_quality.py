"""
FLDD Speed-Quality Evaluation

Evaluates FLDD's claim: "FLDD at low T achieves quality comparable to Standard at high T"
Speed = 1/T, so if FLDD@T=10 matches Standard@T=50, FLDD is 5x faster.

Usage: python evaluate_fldd_speed_quality.py
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import time
import json
import os

# Import from fldd.py - use existing presets
import fldd
from fldd import FLDD, get_data, device, VOCAB_SIZE

# =============================================================================
# CONFIGURATION
# =============================================================================

T_VALUES = [4, 10, 20, 50]       # Timesteps to test
TRAINING_STEPS = 50000           # Paper uses 200k, adjust as needed
EVAL_SAMPLES = 256
BATCH_SIZE = 64
OUTPUT_DIR = './outputs/speed_quality_eval'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config(preset_name, T, training_steps):
    """Get config from fldd.py preset, override T and training steps."""
    # Temporarily set the preset
    original_preset = fldd.PRESET
    fldd.PRESET = preset_name

    # Get base config from preset
    cfg = fldd.apply_preset()

    # Restore original
    fldd.PRESET = original_preset

    # Override for this evaluation
    cfg['num_timesteps'] = T
    cfg['preset'] = f'{preset_name}_T{T}'

    # Scale training steps
    if preset_name == 'standard':
        cfg['reinforce'] = training_steps
    else:
        # For paper/project: split between warmup and reinforce
        cfg['pretrain'] = min(1000, training_steps // 20)
        cfg['warmup'] = training_steps // 2
        cfg['reinforce'] = training_steps - cfg['warmup'] - cfg['pretrain']

    return cfg


def compute_quality(samples, target_ones_ratio=0.15):
    """Compute sample quality metrics."""
    samples_np = samples.cpu().numpy()
    n = samples_np.shape[0]

    ones_ratio = float(np.mean(samples_np))
    pixel_variance = float(np.var(samples_np, axis=0).mean())
    unique = len(np.unique(samples_np.reshape(n, -1), axis=0))
    diversity = unique / n

    # Combined score (lower is better)
    quality_score = abs(ones_ratio - target_ones_ratio) + max(0, 0.1 - pixel_variance) + (1 - diversity)

    return {
        'ones_ratio': ones_ratio,
        'pixel_variance': pixel_variance,
        'diversity': diversity,
        'quality_score': quality_score,
    }


def measure_inference_time(model, T, n_samples=64, n_runs=3):
    """Measure inference time."""
    model.rev.eval()
    times = []

    for _ in range(n_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            z = torch.randint(0, VOCAB_SIZE, (n_samples, 28, 28), device=device)
            for t in reversed(range(1, T + 1)):
                rev_p = model.rev_dist(z, t)
                z = Categorical(probs=rev_p).sample()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)

    model.rev.train()
    return {'mean_time': np.mean(times), 'time_per_sample': np.mean(times) / n_samples}


def train_and_evaluate(preset_name, T, loader, training_steps):
    """Train model and evaluate quality."""
    print(f"\n{'='*60}")
    print(f"Training: {preset_name} @ T={T}")
    print(f"{'='*60}")

    # Get config using preset
    cfg = get_config(preset_name, T, training_steps)

    # Set global T for this run
    original_T = fldd.NUM_TIMESTEPS
    fldd.NUM_TIMESTEPS = T

    try:
        model = FLDD(cfg)
        data_iter = cycle(loader)
        losses = []
        start_time = time.time()

        total_steps = cfg['pretrain'] + cfg['warmup'] + cfg['reinforce']

        for i in range(total_steps):
            x, _ = next(data_iter)
            x = x.squeeze(1).to(device)
            m = model.train_step(x)
            losses.append(m.get('loss', 0))

            if (i + 1) % 5000 == 0:
                print(f"  Step {i+1}/{total_steps} | Loss: {np.mean(losses[-1000:]):.4f} | Phase: {m.get('phase', '?')}")

        training_time = time.time() - start_time

        # Generate samples
        print(f"  Generating {EVAL_SAMPLES} samples...")
        model.rev.eval()
        all_samples = []
        with torch.no_grad():
            for _ in range(EVAL_SAMPLES // BATCH_SIZE):
                all_samples.append(model.sample(BATCH_SIZE))
        all_samples = torch.cat(all_samples, dim=0)[:EVAL_SAMPLES]

        # Evaluate
        quality = compute_quality(all_samples)
        inference = measure_inference_time(model, T)

        print(f"  Quality: {quality['quality_score']:.4f} | ones: {quality['ones_ratio']:.3f} | diversity: {quality['diversity']:.3f}")
        print(f"  Inference: {inference['mean_time']:.3f}s for 64 samples")

        # Save samples
        fig, axes = plt.subplots(4, 8, figsize=(10, 5))
        for i, ax in enumerate(axes.flat):
            ax.imshow(all_samples[i].cpu().numpy(), cmap='gray')
            ax.axis('off')
        plt.suptitle(f"{preset_name} @ T={T} | Quality: {quality['quality_score']:.3f}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{preset_name}_T{T}.png", dpi=150)
        plt.close()

        return {
            'preset': preset_name, 'T': T,
            'training_time': training_time,
            'final_loss': np.mean(losses[-100:]),
            **quality, **inference
        }

    finally:
        fldd.NUM_TIMESTEPS = original_T


def run_evaluation():
    """Run full evaluation."""
    print("="*70)
    print("FLDD SPEED-QUALITY EVALUATION")
    print("="*70)
    print(f"Device: {device}")
    print(f"T values: {T_VALUES}")
    print(f"Training steps: {TRAINING_STEPS}")

    loader = get_data(seed=42)
    results = []

    # Test both presets at each T
    for preset in ['standard', 'paper']:
        for T in T_VALUES:
            result = train_and_evaluate(preset, T, loader, TRAINING_STEPS)
            results.append(result)

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Preset':<12} {'T':<6} {'Quality':<10} {'Diversity':<10} {'Time (s)':<10}")
    print("-"*50)

    for r in sorted(results, key=lambda x: (x['preset'], x['T'])):
        print(f"{r['preset']:<12} {r['T']:<6} {r['quality_score']:<10.4f} {r['diversity']:<10.3f} {r['mean_time']:<10.3f}")

    # Save results
    with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for preset in ['standard', 'paper']:
        data = sorted([r for r in results if r['preset'] == preset], key=lambda x: x['T'])
        Ts = [r['T'] for r in data]
        quality = [r['quality_score'] for r in data]
        times = [r['mean_time'] for r in data]

        axes[0].plot(Ts, quality, 'o-', label=preset, linewidth=2, markersize=8)
        axes[1].plot(times, quality, 'o-', label=preset, linewidth=2, markersize=8)

    axes[0].set_xlabel('Timesteps (T)')
    axes[0].set_ylabel('Quality Score (lower=better)')
    axes[0].set_title('Quality vs Steps')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Inference Time (s)')
    axes[1].set_ylabel('Quality Score (lower=better)')
    axes[1].set_title('Quality vs Speed (key plot!)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {OUTPUT_DIR}/")
    return results


if __name__ == '__main__':
    run_evaluation()
