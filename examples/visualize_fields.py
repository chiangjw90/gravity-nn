"""
Visualize Gravity Field Data
=============================

Extract and plot the physics quantities from each Gravity layer:
  - Density (ρ): scalar compression of input representations
  - Field Potential (φ): multi-scale causal field from density
  - Field Gradient (∇φ): rate of change of the field
  - Attention Weights: local window weights derived from field quantities

Usage:
    # With the pre-trained demo checkpoint:
    python examples/visualize_fields.py

    # With your own checkpoint:
    python examples/visualize_fields.py --checkpoint my_model.pt

    # Custom prompt:
    python examples/visualize_fields.py --prompt "The quick brown fox"

    # Save plots to files instead of showing:
    python examples/visualize_fields.py --save_dir field_plots/

Requirements:
    pip install matplotlib tiktoken
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import torch
import torch.nn.functional as F

from gravity import GravityLM


def extract_field_data(model, token_ids, device="cpu"):
    """Run a forward pass and extract field data from every layer.

    Returns a list of dicts (one per layer), each containing:
        - coins:    [B, N, K]     coin parameters
        - density:  [B, N, C]     density field (ρ)
        - phi:      [B, S, C, N]  field potential
        - gradient: [B, S, C, N]  field gradient
        - attention:[B, S, N, W]  attention weights
    """
    model.eval()
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    layer_data = []

    with torch.no_grad():
        # Embedding
        B, N = ids.shape
        h = model.drop(
            model.tok(ids)
            + model.pos(torch.arange(N, device=device).unsqueeze(0))
        )

        # Each block
        for i, block in enumerate(model.blocks):
            normed = block.norm1(h)
            coins = math.pi * torch.tanh(block.coin_proj(normed))
            rho = block.attn.compute_density(coins)
            phi, gp = block.solver(rho)
            at = block.attn(coins, phi, gp)

            layer_data.append({
                "layer": i,
                "coins": coins.cpu(),
                "density": rho.cpu(),
                "phi": phi.cpu(),
                "gradient": gp.cpu(),
                "attention": at.cpu(),
            })

            # Continue forward pass for next layer
            ci = getattr(block.attn, '_ci', None)
            vm = getattr(block.attn, '_vm', None)
            f = block.feat(coins, at, rho, phi, gp, ci, vm)
            h = h + block.feat_proj(f)
            h = h + block.ffn(block.norm2(h))

    return layer_data


def plot_field_data(layer_data, tokens, save_dir=None):
    """Plot field data for each layer."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.size'] = 9

    n_layers = len(layer_data)

    for li, data in enumerate(layer_data):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f"Gravity Field Data — Layer {li}", fontsize=14, fontweight="bold")

        N = data["density"].shape[1]
        x_positions = range(N)

        # Truncate token labels for readability
        token_labels = [t[:8] for t in tokens[:N]]
        tick_step = max(1, N // 20)
        tick_pos = list(range(0, N, tick_step))
        tick_labels = [token_labels[i] for i in tick_pos]

        # --- 1. Density heatmap ---
        ax = axes[0, 0]
        rho = data["density"][0].T.numpy()  # [C, N]
        im = ax.imshow(rho, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_title("Density ρ(t)  — compressed scalar field")
        ax.set_ylabel("Channel")
        ax.set_xlabel("Token position")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)

        # --- 2. Field Potential (φ) per scale ---
        ax = axes[0, 1]
        phi = data["phi"][0]  # [S, C, N]
        S = phi.shape[0]
        # Average over channels for visualization
        phi_avg = phi.mean(dim=1).numpy()  # [S, N]
        for s in range(S):
            ax.plot(x_positions, phi_avg[s], label=f"Scale {s}", alpha=0.8, linewidth=1.2)
        ax.set_title("Field Potential φ(t)  — multi-scale causal field")
        ax.set_xlabel("Token position")
        ax.set_ylabel("φ (channel-averaged)")
        ax.legend(fontsize=8)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

        # --- 3. Field Gradient (∇φ) per scale ---
        ax = axes[1, 0]
        gp = data["gradient"][0]  # [S, C, N]
        gp_avg = gp.mean(dim=1).numpy()  # [S, N]
        for s in range(S):
            ax.plot(x_positions, gp_avg[s], label=f"Scale {s}", alpha=0.8, linewidth=1.2)
        ax.set_title("Field Gradient ∇φ(t)  — rate of change")
        ax.set_xlabel("Token position")
        ax.set_ylabel("∇φ (channel-averaged)")
        ax.legend(fontsize=8)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

        # --- 4. Attention weights (scale 0) ---
        ax = axes[1, 1]
        attn = data["attention"][0, 0].numpy()  # [N, W]
        im = ax.imshow(attn.T, aspect="auto", cmap="Blues", interpolation="nearest",
                       origin="lower")
        ax.set_title("Attention Weights (Scale 0)  — local causal window")
        ax.set_xlabel("Token position (query)")
        ax.set_ylabel("Window offset (past → present)")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"field_layer_{li}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
            plt.close()
        else:
            plt.show()


def print_field_summary(layer_data, tokens):
    """Print a text summary of field statistics (no matplotlib needed)."""
    N = layer_data[0]["density"].shape[1]
    print(f"\nInput: {N} tokens")
    print(f"Tokens: {' '.join(tokens[:20])}{'...' if len(tokens) > 20 else ''}")
    print()

    for data in layer_data:
        li = data["layer"]
        rho = data["density"][0]      # [N, C]
        phi = data["phi"][0]          # [S, C, N]
        gp = data["gradient"][0]      # [S, C, N]
        attn = data["attention"][0]   # [S, N, W]

        print(f"  Layer {li}:")
        print(f"    Density ρ     — shape {list(rho.shape)}, "
              f"mean={rho.mean():.4f}, std={rho.std():.4f}, "
              f"max={rho.max():.4f}")
        print(f"    Potential φ   — shape {list(phi.shape)}, "
              f"mean={phi.mean():.4f}, std={phi.std():.4f}")
        print(f"    Gradient ∇φ   — shape {list(gp.shape)}, "
              f"mean={gp.mean():.4f}, std={gp.std():.4f}")
        print(f"    Attention     — shape {list(attn.shape)}, "
              f"entropy={-(attn * attn.clamp(min=1e-8).log()).sum(-1).mean():.3f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Visualize Gravity field data")
    parser.add_argument("--checkpoint", type=str, default="gravity_demo.pt",
                        help="Path to checkpoint (default: gravity_demo.pt)")
    parser.add_argument("--prompt", type=str,
                        default="The theory of gravity explains how objects interact through fields",
                        help="Input text to process")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Save plots to this directory (default: show interactively)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Print text summary only, no matplotlib needed")
    args = parser.parse_args()

    # --- Tokenize ---
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        token_ids = enc.encode(args.prompt)
        tokens = [enc.decode([tid]) for tid in token_ids]
        vocab_size = enc.n_vocab
    except ImportError:
        print("tiktoken not installed, using character-level tokenization")
        token_ids = [ord(c) % 256 for c in args.prompt]
        tokens = list(args.prompt)
        vocab_size = 256

    print(f"Prompt: \"{args.prompt}\"")
    print(f"Tokens: {len(token_ids)}")

    # --- Load model ---

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "config" in ckpt:
            cfg = ckpt["config"]
            model = GravityLM(
                vocab_size=ckpt.get("vocab_size", vocab_size),
                max_seq_len=cfg.get("seq_len", 1024),
                d_model=cfg["d_model"],
                n_layers=cfg["n_layers"],
                K=cfg.get("K", 64),
                S=cfg.get("S", 3),
                C=cfg.get("C", 16),
                R=cfg.get("R", 5),
            )
            model.load_state_dict(ckpt["model_state"])
            print(f"Loaded trained model: {sum(p.numel() for p in model.parameters()):,} params")
        else:
            # Raw state dict
            model = GravityLM(vocab_size=vocab_size, max_seq_len=1024)
            model.load_state_dict(ckpt if not isinstance(ckpt, dict) else ckpt)
            print("Loaded raw state dict")
    else:
        print(f"No checkpoint found at '{args.checkpoint}', using random weights")
        model = GravityLM(
            vocab_size=vocab_size, max_seq_len=1024,
            d_model=128, n_layers=2, K=64, S=3, C=16, R=5,
        )
        print(f"Random model: {sum(p.numel() for p in model.parameters()):,} params")

    # --- Extract field data ---
    print("\nExtracting field data...")
    layer_data = extract_field_data(model, token_ids)

    # --- Output ---
    print_field_summary(layer_data, tokens)

    if not args.no_plot:
        try:
            plot_field_data(layer_data, tokens, save_dir=args.save_dir)
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            print("Or use --no_plot for text-only output.")


if __name__ == "__main__":
    main()
