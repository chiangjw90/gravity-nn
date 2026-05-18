"""
Train a Gravity Demo Model
===========================

Train a small Gravity model on WikiText-103 for the HuggingFace demo.
Designed to run on Apple Silicon (M4 Pro / MPS).

Usage:
    cd gravity-nn
    source .venv/bin/activate
    pip install tiktoken datasets
    python examples/train_demo_model.py

Expected:
    ~15M parameters, ~2-4 hours on M4 Pro
    Output: gravity_demo.pt (checkpoint for HuggingFace Space)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time
import json
import torch
import torch.nn.functional as F


# =============================================================================
# Config
# =============================================================================

CONFIG = {
    # Model
    "d_model": 256,
    "n_layers": 4,
    "K": 64,
    "S": 3,
    "C": 16,
    "R": 5,
    "dropout": 0.1,
    "seq_len": 1024,

    # Training
    "batch_size": 8,
    "lr": 3e-4,
    "weight_decay": 0.1,
    "epochs": 3,
    "grad_clip": 1.0,
    "log_interval": 50,

    # Data
    "dataset": "wikitext-103-raw-v1",
    "tokenizer": "gpt2",
    "max_train_tokens": 50_000_000,  # 50M tokens — enough for a demo

    # Output
    "save_path": "gravity_demo.pt",
}


# =============================================================================
# Data loading
# =============================================================================

def load_data(config):
    """Download WikiText-103 and tokenize."""
    from datasets import load_dataset
    import tiktoken

    print("Loading WikiText-103...")
    ds = load_dataset("wikitext", config["dataset"], split="train")

    print("Tokenizing...")
    enc = tiktoken.get_encoding(config["tokenizer"])
    vocab_size = enc.n_vocab

    all_ids = []
    for example in ds:
        text = example["text"]
        if text.strip():
            all_ids.extend(enc.encode(text))
        if len(all_ids) >= config["max_train_tokens"]:
            all_ids = all_ids[:config["max_train_tokens"]]
            break

    print(f"Total tokens: {len(all_ids):,} (vocab: {vocab_size:,})")

    # Build chunks
    seq_len = config["seq_len"]
    chunk_size = seq_len + 1
    n_chunks = len(all_ids) // chunk_size
    data = torch.tensor(all_ids[:n_chunks * chunk_size], dtype=torch.long)
    data = data.view(n_chunks, chunk_size)

    # Train/val split (95/5)
    n_val = max(1, int(n_chunks * 0.05))
    train_data = data[:-n_val]
    val_data = data[-n_val:]

    print(f"Train: {train_data.shape[0]:,} chunks, Val: {val_data.shape[0]:,} chunks")

    # Also load test split for final eval
    ds_test = load_dataset("wikitext", config["dataset"], split="test")
    test_ids = []
    for example in ds_test:
        text = example["text"]
        if text.strip():
            test_ids.extend(enc.encode(text))
    n_test = len(test_ids) // chunk_size
    test_data = torch.tensor(
        test_ids[:n_test * chunk_size], dtype=torch.long
    ).view(n_test, chunk_size)
    print(f"Test: {test_data.shape[0]:,} chunks")

    return train_data, val_data, test_data, vocab_size, enc


# =============================================================================
# Evaluate
# =============================================================================

@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size].to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    ppl = math.exp(min(total_loss / total_tokens, 20))
    model.train()
    return ppl


# =============================================================================
# Memory comparison
# =============================================================================

def memory_comparison(vocab_size, config, device):
    """Show memory usage: Gravity vs Transformer at various context lengths."""
    from gravity import GravityLM, TransformerLM

    print("\n" + "=" * 60)
    print("  Memory Comparison: Gravity vs Transformer")
    print("=" * 60)

    for ctx_len in [512, 1024, 2048, 4096, 8192]:
        results = {}
        for name, ModelClass in [("Gravity", GravityLM), ("Transformer", TransformerLM)]:
            try:
                torch.mps.empty_cache() if device == "mps" else None
                model = ModelClass(
                    vocab_size=vocab_size,
                    max_seq_len=ctx_len,
                    d_model=config["d_model"],
                    n_layers=config["n_layers"],
                ).to(device)
                ids = torch.randint(0, vocab_size, (1, ctx_len), device=device)

                if device == "mps":
                    torch.mps.reset_peak_memory_stats()
                    with torch.no_grad():
                        _ = model(ids)
                    mem_mb = torch.mps.max_memory_allocated() / 1024 / 1024
                else:
                    torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad():
                        _ = model(ids)
                    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

                results[name] = f"{mem_mb:.0f} MB"
                del model, ids
            except Exception as e:
                results[name] = f"OOM/Error"

        g = results.get("Gravity", "?")
        t = results.get("Transformer", "?")
        print(f"  ctx={ctx_len:5d}:  Gravity={g:>8s}  Transformer={t:>8s}")

    print()


# =============================================================================
# Train
# =============================================================================

def train(config):
    from gravity import GravityLM

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Data
    train_data, val_data, test_data, vocab_size, enc = load_data(config)

    # Model
    model = GravityLM(
        vocab_size=vocab_size,
        max_seq_len=config["seq_len"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        K=config["K"],
        S=config["S"],
        C=config["C"],
        R=config["R"],
        dropout=config["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nGravity model: {n_params:,} parameters")
    print(f"Inference state: {config['S'] * config['C'] * 4} bytes/layer\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    best_val_ppl = float("inf")
    t0 = time.time()

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0
        total_tokens = 0
        perm = torch.randperm(len(train_data))
        train_shuffled = train_data[perm]

        for i in range(0, len(train_shuffled), config["batch_size"]):
            batch = train_shuffled[i:i + config["batch_size"]].to(device)
            x, y = batch[:, :-1], batch[:, 1:]

            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), y.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            step = i // config["batch_size"] + 1
            if step % config["log_interval"] == 0:
                avg_ppl = math.exp(min(total_loss / total_tokens, 20))
                elapsed = (time.time() - t0) / 60
                print(f"  Epoch {epoch} step {step:,} | "
                      f"PPL: {avg_ppl:.1f} | "
                      f"Time: {elapsed:.1f}m", end="\r")

        train_ppl = math.exp(min(total_loss / total_tokens, 20))
        val_ppl = evaluate(model, val_data, config["batch_size"], device)
        elapsed = (time.time() - t0) / 60

        print(f"\n  Epoch {epoch}/{config['epochs']} | "
              f"Train PPL: {train_ppl:.1f} | Val PPL: {val_ppl:.1f} | "
              f"Time: {elapsed:.1f}m")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            # Save checkpoint
            checkpoint = {
                "model_state": model.state_dict(),
                "config": config,
                "vocab_size": vocab_size,
                "val_ppl": val_ppl,
                "epoch": epoch,
            }
            torch.save(checkpoint, config["save_path"])
            print(f"  → Saved (Val PPL: {val_ppl:.1f})")

    # Final test evaluation
    test_ppl = evaluate(model, test_data, config["batch_size"], device)
    total_time = (time.time() - t0) / 60
    print(f"\nTraining done! ({total_time:.1f} min)")
    print(f"  Best Val PPL: {best_val_ppl:.1f}")
    print(f"  Test PPL: {test_ppl:.1f}")
    print(f"  Checkpoint: {config['save_path']}")

    # Memory comparison
    memory_comparison(vocab_size, config, device)

    # Quick generation test
    print("=" * 60)
    print("  Sample Generation")
    print("=" * 60)
    model.eval()
    prompt = "The meaning of life is"
    prompt_ids = enc.encode(prompt)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(100):
            logits = model(ids)[:, -1, :]
            logits = logits / 0.8  # temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)

    generated = enc.decode(ids[0].tolist())
    print(f"\n  Prompt: {prompt}")
    print(f"  Output: {generated}\n")


if __name__ == "__main__":
    train(CONFIG)
