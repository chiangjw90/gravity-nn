"""
Train Gravity on Your Own Data
==============================

This example shows how to train a Gravity language model on any text data.

Usage:
    # Train on a single text file
    python examples/train_custom.py --data my_data.txt

    # Train on a folder of .txt files
    python examples/train_custom.py --data my_data_folder/

    # Larger model, more epochs
    python examples/train_custom.py --data my_data.txt --d_model 768 --n_layers 12 --epochs 10

Requirements:
    pip install -e .   # from repo root
    pip install tiktoken
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import argparse
import torch
import torch.nn.functional as F


# =============================================================================
# 1. Load your text data
# =============================================================================

def load_texts(path):
    """Load text from a file or folder of .txt files."""
    texts = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    elif os.path.isdir(path):
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".txt"):
                with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                    texts.append(f.read())
    else:
        raise ValueError(f"Path not found: {path}")
    print(f"Loaded {len(texts)} file(s), "
          f"{sum(len(t) for t in texts):,} characters total")
    return texts


# =============================================================================
# 2. Tokenize
# =============================================================================

def tokenize(texts, tokenizer="gpt2"):
    """
    Tokenize texts using tiktoken (GPT-2 tokenizer by default).
    Returns a flat list of token ids and the vocab size.

    To use a different tokenizer, replace this function with your own.
    For example, with HuggingFace:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        ids = tok.encode(text)
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding(tokenizer)
        vocab_size = enc.n_vocab
    except ImportError:
        raise ImportError(
            "tiktoken not installed. Run: pip install tiktoken\n"
            "Or replace the tokenize() function with your own tokenizer."
        )

    all_ids = []
    for text in texts:
        all_ids.extend(enc.encode(text))

    print(f"Tokenized: {len(all_ids):,} tokens, vocab size: {vocab_size:,}")
    return all_ids, vocab_size


# =============================================================================
# 3. Build train/val split
# =============================================================================

def build_dataset(all_ids, seq_len, val_fraction=0.1):
    """
    Split tokens into overlapping chunks of seq_len.
    Returns train and val tensors of shape [n_chunks, seq_len+1].
    The +1 is for the next-token prediction target.
    """
    chunk_size = seq_len + 1
    n_chunks = len(all_ids) // chunk_size
    if n_chunks < 2:
        raise ValueError(
            f"Not enough data for seq_len={seq_len}. "
            f"Need at least {chunk_size * 2} tokens, got {len(all_ids)}."
        )

    data = torch.tensor(all_ids[:n_chunks * chunk_size], dtype=torch.long)
    data = data.view(n_chunks, chunk_size)

    n_val = max(1, int(n_chunks * val_fraction))
    train_data = data[:-n_val]
    val_data = data[-n_val:]

    print(f"Dataset: {train_data.shape[0]} train chunks, "
          f"{val_data.shape[0]} val chunks, seq_len={seq_len}")
    return train_data, val_data


# =============================================================================
# 4. Evaluate perplexity
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
# 5. Training loop
# =============================================================================

def train(args):
    from gravity import GravityLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load and tokenize data
    texts = load_texts(args.data)
    all_ids, vocab_size = tokenize(texts)

    # Build dataset
    train_data, val_data = build_dataset(all_ids, args.seq_len)

    # Build model
    model = GravityLM(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        K=args.K,
        S=args.S,
        C=args.C,
        R=args.R,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(f"Inference state: {args.S * args.C * 4} bytes/layer (O(1), independent of context length)\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1
    )

    # Optional: mixed precision on CUDA
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_ppl = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_tokens = 0
        perm = torch.randperm(len(train_data))
        train_data = train_data[perm]

        for i in range(0, len(train_data), args.batch_size):
            batch = train_data[i:i + args.batch_size].to(device)
            x, y = batch[:, :-1], batch[:, 1:]

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size), y.reshape(-1)
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size), y.reshape(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

        train_ppl = math.exp(min(total_loss / total_tokens, 20))
        val_ppl = evaluate(model, val_data, args.batch_size, device)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

        # Save best checkpoint
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            if args.save:
                torch.save(model.state_dict(), args.save)
                print(f"  Saved checkpoint → {args.save}")

    print(f"\nDone! Best val PPL: {best_val_ppl:.2f}")
    if args.save:
        print(f"Checkpoint saved at: {args.save}")


# =============================================================================
# 6. Generate text from a trained model
# =============================================================================

@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, device, temperature=1.0, top_k=50):
    """Simple autoregressive text generation."""
    model.eval()
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]  # [1, vocab_size]
        logits = logits / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1:]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[0].tolist()


# =============================================================================
# Argument parsing
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Gravity on custom text data"
    )

    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Path to a .txt file or folder of .txt files")
    parser.add_argument("--seq_len", type=int, default=1024,
                        help="Training sequence length (default: 1024)")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")

    # Model
    parser.add_argument("--d_model", type=int, default=256,
                        help="Model dimension (default: 256)")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of Gravity blocks (default: 4)")
    parser.add_argument("--K", type=int, default=64,
                        help="Coin parameter dimension (default: 64)")
    parser.add_argument("--S", type=int, default=3,
                        help="Number of field scales (default: 3)")
    parser.add_argument("--C", type=int, default=16,
                        help="Density channels (default: 16)")
    parser.add_argument("--R", type=int, default=5,
                        help="Window half-size (default: 5)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (default: 0.1)")

    # Training
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")

    # Output
    parser.add_argument("--save", type=str, default="gravity_checkpoint.pt",
                        help="Path to save the best checkpoint (default: gravity_checkpoint.pt)")

    args = parser.parse_args()
    train(args)
