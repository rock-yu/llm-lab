"""
STEP 4 — Train the tiny GPT from scratch.

The next-token training loop used here is:

    1. take a batch of token sequences
    2. forward pass  → the model's predicted distribution for each next token
    3. loss          → how surprised was it by the real next token? (cross-entropy)
    4. backward pass → gradient of the loss w.r.t. every parameter
    5. optimiser     → nudge every parameter a little downhill
    6. every so often, measure on data it ISN'T trained on (validation)

Broad-corpus pretraining uses the same basic next-token objective at much
larger scale. Here we train on prompt→answer pairs from random weights.
Supervised fine-tuning (SFT) commonly uses such pairs with a pretrained model.
Only ANSWER tokens are scored by default. With `--no_mask_prompt`, the model
is also graded on prompt predictions. Challenge 6 compares both objectives;
neither is guaranteed to generalize better.

Usage:
    python step4_train.py                         # default run → runs/baseline
    python step4_train.py --overfit_one_batch     # debugging sanity check
    python step4_train.py --name small_data --train_size 1000
    python step4_train.py --name no_mask --no_mask_prompt

Outputs (in runs/<name>/): config.json, log.csv, ckpt_best.pt, ckpt_last.pt,
training_manifest.json
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch

from evals import load_jsonl
from step2_tokenizer import CharTokenizer
from step3_model import GPTConfig, TinyGPT, predict
from provenance import make_manifest, write_manifest


def build_tensors(rows, tok, block_size, mask_prompt):
    """Encode 'input=output\\n' and build (x, y) where y is x shifted left by one.
    Targets we don't want to learn from are set to -100 (ignored by the loss)."""
    X = torch.full((len(rows), block_size), tok.pad_id, dtype=torch.long)
    Y = torch.full((len(rows), block_size), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        prompt = r["input"] + "="
        ids = tok.encode(prompt + r["output"] + "\n")
        assert len(ids) - 1 <= block_size, f"example longer than block_size: {r['input']!r}"
        x, y = ids[:-1], ids[1:]
        y = list(y)
        if mask_prompt:
            # y[t] is the token at position t+1. Positions 0..len(prompt)-2 predict
            # prompt characters — mask them. y[len(prompt)-1] is the first answer char.
            for t in range(len(prompt) - 1):
                y[t] = -100
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
    return X, Y


def lr_at(step, args):
    """Linear warmup, then cosine decay to 10% of peak. Warmup stops the first
    few (random, large) gradients from wrecking the freshly initialised weights."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


@torch.no_grad()
def mean_loss(model, X, Y, batch_size=512):
    model.eval()
    total, count = 0.0, 0
    for s in range(0, len(X), batch_size):
        x, y = X[s:s + batch_size], Y[s:s + batch_size]
        _, loss = model(x, y)
        n = (y != -100).sum().item()
        total += loss.item() * n
        count += n
    model.train()
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--data", default="data")
    ap.add_argument("--train_size", type=int, default=None, help="use only the first N train rows")
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--block_size", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--no_mask_prompt", action="store_true")
    ap.add_argument("--overfit_one_batch", action="store_true")
    ap.add_argument("--diagnostic_no_causal_mask", action="store_true",
                    help="INTENTIONALLY BROKEN: allow future-token leakage for the debugging exercise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    args = ap.parse_args()
    started = time.perf_counter()
    if args.diagnostic_no_causal_mask:
        print("WARNING: diagnostic mode removes causality. Loss can cheat by seeing future tokens; "
              "this is not a valid autoregressive model.")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    tok = CharTokenizer()
    train_rows = load_jsonl(Path(args.data) / "train.jsonl")[:args.train_size]
    val_rows = load_jsonl(Path(args.data) / "val.jsonl")

    mask_prompt = not args.no_mask_prompt
    Xtr, Ytr = build_tensors(train_rows, tok, args.block_size, mask_prompt)
    # Validation loss is ALWAYS measured on answer tokens only, whatever we train
    # on — otherwise runs with and without masking aren't comparable.
    Xva, Yva = build_tensors(val_rows, tok, args.block_size, mask_prompt=True)
    Xtr, Ytr, Xva, Yva = Xtr.to(device), Ytr.to(device), Xva.to(device), Yva.to(device)

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
                    causal=not args.diagnostic_no_causal_mask)
    model = TinyGPT(cfg).to(device)

    # AdamW with weight decay on matrices only (not on biases / LayerNorm gains).
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    (run_dir / "config.json").write_text(json.dumps({"args": vars(args), "model": cfg.__dict__}, indent=2))
    print(f"run '{args.name}' on {device}: {model.num_params():,} params, "
          f"{len(train_rows):,} train rows, mask_prompt={mask_prompt}")

    # SANITY CHECK 1: near-uniform random predictions have loss near ln(vocab).
    init_loss = mean_loss(model, Xva, Yva)
    print(f"step 0 val loss {init_loss:.3f}  (expect ≈ ln({tok.vocab_size}) = {math.log(tok.vocab_size):.3f})")

    fixed_batch = torch.randint(len(Xtr), (args.batch_size,), device=device)
    val_inputs = [r["input"] for r in val_rows[:200]]
    val_gold = [r["output"] for r in val_rows[:200]]

    log_f = open(run_dir / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "lr", "train_loss", "val_loss", "val_exact_match", "elapsed_s"])

    best_val, t0, running = float("inf"), time.time(), None
    model.train()
    for step in range(args.max_steps + 1):
        # ---- periodic evaluation ------------------------------------------------
        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss = mean_loss(model, Xva, Yva)
            preds = predict(model, tok, val_inputs, device)
            val_em = sum(p == g for p, g in zip(preds, val_gold)) / len(val_gold)
            model.train()
            tr = running if running is not None else float("nan")
            print(f"step {step:5d} | lr {lr_at(step, args):.2e} | train {tr:.4f} | val {val_loss:.4f} "
                  f"| val EM {100*val_em:5.1f}% | {time.time()-t0:5.0f}s | "
                  f"{val_inputs[0]!r} -> {preds[0]!r}")
            log.writerow([step, f"{lr_at(step, args):.6f}", f"{tr:.5f}", f"{val_loss:.5f}",
                          f"{val_em:.4f}", f"{time.time()-t0:.1f}"])
            log_f.flush()
            ckpt = {"model": model.state_dict(), "config": cfg.__dict__, "step": step, "val_loss": val_loss}
            if val_loss < best_val:   # checkpoint SELECTION uses val, never test
                best_val = val_loss
                torch.save(ckpt, run_dir / "ckpt_best.pt")
            if step == args.max_steps:
                torch.save(ckpt, run_dir / "ckpt_last.pt")
                break

        # ---- one optimisation step ---------------------------------------------
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        ix = fixed_batch if args.overfit_one_batch else torch.randint(len(Xtr), (args.batch_size,), device=device)
        _, loss = model(Xtr[ix], Ytr[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # tames rare huge gradients
        opt.step()
        running = loss.item() if running is None else 0.9 * running + 0.1 * loss.item()

    log_f.close()
    write_manifest(run_dir / "training_manifest.json", make_manifest(
        args, [Path(args.data) / f"{s}.jsonl" for s in ("train", "val")],
        device, started, model=cfg.__dict__, objective="next-token cross-entropy",
        tokenizer={"kind": "character", "vocabulary": tok.itos},
        train_rows=len(train_rows), checkpoint_selection="minimum validation answer-token loss"))
    print(f"\ndone in {time.time()-t0:.0f}s. best val loss {best_val:.4f}. "
          f"Next: python step5_eval.py --run {run_dir}")


if __name__ == "__main__":
    main()
