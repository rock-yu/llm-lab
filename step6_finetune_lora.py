"""
STEP 6 (Part 2) — Same task, but start from a model that already knows language.

Part 1 trained ~0.8M parameters from random noise. Here we take a small
PRETRAINED, instruction-tuned model (SmolLM2-135M-Instruct, Apache-2.0) and:

  --mode zeroshot   just ask it (instruction only)              ← prompting baseline
  --mode fewshot    ask it with 7 worked examples in the prompt ← better prompting baseline
  --mode train      LoRA fine-tune on our train set, then evaluate

…all scored by the SAME harness (evals.py) on the SAME held-out sets as Part 1.
The question you're answering is the one teams face in real life:
"Do we need to fine-tune at all, and if so, what does it buy us?"

What's different from Part 1 (and why):
  * Sub-word tokenizer + chat template — the model was trained on a specific
    conversation format (<|im_start|>user …). Use the template, or it underperforms.
  * LoRA — instead of updating all 135M weights, freeze them and learn small
    low-rank "adapter" matrices (≈0.3% of params) on the attention projections.
    Cheaper, faster, and the original model is untouched.
  * Loss masking — same idea as Part 1: only the answer tokens are scored.

Usage:
    pip install -r requirements.txt
    python step6_finetune_lora.py --mode zeroshot
    python step6_finetune_lora.py --mode fewshot
    python step6_finetune_lora.py --mode train
    python step5_eval.py --compare runs/smollm/report_fewshot.json runs/smollm/report_lora.json

First run downloads the model (~270MB) from huggingface.co.
CPU works; Apple Silicon (mps) or a CUDA GPU is faster.
"""
import argparse
import math
import random
import time
from pathlib import Path

import torch

from evals import load_jsonl, print_report, run_suite, save_results

SYSTEM = ("Convert the date to ISO 8601 format (YYYY-MM-DD). "
          "Dates written with slashes are day/month/year. Reply with the date only.")


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_prompt(tok, text, shots):
    """The prompt IS part of your system's spec — version it like code."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for s in shots:
        msgs += [{"role": "user", "content": s["input"]}, {"role": "assistant", "content": s["output"]}]
    msgs.append({"role": "user", "content": text})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def make_predict_fn(model, tok, device, shots, batch_size):
    def predict(inputs):
        model.eval()
        tok.padding_side = "left"          # generation needs LEFT padding (new tokens go on the right)
        outs = []
        for s in range(0, len(inputs), batch_size):
            prompts = [build_prompt(tok, x, shots) for x in inputs[s:s + batch_size]]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            gen = model.generate(**enc, max_new_tokens=12, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for row in gen[:, enc["input_ids"].shape[1]:]:
                text = tok.decode(row, skip_special_tokens=True).strip()
                outs.append(text.splitlines()[0].strip() if text else "")
        return outs
    return predict


def encode_train_example(tok, ex):
    prompt_ids = tok(build_prompt(tok, ex["input"], []), add_special_tokens=False)["input_ids"]
    answer_ids = tok(ex["output"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids   # mask the prompt


def collate(batch, pad_id):
    T = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), T), pad_id)
    labels = torch.full((len(batch), T), -100)
    mask = torch.zeros((len(batch), T), dtype=torch.long)
    for i, (x, y) in enumerate(batch):           # RIGHT padding for training
        ids[i, :len(x)], labels[i, :len(y)], mask[i, :len(x)] = torch.tensor(x), torch.tensor(y), 1
    return ids, labels, mask


def train_lora(model, tok, args, device):
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                          target_modules=args.target_modules.split(","), task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    rows = load_jsonl(Path(args.data) / "train.jsonl")[:args.train_size]
    data = [encode_train_example(tok, r) for r in rows]
    val = [encode_train_example(tok, r) for r in load_jsonl(Path(args.data) / "val.jsonl")[:200]]
    steps = math.ceil(len(data) / args.batch_size) * args.epochs
    warmup = max(1, steps // 20)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (steps - s) / max(1, steps - warmup)))

    @torch.no_grad()
    def val_loss():
        model.eval()
        tot, n = 0.0, 0
        for s in range(0, len(val), args.batch_size):
            ids, labels, mask = (t.to(device) for t in collate(val[s:s + args.batch_size], tok.pad_token_id))
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            k = (labels[:, 1:] != -100).sum().item()
            tot, n = tot + out.loss.item() * k, n + k
        model.train()
        return tot / n

    print(f"LoRA training: {len(data)} examples × {args.epochs} epoch(s) = {steps} steps, batch {args.batch_size}")
    print(f"val loss before training: {val_loss():.4f}")
    model.train()
    step, t0, running = 0, time.time(), None
    for epoch in range(args.epochs):
        random.shuffle(data)
        for s in range(0, len(data), args.batch_size):
            ids, labels, mask = (t.to(device) for t in collate(data[s:s + args.batch_size], tok.pad_token_id))
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running = loss.item() if running is None else 0.9 * running + 0.1 * loss.item()
            if step % args.log_every == 0 or step == steps:
                print(f"step {step:4d}/{steps} | loss {running:.4f} | lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:5.0f}s")
    print(f"val loss after training: {val_loss():.4f}")
    adapter_dir = Path(args.out) / "lora_adapter"
    model.save_pretrained(adapter_dir)
    print(f"saved adapter to {adapter_dir} (only the small LoRA weights, not the base model)")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["zeroshot", "fewshot", "train"])
    ap.add_argument("--model_id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/smollm")
    ap.add_argument("--train_size", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--limit", type=int, default=300, help="examples per eval set (generation is the slow part)")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype).to(device)
    print(f"{args.model_id} on {device}: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    shots = load_jsonl(Path(args.data) / "fewshot.jsonl") if args.mode == "fewshot" else []
    print("\nexample prompt the model sees:\n" + "-" * 60)
    print(build_prompt(tok, "3rd of March, 2024", shots) + "-" * 60)

    if args.mode == "train":
        model = train_lora(model, tok, args, device)
    tag = {"zeroshot": "zeroshot", "fewshot": "fewshot", "train": "lora"}[args.mode]

    results = run_suite(make_predict_fn(model, tok, device, shots, args.batch_size),
                        args.data, limit=args.limit)
    print_report(f"{args.model_id} — {tag}", results)
    print(f"\nsaved {save_results(results, args.out, tag)}")


if __name__ == "__main__":
    main()
