"""
STEP 5 — Evaluate. This is where most real-world LLM work actually lives.

Three modes:

  1) Score a trained tiny GPT on every held-out set:
        python step5_eval.py --run runs/baseline

  2) Score a NON-ML baseline (python-dateutil) on the same sets:
        python step5_eval.py --dateutil
     Always ask "what does the boring solution score?" before celebrating.

  3) Compare two scored systems on the SAME items (paired test), optionally as a
     regression gate that exits non-zero if B is significantly worse than A:
        python step5_eval.py --compare runs/baseline/report_tinygpt.json \
                                       runs/no_mask/report_tinygpt.json --gate

Rules this script enforces by construction:
  * greedy decoding (deterministic: same model → same score, every time)
  * checkpoint chosen on VAL during training; TEST is only read here
  * every prediction is saved, so you can do error analysis and paired tests
"""
import argparse
import json
import sys
from pathlib import Path

from evals import load_jsonl, mcnemar_exact, pct, print_report, run_suite, save_results


def eval_tinygpt(args):
    import torch
    from step2_tokenizer import CharTokenizer
    from step3_model import GPTConfig, TinyGPT, predict

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    ckpt = torch.load(Path(args.run) / f"ckpt_{args.ckpt}.pt", map_location=device)
    model = TinyGPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    tok = CharTokenizer()
    print(f"loaded {args.run} ckpt_{args.ckpt} (step {ckpt['step']}, val loss {ckpt['val_loss']:.4f})")

    results = run_suite(lambda xs: predict(model, tok, xs, device), args.data, limit=args.limit)
    print_report(f"tiny GPT — {args.run}", results)
    path = save_results(results, args.run, "tinygpt")
    print(f"\nsaved {path}")


def eval_dateutil(args):
    from dateutil import parser

    def predict(xs):
        out = []
        for s in xs:
            try:
                out.append(parser.parse(s, dayfirst=True, fuzzy=True).date().isoformat())
            except (ValueError, OverflowError):
                out.append("")
        return out

    results = run_suite(predict, args.data, limit=args.limit)
    print_report("python-dateutil (no ML)", results)
    path = save_results(results, "runs/dateutil", "dateutil")
    print(f"\nsaved {path}")


def compare(a_report, b_report, gate):
    def preds(report_path, set_name):
        report_path = Path(report_path)
        tag = report_path.stem.removeprefix("report_")
        return {r["id"]: r for r in load_jsonl(report_path.parent / f"preds_{tag}_{set_name}.jsonl")}

    a_sum, b_sum = json.loads(Path(a_report).read_text()), json.loads(Path(b_report).read_text())
    print(f"\nA = {a_report}\nB = {b_report}\n")
    print(f"{'eval set':<17}{'A':>8}{'B':>8}{'B−A':>8}   {'B wins':>6} {'A wins':>6}   {'p (McNemar)':>11}  verdict")
    regressions = []
    for name in a_sum:
        if name not in b_sum:
            continue
        pa, pb = preds(a_report, name), preds(b_report, name)
        ids = sorted(set(pa) & set(pb))           # pair on identical items only
        a_ok = [pa[i]["error"] == "correct" for i in ids]
        b_ok = [pb[i]["error"] == "correct" for i in ids]
        b_wins, a_wins, p = mcnemar_exact(a_ok, b_ok)
        ea, eb = sum(a_ok) / len(ids), sum(b_ok) / len(ids)
        if p >= 0.05:
            verdict = "no significant difference"
        elif eb > ea:
            verdict = "B better"
        else:
            verdict = "B WORSE"
            regressions.append(name)
        print(f"{name:<17}{pct(ea):>8}{pct(eb):>8}{100*(eb-ea):>+7.1f}   {b_wins:>6} {a_wins:>6}   {p:>11.4f}  {verdict}")
    if gate and regressions:
        print(f"\nGATE FAILED: significant regression on {regressions}")
        sys.exit(1)
    if gate:
        print("\nGATE PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run directory from step4_train.py")
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--dateutil", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A_REPORT", "B_REPORT"))
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--data", default="data")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare, args.gate)
    elif args.dateutil:
        eval_dateutil(args)
    elif args.run:
        eval_tinygpt(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
