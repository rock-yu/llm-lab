"""
STEP 5 — Evaluate. This is where most real-world LLM work actually lives.

Three modes:

  1) Score a trained tiny GPT on development sets:
        python step5_eval.py --run runs/baseline

  2) Score a NON-ML baseline (python-dateutil) on the same sets:
        python step5_eval.py --dateutil
     Always ask "what does the boring solution score?" before celebrating.

  3) Compare two scored systems on the SAME items (paired test), optionally as a
     regression gate that exits non-zero if B is significantly worse than A:
        python step5_eval.py --compare runs/baseline/report_tinygpt.json \
                                       runs/no_mask/report_tinygpt.json --gate

Rules this script enforces by construction:
  * greedy decoding (no token sampling; environments can still affect results)
  * load the requested checkpoint (best validation loss by default)
  * final TEST requires --final --decision PATH
  * every prediction is saved, so you can do error analysis and paired tests
"""
import argparse
import json
import sys
import time
from pathlib import Path

from evals import (add_eval_arguments, evaluation_plan, evaluation_tag, load_jsonl,
                   mcnemar_exact, pct, print_report, run_suite, save_results)
from provenance import file_identity, make_manifest


def eval_tinygpt(args, sets, decision):
    started = time.perf_counter()
    import torch
    from step2_tokenizer import CharTokenizer
    from step3_model import GPTConfig, TinyGPT, predict

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    checkpoint_path = Path(args.run) / f"ckpt_{args.ckpt}.pt"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = TinyGPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    tok = CharTokenizer()
    print(f"loaded {args.run} ckpt_{args.ckpt} (step {ckpt['step']}, val loss {ckpt['val_loss']:.4f})")

    results = run_suite(lambda xs: predict(model, tok, xs, device), args.data, sets, limit=args.limit)
    print_report(f"tiny GPT — {args.run}", results)
    manifest = make_manifest(
        args, [Path(args.data) / f"{s}.jsonl" for s in sets], device, started,
        decoding="greedy", max_new_tokens=11, decision=decision,
        checkpoint=file_identity(checkpoint_path), model=ckpt["config"])
    path = save_results(results, args.run, evaluation_tag("tinygpt", args), manifest)
    print(f"\nsaved {path}")


def eval_dateutil(args, sets, decision):
    started = time.perf_counter()
    from dateutil import parser

    def predict(xs):
        out = []
        for s in xs:
            try:
                out.append(parser.parse(s, dayfirst=True, fuzzy=True).date().isoformat())
            except (ValueError, OverflowError):
                out.append("")
        return out

    results = run_suite(predict, args.data, sets, limit=args.limit)
    print_report("python-dateutil (no ML)", results)
    manifest = make_manifest(
        args, [Path(args.data) / f"{s}.jsonl" for s in sets], "cpu", started,
        decoding=None, decision=decision, parser={"dayfirst": True, "fuzzy": True})
    path = save_results(results, args.out, evaluation_tag("dateutil", args), manifest)
    print(f"\nsaved {path}")


def compare(a_report, b_report, gate):
    def preds(report_path, set_name):
        report_path = Path(report_path)
        tag = report_path.stem.removeprefix("report_")
        rows = load_jsonl(report_path.parent / f"preds_{tag}_{set_name}.jsonl")
        by_id = {r["id"]: r for r in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"duplicate item IDs in {report_path}: {set_name}")
        return by_id

    a_sum, b_sum = json.loads(Path(a_report).read_text()), json.loads(Path(b_report).read_text())
    if not a_sum or set(a_sum) != set(b_sum):
        raise ValueError("reports must contain the same nonempty evaluation suites")
    manifests = []
    for report in (a_report, b_report):
        path = Path(report)
        manifest_path = path.with_name(path.name.replace("report_", "manifest_", 1))
        if manifest_path.is_file():
            manifests.append(json.loads(manifest_path.read_text()))
        else:
            print(f"WARNING: no manifest for {report}; provenance cannot be checked", file=sys.stderr)
            manifests.append(None)
    if all(m is not None for m in manifests):
        a_decode, b_decode = (m.get("decoding") for m in manifests)
        if a_decode is not None and b_decode is not None and a_decode != b_decode:
            raise ValueError("reports use different decoding policies")

    pairs = {}
    for name in a_sum:
        pa, pb = preds(a_report, name), preds(b_report, name)
        if not pa or set(pa) != set(pb):
            raise ValueError(f"{name}: reports must contain identical nonempty item IDs; "
                             "rerun both with the same --limit")
        if len(pa) != a_sum[name]["n"] or len(pb) != b_sum[name]["n"]:
            raise ValueError(f"{name}: report counts do not match prediction files")
        for item_id in pa:
            if any(pa[item_id][key] != pb[item_id][key] for key in ("input", "gold", "format")):
                raise ValueError(f"{name}/{item_id}: input, label, or format differs")
        pairs[name] = (pa, pb)

    print(f"\nA = {a_report}\nB = {b_report}\n")
    print(f"{'eval set':<17}{'A':>8}{'B':>8}{'B−A':>8}   {'B wins':>6} {'A wins':>6}   {'p (McNemar)':>11}  verdict")
    regressions = []
    for name in a_sum:
        pa, pb = pairs[name]
        ids = sorted(pa)
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
    ap.add_argument("--out", default="runs/dateutil", help="dateutil output directory")
    ap.add_argument("--compare", nargs=2, metavar=("A_REPORT", "B_REPORT"))
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--data", default="data")
    add_eval_arguments(ap)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    try:
        if args.compare:
            if args.final or args.decision or args.limit is not None:
                ap.error("--compare reads existing reports; omit --final, --decision, and --limit")
            compare(*args.compare, args.gate)
        elif args.dateutil or args.run:
            sets, decision = evaluation_plan(args)
            if args.dateutil:
                eval_dateutil(args, sets, decision)
            else:
                eval_tinygpt(args, sets, decision)
        else:
            ap.print_help()
    except (ValueError, OSError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
