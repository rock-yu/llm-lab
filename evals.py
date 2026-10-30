"""
The eval harness — shared by Part 1 (tiny GPT) and Part 2 (LoRA fine-tune).

It is deliberately MODEL-AGNOSTIC: anything that can turn a list of inputs into
a list of output strings can be scored. That's how you compare a from-scratch
model, a prompted pretrained model, a fine-tuned model and a plain old library
on exactly the same footing.

What it measures, and why:
  exact_match      the metric you actually care about (the whole answer right)
  95% CI           a score on 500 examples is an estimate, not a fact
  year/month/day   partial credit — tells you WHICH sub-skill is failing
  valid_date       did it at least produce a real calendar date?
  by_format        averages hide failures; slice by the thing you control
  error types      a taxonomy turns "it's wrong 4% of the time" into a fix
  McNemar test     is run B really better than run A on the SAME items?
"""
import datetime as dt
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
DEV_SETS = ("dev", "dev_ood_format", "dev_ood_year")
FINAL_SETS = ("test", "test_ood_format", "test_ood_year")


def add_eval_arguments(parser, default_limit=None):
    parser.add_argument("--final", action="store_true", help="open the final evaluation suite")
    parser.add_argument("--decision", help="nonempty frozen decision document, required with --final")
    parser.add_argument("--limit", type=int, default=default_limit, help="items per development set")


def evaluation_plan(args):
    """Validate before loading a model or exposing final results."""
    from provenance import decision_snapshot

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.decision and not args.final:
        raise ValueError("--decision requires --final")
    decision = None
    if args.final:
        if not args.decision:
            raise ValueError("--final requires --decision PATH; freeze your system choice first")
        if args.limit is not None:
            raise ValueError("final evaluation must use all items; omit --limit")
        decision = decision_snapshot(args.decision)
    sets = FINAL_SETS if args.final else DEV_SETS
    for name in sets:
        path = Path(args.data) / f"{name}.jsonl"
        if not path.is_file():
            raise ValueError(
                f"missing {path}; regenerate data with step1_make_data.py "
                "(back up old data/runs first). Test files are never used as a fallback.")
    return sets, decision


def evaluation_tag(tag, args):
    return f"{tag}_final" if args.final else tag


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def parse_iso(s):
    """Return (y, m, d) ints if the string LOOKS like YYYY-MM-DD, else None."""
    m = ISO_RE.match(s.strip())
    return tuple(int(g) for g in m.groups()) if m else None


def is_real_date(ymd):
    try:
        dt.date(*ymd)
        return True
    except ValueError:
        return False


def classify(pred: str, gold: str) -> str:
    """Put every answer into exactly one bucket."""
    if pred.strip() == gold:
        return "correct"
    p, g = parse_iso(pred), parse_iso(gold)
    if p is None:
        return "malformed"
    if not is_real_date(p):
        return "impossible_date"          # e.g. 2023-02-30
    if (p[0], p[1], p[2]) == (g[0], g[2], g[1]):
        return "day_month_swap"           # the classic DD/MM vs MM/DD bug
    wrong = [name for name, a, b in zip(("year", "month", "day"), p, g) if a != b]
    return f"wrong_{wrong[0]}" if len(wrong) == 1 else "multiple_fields"


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% confidence interval for a proportion. Better than ±1.96·√(p(1−p)/n)
    when the score is near 0% or 100%, which is exactly where good models live."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def score(examples, preds):
    assert len(examples) == len(preds)
    n = len(examples)
    rows = []
    for ex, pred in zip(examples, preds):
        p, g = parse_iso(pred or ""), parse_iso(ex["output"])
        rows.append({
            "id": ex["id"], "input": ex["input"], "gold": ex["output"], "pred": pred,
            "format": ex["format"], "error": classify(pred or "", ex["output"]),
            "year_ok": bool(p and p[0] == g[0]),
            "month_ok": bool(p and p[1] == g[1]),
            "day_ok": bool(p and p[2] == g[2]),
            "valid": bool(p and is_real_date(p)),
        })
    k = sum(r["error"] == "correct" for r in rows)
    by_format = defaultdict(list)
    for r in rows:
        by_format[r["format"]].append(r["error"] == "correct")
    return {
        "n": n,
        "exact_match": k / n,
        "exact_match_ci95": wilson_ci(k, n),
        "year_acc": sum(r["year_ok"] for r in rows) / n,
        "month_acc": sum(r["month_ok"] for r in rows) / n,
        "day_acc": sum(r["day_ok"] for r in rows) / n,
        "valid_date_rate": sum(r["valid"] for r in rows) / n,
        "by_format": {f: sum(v) / len(v) for f, v in sorted(by_format.items())},
        "errors": dict(Counter(r["error"] for r in rows if r["error"] != "correct").most_common()),
        "rows": rows,
    }


def run_suite(predict_fn, data_dir="data", sets=DEV_SETS, limit=None):
    """predict_fn: list[str] -> list[str]. Returns {set_name: score_dict}."""
    results = {}
    for name in sets:
        examples = load_jsonl(Path(data_dir) / f"{name}.jsonl")
        if limit:
            examples = examples[:limit]
        if not examples:
            raise ValueError(f"{name} contains no evaluation items")
        started = time.perf_counter()
        preds = predict_fn([ex["input"] for ex in examples])
        elapsed = time.perf_counter() - started
        results[name] = {**score(examples, preds),
                         "prediction_seconds": elapsed,
                         "seconds_per_item": elapsed / len(examples)}
    return results


def pct(x):
    return f"{100 * x:5.1f}%"


def print_report(title, results, show_failures=6):
    print(f"\n=== {title} ===")
    print(f"{'eval set':<17}{'n':>5}  {'exact match (95% CI)':<24}{'year':>7}{'month':>7}{'day':>7}{'valid':>7}")
    for name, r in results.items():
        lo, hi = r["exact_match_ci95"]
        em = f"{pct(r['exact_match'])} ({100*lo:.1f}–{100*hi:.1f})"
        print(f"{name:<17}{r['n']:>5}  {em:<24}{pct(r['year_acc']):>7}{pct(r['month_acc']):>7}"
              f"{pct(r['day_acc']):>7}{pct(r['valid_date_rate']):>7}")
    for name, r in results.items():
        print(f"\n  [{name}] exact match by format: "
              + ", ".join(f"{f} {pct(v).strip()}" for f, v in r["by_format"].items()))
        if r["errors"]:
            print(f"  [{name}] error types: " + ", ".join(f"{k}={v}" for k, v in r["errors"].items()))
        fails = [x for x in r["rows"] if x["error"] != "correct"][:show_failures]
        for x in fails:
            print(f"      {x['input']!r:<32} gold {x['gold']}  pred {x['pred']!r:<14} ({x['error']})")


def save_results(results, out_dir, tag, manifest=None):
    """Writes predictions (for error analysis and paired comparisons) + a summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, r in results.items():
        with open(out / f"preds_{tag}_{name}.jsonl", "w") as f:
            for row in r["rows"]:
                f.write(json.dumps(row) + "\n")
        summary[name] = {k: v for k, v in r.items() if k != "rows"}
    (out / f"report_{tag}.json").write_text(json.dumps(summary, indent=2))
    if manifest is not None:
        from provenance import write_manifest
        write_manifest(out / f"manifest_{tag}.json", manifest)
    return out / f"report_{tag}.json"


def mcnemar_exact(a_correct, b_correct):
    """Paired significance test for two systems scored on the SAME items.
    Only the items where they disagree carry information. Returns (b_only, a_only, p)."""
    a_only = sum(a and not b for a, b in zip(a_correct, b_correct))
    b_only = sum(b and not a for a, b in zip(a_correct, b_correct))
    n = a_only + b_only
    if n == 0:
        return b_only, a_only, 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b_only, a_only, min(1.0, 2 * p)
