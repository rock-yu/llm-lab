"""
STEP 1 — Build the dataset (and the eval sets) BEFORE you touch a model.

The task: turn a messy, human-written date into ISO 8601.

    "3rd of March, 2024"   ->  "2024-03-03"
    "Sunday, 3 March 2024" ->  "2024-03-03"
    "03/03/2024"           ->  "2024-03-03"   (Australian DD/MM/YYYY)

Why this task?  It's small enough to train from scratch on a laptop CPU in
minutes, but it has the properties that make eval engineering interesting:
  * every answer has an exact, checkable ground truth (no LLM-as-judge needed)
  * it has natural sub-skills (copy the year, map month names, strip ordinals)
  * it has natural distribution shifts we can test on purpose

Outputs (in ./data):
  train.jsonl            what the model learns from
  val.jsonl              what WE look at during training (checkpoint selection)
  dev.jsonl              in-distribution benchmark for iteration
  dev_ood_format.jsonl   development benchmark for unseen formats
  dev_ood_year.jsonl     development benchmark for unseen years
  test.jsonl             final evaluation after freezing a decision
  test_ood_format.jsonl  a date format excluded from this lab's training data
  test_ood_year.jsonl    years (2040–2099) excluded from this lab's training data
  fewshot.jsonl          examples from the training date population for prompting
  data_card.json         counts, distributions and leakage checks

Key ideas this step teaches:
  1. Define the label precisely (DD/MM vs MM/DD is a *labelling* decision).
  2. Split by the underlying entity (the date), not by the row, to avoid leakage.
  3. Build out-of-distribution test sets deliberately, so you learn *how* the
     model generalises, not just *whether* it scores well on familiar data.
"""
import argparse
import datetime as dt
import json
import random
from collections import Counter
from pathlib import Path

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]


def ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


# Formats the model is allowed to learn from.
TRAIN_FORMATS = {
    "dmy_long":  lambda d: f"{d.day} {MONTHS[d.month-1]} {d.year}",            # 3 March 2024
    "mdy_long":  lambda d: f"{MONTHS[d.month-1]} {d.day}, {d.year}",           # March 3, 2024
    "dmy_slash": lambda d: f"{d.day:02d}/{d.month:02d}/{d.year}",              # 03/03/2024
    "ymd_dots":  lambda d: f"{d.year}.{d.month:02d}.{d.day:02d}",              # 2024.03.03
    "ordinal":   lambda d: f"{ordinal(d.day)} of {MONTHS[d.month-1]}, {d.year}",  # 3rd of March, 2024
    "dmy_abbr":  lambda d: f"{d.day} {MONTHS[d.month-1][:3]} {d.year}",        # 3 Mar 2024
    "weekday":   lambda d: f"{WEEKDAYS[d.weekday()]}, {d.day} {MONTHS[d.month-1]} {d.year}",
}

# A format held back from training inputs. Every *piece* appears in training
# (abbreviated month, zero-padded day, 4-digit year) — only the combination
# as a dash-separated input is new; '-' already appears in output labels.
HELDOUT_FORMATS = {
    "bank_style": lambda d: f"{d.day:02d}-{MONTHS[d.month-1][:3]}-{d.year}",   # 03-Mar-2024
}


# Optional extra training formats (Experiment 5: does more DIVERSE data help the
# model generalise to the held-out format?). Used for train rows only, so every
# eval set stays identical and results stay comparable.
EXTRA_FORMATS = {
    "dmy_dash":     lambda d: f"{d.day:02d}-{d.month:02d}-{d.year}",              # 03-03-2024
    "dmy_abbr_pad": lambda d: f"{d.day:02d} {MONTHS[d.month-1][:3]} {d.year}",   # 03 Mar 2024
    "mdy_abbr":     lambda d: f"{MONTHS[d.month-1][:3]} {d.day} {d.year}",       # Mar 3 2024
    "abbr_slash":   lambda d: f"{d.day}/{MONTHS[d.month-1][:3]}/{d.year}",       # 3/Mar/2024
}


def make_example(d: dt.date, fmt: str, split: str) -> dict:
    render = TRAIN_FORMATS.get(fmt) or EXTRA_FORMATS.get(fmt) or HELDOUT_FORMATS[fmt]
    return {
        "input": render(d),
        "output": d.isoformat(),
        "format": fmt,
        "year": d.year,
        "split": split,
    }


def date_range(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_dev", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--n_ood", type=int, default=500)
    ap.add_argument("--extra_formats", action="store_true",
                    help="add 4 more formats to TRAIN only (Experiment 5)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    if min(args.n_train, args.n_val, args.n_dev, args.n_test, args.n_ood) <= 0:
        ap.error("all dataset sizes must be positive")
    if args.n_ood > min(args.n_dev, args.n_test):
        ap.error("--n_ood must not exceed --n_dev or --n_test")
    population = (dt.date(2039, 12, 31) - dt.date(1950, 1, 1)).days + 1
    available = population - args.n_test - args.n_dev - args.n_val
    n_formats = len(TRAIN_FORMATS) + (len(EXTRA_FORMATS) if args.extra_formats else 0)
    if available < len(TRAIN_FORMATS) or args.n_train > available * n_formats:
        ap.error("requested splits leave too few training dates or unique training examples")
    future_population = (dt.date(2099, 12, 31) - dt.date(2040, 1, 1)).days + 1
    if 2 * args.n_ood > future_population:
        ap.error("requested OOD-year sets exceed the available disjoint future dates")

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. The population of dates we're allowed to train on -----------------
    in_dist_dates = list(date_range(dt.date(1950, 1, 1), dt.date(2039, 12, 31)))
    rng.shuffle(in_dist_dates)

    # --- 2. GROUP split: reserve final/dev/val DATES first --------------------
    # If "3 March 2024" were in train and "03/03/2024" in test, the model would
    # already have seen the exact answer string "2024-03-03". Splitting by date
    # (the entity) rather than by row prevents that subtle leak.
    test_dates = in_dist_dates[:args.n_test]
    dev_end = args.n_test + args.n_dev
    dev_dates = in_dist_dates[args.n_test:dev_end]
    val_dates = in_dist_dates[dev_end:dev_end + args.n_val]
    train_dates = in_dist_dates[dev_end + args.n_val:]

    fmts = list(TRAIN_FORMATS)
    train_fmts = fmts + (list(EXTRA_FORMATS) if args.extra_formats else [])

    # --- 3. Train: random (date, format) pairs, de-duplicated ------------------
    seen, train = set(), []
    while len(train) < args.n_train:
        ex = make_example(rng.choice(train_dates), rng.choice(train_fmts), "train")
        if ex["input"] in seen:
            continue                      # exact duplicates waste compute and skew loss
        seen.add(ex["input"])
        train.append(ex)

    # --- 4. Val / dev / test: round-robin formats for roughly equal counts -----
    val = [make_example(d, fmts[i % len(fmts)], "val") for i, d in enumerate(val_dates)]
    dev = [make_example(d, fmts[i % len(fmts)], "dev") for i, d in enumerate(dev_dates)]
    test = [make_example(d, fmts[i % len(fmts)], "test") for i, d in enumerate(test_dates)]

    # --- 5. Out-of-distribution sets -------------------------------------------
    ood_format = [make_example(d, "bank_style", "test_ood_format")
                  for d in test_dates[:args.n_ood]]
    dev_ood_format = [make_example(d, "bank_style", "dev_ood_format")
                      for d in dev_dates[:args.n_ood]]
    future = list(date_range(dt.date(2040, 1, 1), dt.date(2099, 12, 31)))
    random.Random(args.seed + 1).shuffle(future)   # own RNG: identical whatever the train options
    ood_year = [make_example(d, fmts[i % len(fmts)], "test_ood_year")
                for i, d in enumerate(future[:args.n_ood])]
    dev_ood_year = [make_example(d, fmts[i % len(fmts)], "dev_ood_year")
                    for i, d in enumerate(future[args.n_ood:2 * args.n_ood])]

    fewshot = [make_example(train_dates[i], fmts[i % len(fmts)], "fewshot") for i in range(len(fmts))]

    sets = {"train": train, "val": val, "dev": dev,
            "dev_ood_format": dev_ood_format, "dev_ood_year": dev_ood_year, "test": test,
            "test_ood_format": ood_format, "test_ood_year": ood_year,
            "fewshot": fewshot}
    for name, rows in sets.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"id": f"{name}-{i:05d}", **r}) + "\n")

    # --- 6. Leakage checks: these should ALL be zero ---------------------------
    train_inputs = {r["input"] for r in train}
    train_answers = {r["output"] for r in train}
    checks = {}
    for name in ["val", "dev", "dev_ood_format", "dev_ood_year",
                 "test", "test_ood_format", "test_ood_year"]:
        checks[f"{name}: inputs also in train"] = sum(r["input"] in train_inputs for r in sets[name])
        checks[f"{name}: answers also in train"] = sum(r["output"] in train_answers for r in sets[name])
    checks["val/test date overlap"] = len({r["output"] for r in val} & {r["output"] for r in test})
    groups = {
        "train": train_answers,
        "val": {r["output"] for r in val},
        "dev": {r["output"] for rows in (dev, dev_ood_format, dev_ood_year) for r in rows},
        "final": {r["output"] for rows in (test, ood_format, ood_year) for r in rows},
    }
    for i, (name, dates) in enumerate(groups.items()):
        for other in list(groups)[i + 1:]:
            checks[f"{name}/{other} date overlap"] = len(dates & groups[other])
    if any(checks.values()):
        raise RuntimeError(f"dataset leakage detected: {checks}")

    card = {
        "protocol_version": 2,
        "seed": args.seed,
        "sizes": {k: len(v) for k, v in sets.items()},
        "train_format_counts": dict(Counter(r["format"] for r in train)),
        "train_year_range": [min(r["year"] for r in train), max(r["year"] for r in train)],
        "ood_year_range": [min(r["year"] for r in ood_year), max(r["year"] for r in ood_year)],
        "max_input_chars": max(len(r["input"]) for rows in sets.values() for r in rows),
        "leakage_checks": checks,
    }
    (out / "data_card.json").write_text(json.dumps(card, indent=2))

    print("== Data card ==")
    for k, v in card["sizes"].items():
        print(f"  {k:<18} {v:>6} examples")
    print(f"  train years {card['train_year_range']}, OOD years {card['ood_year_range']}")
    print("  train format mix:", card["train_format_counts"])
    print("\n== Leakage checks (all should be 0) ==")
    for k, v in checks.items():
        print(f"  {'OK ' if v == 0 else '!! '} {k}: {v}")
    print("\n== A few training rows ==")
    for r in train[:5]:
        print(f"  {r['input']!r:<34} -> {r['output']}")


if __name__ == "__main__":
    main()
