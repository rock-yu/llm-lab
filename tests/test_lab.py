import argparse
import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch

from evals import (DEV_SETS, FINAL_SETS, evaluation_plan, evaluation_tag,
                   load_jsonl, run_suite, save_results)
from provenance import file_identity, make_manifest, write_manifest
from step2_tokenizer import CharTokenizer
from step3_model import GPTConfig, TinyGPT
from step4_train import build_tensors
from step5_eval import compare
from step6_finetune_lora import collate, encode_train_example


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def generate(directory, *extra):
    subprocess.run(
        [sys.executable, str(ROOT / "step1_make_data.py"), "--out", str(directory),
         "--n_train", "40", "--n_val", "7", "--n_dev", "14", "--n_test", "14",
         "--n_ood", "7", *extra],
        check=True, capture_output=True, text=True)


class DatasetTests(unittest.TestCase):
    def test_isolation_and_augmentation_preserve_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, aug = Path(tmp) / "data", Path(tmp) / "aug"
            generate(data)
            generate(aug, "--extra_formats")
            for name in ("val", "fewshot", *DEV_SETS, *FINAL_SETS):
                self.assertEqual((data / f"{name}.jsonl").read_bytes(),
                                 (aug / f"{name}.jsonl").read_bytes())
            groups = [
                {r["output"] for r in load_jsonl(data / "train.jsonl")},
                {r["output"] for r in load_jsonl(data / "val.jsonl")},
                {r["output"] for s in DEV_SETS for r in load_jsonl(data / f"{s}.jsonl")},
                {r["output"] for s in FINAL_SETS for r in load_jsonl(data / f"{s}.jsonl")},
            ]
            for i, dates in enumerate(groups):
                for other in groups[i + 1:]:
                    self.assertFalse(dates & other)
            card = json.loads((data / "data_card.json").read_text())
            self.assertTrue(all(v == 0 for v in card["leakage_checks"].values()))
            train = load_jsonl(aug / "train.jsonl")
            self.assertEqual(len(train), 40)
            self.assertTrue(any(r["format"] in ("dmy_dash", "dmy_abbr_pad", "mdy_abbr", "abbr_slash")
                                for r in train))

    def test_impossible_sizes_fail_instead_of_sampling_forever(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "step1_make_data.py"), "--n_train", "999999999"],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("too few", result.stderr)


class EvaluationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        for name in (*DEV_SETS, *FINAL_SETS):
            (self.data / f"{name}.jsonl").write_text(json.dumps({
                "id": f"{name}-0", "input": "1 Jan 2000", "output": "2000-01-01",
                "format": "dmy_abbr"}) + "\n")
        self.args = argparse.Namespace(data=str(self.data), final=False, decision=None, limit=None)

    def test_default_never_opens_final_and_old_data_has_migration_error(self):
        sets, decision = evaluation_plan(self.args)
        self.assertEqual(sets, DEV_SETS)
        self.assertIsNone(decision)
        (self.data / "dev.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "regenerate data"):
            evaluation_plan(self.args)

    def test_final_requires_frozen_decision_and_full_suite(self):
        self.args.final = True
        with self.assertRaisesRegex(ValueError, "freeze"):
            evaluation_plan(self.args)
        decision = self.data / "decision.md"
        decision.write_text("   \n")
        self.args.decision = str(decision)
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluation_plan(self.args)
        decision.write_text("Frozen parser; no training.")
        self.args.limit = 1
        with self.assertRaisesRegex(ValueError, "all items"):
            evaluation_plan(self.args)
        self.args.limit = None
        sets, snapshot = evaluation_plan(self.args)
        self.assertEqual(sets, FINAL_SETS)
        self.assertEqual(snapshot["text"], decision.read_text())
        self.assertEqual(snapshot["sha256"], file_identity(decision)["sha256"])
        self.assertEqual(evaluation_tag("dateutil", self.args), "dateutil_final")

    def test_default_suite_limit_and_empty_set(self):
        calls = []

        def predict(inputs):
            calls.append(inputs)
            return ["2000-01-01"] * len(inputs)

        results = run_suite(predict, self.data, limit=1)
        self.assertEqual(tuple(results), DEV_SETS)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(r["n"] == 1 and r["exact_match"] == 1 for r in results.values()))
        self.assertTrue(all(r["seconds_per_item"] >= 0 for r in results.values()))
        (self.data / "dev.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "no evaluation items"):
            run_suite(predict, self.data)

    def test_invalid_final_lora_training_fails_before_download(self):
        decision = self.data / "decision.md"
        decision.write_text("Freeze an existing adapter.")
        result = subprocess.run(
            [sys.executable, str(ROOT / "step6_finetune_lora.py"), "--mode", "train",
             "--data", str(self.data), "--final", "--decision", str(decision)],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("freeze training first", result.stderr)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        examples = [
            {"id": "dev-0", "input": "1 Jan 2000", "output": "2000-01-01", "format": "dmy_abbr"},
            {"id": "dev-1", "input": "2 Jan 2000", "output": "2000-01-02", "format": "dmy_abbr"},
        ]
        from evals import score
        self.results = {"dev": score(examples, [r["output"] for r in examples])}

    def reports(self, b_results=None, b_decoding="greedy"):
        a = save_results(self.results, self.directory / "a", "tinygpt", {"decoding": "greedy"})
        b = save_results(b_results if b_results is not None else self.results,
                         self.directory / "b", "tinygpt", {"decoding": b_decoding})
        return a, b

    def test_identical_items_and_parser_policy_can_compare(self):
        a, b = self.reports(b_decoding=None)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            compare(a, b, gate=True)
        self.assertIn("GATE PASSED", out.getvalue())

    def test_rejects_partial_overlap_and_changed_examples(self):
        for change in ("id", "input", "gold", "format"):
            with self.subTest(change=change):
                results = copy.deepcopy(self.results)
                results["dev"]["rows"][0][change] = "changed"
                a, b = self.reports(results)
                with self.assertRaises(ValueError):
                    compare(a, b, gate=False)

    def test_rejects_duplicates_counts_suites_and_decoding(self):
        for change in ("duplicate", "count", "suite", "decoding", "empty"):
            with self.subTest(change=change):
                results = copy.deepcopy(self.results)
                policy = "greedy"
                if change == "duplicate":
                    results["dev"]["rows"][1]["id"] = results["dev"]["rows"][0]["id"]
                elif change == "count":
                    results["dev"]["n"] += 1
                elif change == "suite":
                    results["different"] = results.pop("dev")
                elif change == "decoding":
                    policy = "sampling"
                else:
                    results["dev"]["rows"] = []
                a, b = self.reports(results, b_decoding=policy)
                with self.assertRaises(ValueError):
                    compare(a, b, gate=False)

    def test_legacy_warning_and_statistical_regression_gate(self):
        from evals import score
        examples = [{"id": f"dev-{i}", "input": f"{i + 1} Jan 2000",
                     "output": f"2000-01-{i + 1:02d}", "format": "dmy_abbr"} for i in range(10)]
        a = save_results({"dev": score(examples, [r["output"] for r in examples])},
                         self.directory / "a", "good")
        b = save_results({"dev": score(examples, ["wrong"] * len(examples))},
                         self.directory / "b", "bad")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as exc:
                compare(a, b, gate=True)
        self.assertEqual(exc.exception.code, 1)
        self.assertIn("no manifest", err.getvalue())


class TrainingAndProvenanceTests(unittest.TestCase):
    def test_shifted_answer_mask_and_padding(self):
        tok = CharTokenizer()
        row = {"input": "3 March 2024", "output": "2024-03-03"}
        x, y = build_tensors([row], tok, 48, True)
        first = len(row["input"] + "=") - 1
        self.assertTrue((y[0, :first] == -100).all())
        self.assertEqual(x[0, first].item(), tok.sep_id)
        self.assertEqual(tok.decode([y[0, first].item()]), "2")
        self.assertEqual(y[0, first + 10].item(), tok.eos_id)
        self.assertTrue((y[0, first + 11:] == -100).all())
        _, unmasked = build_tensors([row], tok, 48, False)
        self.assertEqual(tok.decode([unmasked[0, 0].item()]), " ")

    def test_causality_and_intentionally_broken_option(self):
        torch.manual_seed(7)
        x = torch.tensor([[1, 2, 3, 4]])
        changed = torch.tensor([[1, 2, 5, 6]])
        for causal in (True, False):
            model = TinyGPT(GPTConfig(vocab_size=8, block_size=8, n_layer=1, n_head=2,
                                      n_embd=16, causal=causal)).eval()
            a, _ = model(x)
            b, _ = model(changed)
            if causal:
                torch.testing.assert_close(a[:, :2], b[:, :2], rtol=0, atol=0)
            else:
                self.assertGreater((a[:, :2] - b[:, :2]).abs().max().item(), 1e-6)

    def test_lora_target_mask_and_right_padding(self):
        tok = CharTokenizer()
        tok.apply_chat_template = lambda messages, **kwargs: messages[-1]["content"] + "="
        # A lightweight tokenizer interface avoids downloading a model.
        class Tokenizer:
            eos_token_id = tok.eos_id
            apply_chat_template = staticmethod(tok.apply_chat_template)

            def __call__(self, text, **kwargs):
                return {"input_ids": tok.encode(text)}

        ids, labels = encode_train_example(Tokenizer(), {"input": "short", "output": "2000-01-01"})
        self.assertEqual(labels[:6], [-100] * 6)
        self.assertEqual(labels[6:], ids[6:])
        batch_ids, batch_labels, attention = collate([(ids, labels), (ids[:-1], labels[:-1])], tok.pad_id)
        self.assertEqual(batch_ids[1, -1].item(), tok.pad_id)
        self.assertEqual(batch_labels[1, -1].item(), -100)
        self.assertEqual(attention[1, -1].item(), 0)

    def test_manifest_hashes_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "dev.jsonl"
            data.write_text('{"id": "one"}\n')
            args = argparse.Namespace(seed=7, limit=1)
            manifest = make_manifest(args, [data], "cpu", time.perf_counter(), decoding="greedy")
            self.assertEqual(manifest["datasets"][0], file_identity(data))
            self.assertEqual(manifest["configuration"]["seed"], 7)
            self.assertIn("torch", manifest["dependencies"])
            self.assertTrue(manifest["source"]["python_files"])
            path = Path(tmp) / "manifest.json"
            write_manifest(path, manifest)
            self.assertEqual(json.loads(path.read_text()), manifest)
            old_hash = manifest["datasets"][0]["sha256"]
            data.write_text("changed")
            self.assertNotEqual(old_hash, file_identity(data)["sha256"])


if __name__ == "__main__":
    unittest.main()
