"""Offline end-to-end checks; all generated artifacts live in temporary folders."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_lab import generate


ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        generate(self.work / "data")
        (self.work / "decision.md").write_text("Frozen smoke-test system; no quality claim.")

    def run_cli(self, script, *args, expected=0):
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
        result = subprocess.run(
            [sys.executable, str(ROOT / script), *args], cwd=self.work,
            env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_tinygpt_parser_and_final_protocol(self):
        self.run_cli("step4_train.py", "--name", "smoke", "--device", "cpu",
                     "--n_layer", "1", "--n_head", "2", "--n_embd", "16",
                     "--batch_size", "4", "--max_steps", "2", "--eval_interval", "1")
        run = self.work / "runs" / "smoke"
        training = json.loads((run / "training_manifest.json").read_text())
        self.assertEqual(training["train_rows"], 40)
        self.assertEqual(training["configuration"]["seed"], 0)
        self.run_cli("step5_eval.py", "--run", "runs/smoke", "--device", "cpu", "--limit", "2")
        dev_report = (run / "report_tinygpt.json").read_bytes()
        self.run_cli("step5_eval.py", "--dateutil", "--limit", "2")
        self.run_cli("step5_eval.py", "--compare", "runs/smoke/report_tinygpt.json",
                     "runs/smoke/report_tinygpt.json", "--gate")
        denied = self.run_cli("step5_eval.py", "--dateutil", "--final", expected=2)
        self.assertIn("--decision", denied.stderr)
        self.run_cli("step5_eval.py", "--run", "runs/smoke", "--device", "cpu",
                     "--final", "--decision", "decision.md")
        self.run_cli("step5_eval.py", "--dateutil", "--final", "--decision", "decision.md")
        self.assertEqual((run / "report_tinygpt.json").read_bytes(), dev_report)
        final = json.loads((run / "report_tinygpt_final.json").read_text())
        self.assertEqual({k: v["n"] for k, v in final.items()},
                         {"test": 14, "test_ood_format": 7, "test_ood_year": 7})
        manifest = json.loads((run / "manifest_tinygpt_final.json").read_text())
        self.assertEqual(manifest["decision"]["text"], (self.work / "decision.md").read_text())
        self.assertIn("sha256", manifest["checkpoint"])
        self.run_cli("step4_train.py", "--name", "broken", "--device", "cpu",
                     "--n_layer", "1", "--n_head", "2", "--n_embd", "16",
                     "--batch_size", "4", "--max_steps", "1", "--eval_interval", "1",
                     "--diagnostic_no_causal_mask")
        config = json.loads((self.work / "runs/broken/config.json").read_text())
        self.assertFalse(config["model"]["causal"])

    def test_local_lora_train_reload_and_final_without_network(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        base = self.work / "toy_model"
        tokenizer = Tokenizer(models.WordLevel(
            {"<pad>": 0, "<eos>": 1, "<unk>": 2, "assistant": 3, ":": 4}, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tok = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
            model_input_names=["input_ids", "attention_mask"])
        tok.chat_template = (
            "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}")
        tok.save_pretrained(base)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=5, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
            pad_token_id=0, eos_token_id=1))
        model.save_pretrained(base)
        common = ("--model_id", str(base), "--device", "cpu", "--batch_size", "2")
        for mode in ("zeroshot", "fewshot", "train"):
            self.run_cli("step6_finetune_lora.py", *common, "--mode", mode, "--limit", "2",
                         "--train_size", "4", "--epochs", "1")
        self.run_cli("step5_eval.py", "--compare", "runs/smollm/report_fewshot.json",
                     "runs/smollm/report_lora.json")
        run = self.work / "runs/smollm"
        training = json.loads((run / "training_manifest_lora.json").read_text())
        self.assertEqual(training["train_rows"], 4)
        self.assertEqual(training["base_model"]["chat_template"], tok.chat_template)
        self.assertEqual(len(training["adapter"]), 2)
        self.run_cli("step6_finetune_lora.py", *common, "--mode", "lora",
                     "--adapter", "runs/smollm/lora_adapter",
                     "--final", "--decision", "decision.md")
        final = json.loads((run / "manifest_lora_final.json").read_text())
        self.assertEqual(final["decoding"], "greedy")
        self.assertEqual(final["adapter"], training["adapter"])


if __name__ == "__main__":
    unittest.main()
