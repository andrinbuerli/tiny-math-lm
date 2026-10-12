from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from tinymath.checkpoint import load_checkpoint, save_checkpoint
from tinymath.config import load_config
from tinymath.repro import seed_everything


class InfrastructureTests(unittest.TestCase):
    def test_yaml_config_requires_mapping_and_typed_values(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("seed: 17\nname: pilot\n", encoding="utf-8")

            config = load_config(config_path)
            self.assertEqual(config.require("seed", int), 17)
            self.assertEqual(config.require("name", str), "pilot")

    def test_seed_everything_repeats_torch_random_sequence(self):
        seed_everything(23)
        expected = torch.rand(5)

        seed_everything(23)
        self.assertTrue(torch.equal(torch.rand(5), expected))

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "nested" / "checkpoint.pt"
            payload = {"step": 9, "weights": torch.arange(4)}

            save_checkpoint(checkpoint_path, payload)
            restored = load_checkpoint(checkpoint_path)

            self.assertEqual(restored["step"], 9)
            self.assertTrue(torch.equal(restored["weights"], payload["weights"]))


if __name__ == "__main__":
    unittest.main()
