"""Small synthetic checks: no real recordings or downloads are required."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hello import EEGCNN, EEGDataset
import optuna_tuning


class EEGVersionTwoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.rows = []
        self.windows = {}
        generator = np.random.default_rng(42)
        for recording_id in range(6):
            subject = "S001" if recording_id < 3 else "S002"
            split = ["train", "validation", "test"][recording_id % 3]
            windows = generator.normal(0, 1e-5, size=(4, 3, 128))
            if split == "test":
                windows += 100  # Must not affect training normalization.
            self.windows[recording_id] = windows
            np.save(self.folder / f"recording_{recording_id}_windows.npy", windows)
            for window_id in range(len(windows)):
                self.rows.append({"recording_id": recording_id, "subject_id": subject,
                    "group_id": f"{subject}_session_{recording_id}",
                    "window_id": window_id, "split": split})
        self.features = pd.DataFrame(self.rows)

    def tearDown(self):
        self.temporary.cleanup()

    def test_dataset_and_binary_labels(self):
        dataset = EEGDataset(self.features, str(self.folder), "S001")
        self.assertEqual(len(dataset), 24)
        window, label = dataset[0]
        self.assertEqual(tuple(window.shape), (3, 128))
        self.assertEqual(window.dtype, torch.float32)
        self.assertEqual(label.item(), 1.0)
        self.assertEqual(dataset[12][1].item(), 0.0)

    def test_cpu_forward_and_weight_update(self):
        torch.manual_seed(42)
        model = EEGCNN(3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = model.classifier.weight.detach().clone()
        prediction = model(torch.randn(4, 3, 128))
        self.assertEqual(tuple(prediction.shape), (4, 1))
        loss = torch.nn.BCELoss()(prediction, torch.tensor([[0.], [1.], [0.], [1.]]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertFalse(torch.equal(before, model.classifier.weight))

    def test_six_trials_without_test_data_leakage(self):
        real_load = np.load
        def load_training_or_validation(path, *args, **kwargs):
            self.assertNotIn(Path(path).name,
                             {"recording_2_windows.npy", "recording_5_windows.npy"})
            return real_load(path, *args, **kwargs)

        with patch.object(optuna_tuning.np, "load", side_effect=load_training_or_validation):
            train, validation, mean, std = optuna_tuning.prepare_tuning_data(
                self.features, str(self.folder), "S001")
            expected = np.concatenate([self.windows[0], self.windows[3]]).astype(np.float32)
            np.testing.assert_allclose(mean, expected.mean(axis=(0, 2))[:, None], atol=1e-12)
            np.testing.assert_allclose(std, expected.std(axis=(0, 2))[:, None], atol=1e-12)
            np.testing.assert_allclose(validation.windows[0],
                ((self.windows[1][0] - mean) / std).astype(np.float32))
            with patch.object(torch.cuda, "is_available", return_value=False):
                study = optuna_tuning.tune_cnn(self.features, str(self.folder),
                    ["F3", "F4", "Cz"], claimed_subject="S001", epochs=1)
        self.assertEqual(len(study.trials), 6)
        self.assertTrue(all(trial.state.name == "COMPLETE" for trial in study.trials))
        saved = self.folder / "optuna_cnn"
        summary = json.loads((saved / "best_parameters.json").read_text())
        self.assertFalse(summary["test_used"])
        self.assertAlmostEqual(summary["validation_loss"], min(t.value for t in study.trials))
        checkpoint = torch.load(saved / "best_cnn.pt", map_location="cpu", weights_only=True)
        np.testing.assert_allclose(checkpoint["channel_mean"].numpy(), mean)
        model = EEGCNN(3)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        with torch.no_grad():
            result = model(torch.tensor(validation.windows[0])[None, :, :])
        self.assertEqual(tuple(result.shape), (1, 1))
        self.assertTrue(torch.isfinite(result).all().item())

    def test_insufficient_and_crossed_groups_are_rejected(self):
        insufficient = self.features[self.features.recording_id != 2]
        with self.assertRaises(ValueError):
            optuna_tuning.prepare_tuning_data(insufficient, str(self.folder), "S001")
        crossed = self.features.copy()
        crossed.loc[crossed.recording_id == 1, "split"] = "train"
        crossed.loc[crossed.index[0], "split"] = "validation"
        with self.assertRaises(ValueError):
            optuna_tuning.prepare_tuning_data(crossed, str(self.folder), "S001")


if __name__ == "__main__":
    unittest.main()
