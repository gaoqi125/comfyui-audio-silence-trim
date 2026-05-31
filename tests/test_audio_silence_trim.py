import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "audio_silence_trim.py"
SPEC = importlib.util.spec_from_file_location("audio_silence_trim", MODULE_PATH)
audio_silence_trim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audio_silence_trim)

AudioSilenceTrim = audio_silence_trim.AudioSilenceTrim
find_silence_runs = audio_silence_trim.find_silence_runs


def comfy_audio(samples, sample_rate=10):
    waveform = np.asarray(samples, dtype=np.float32).reshape(1, 1, -1)
    return {"waveform": waveform, "sample_rate": sample_rate}


class AudioSilenceTrimTests(unittest.TestCase):
    def test_finds_silence_runs_with_minimum_length(self):
        amplitude = np.asarray([0.2, 0.0, 0.0, 0.0, 0.3, 0.0], dtype=np.float32)

        self.assertEqual(find_silence_runs(amplitude, threshold=0.01, min_samples=2), [(1, 4)])

    def test_trims_long_silence_with_max_keep_cap(self):
        audio = comfy_audio([0.5] * 10 + [0.0] * 10 + [0.25] * 10, sample_rate=10)

        trimmed, report = AudioSilenceTrim().trim(
            audio,
            keep_ratio=0.5,
            max_keep_seconds=0.3,
            silence_threshold_db=-38.0,
            min_silence_seconds=0.5,
            min_cut_seconds=0.05,
        )

        self.assertEqual(trimmed["sample_rate"], 10)
        self.assertEqual(tuple(trimmed["waveform"].shape), (1, 1, 23))
        payload = json.loads(report)
        self.assertEqual(payload["detected_silences"], 1)
        self.assertAlmostEqual(payload["removed_seconds"], 0.7)

    def test_keeps_short_silence_when_cut_is_below_minimum(self):
        audio = comfy_audio([0.5] * 10 + [0.0] + [0.25] * 10, sample_rate=10)

        trimmed, report = AudioSilenceTrim().trim(
            audio,
            keep_ratio=0.5,
            max_keep_seconds=0.3,
            silence_threshold_db=-38.0,
            min_silence_seconds=0.1,
            min_cut_seconds=0.2,
        )

        self.assertEqual(tuple(trimmed["waveform"].shape), (1, 1, 21))
        self.assertEqual(json.loads(report)["trimmed_silences"], 0)

    def test_no_silence_returns_same_length(self):
        audio = comfy_audio([0.5] * 20, sample_rate=10)

        trimmed, report = AudioSilenceTrim().trim(
            audio,
            keep_ratio=0.5,
            max_keep_seconds=0.3,
            silence_threshold_db=-38.0,
            min_silence_seconds=0.5,
            min_cut_seconds=0.05,
        )

        self.assertEqual(tuple(trimmed["waveform"].shape), (1, 1, 20))
        self.assertEqual(json.loads(report)["detected_silences"], 0)

    def test_node_defaults_match_audio_workflow_rule(self):
        required = AudioSilenceTrim.INPUT_TYPES()["required"]

        self.assertEqual(required["keep_ratio"][1]["default"], 0.5)
        self.assertEqual(required["max_keep_seconds"][1]["default"], 0.3)
        self.assertEqual(required["silence_threshold_db"][1]["default"], -38.0)
        self.assertEqual(required["min_silence_seconds"][1]["default"], 0.18)
        self.assertEqual(required["min_cut_seconds"][1]["default"], 0.05)


if __name__ == "__main__":
    unittest.main()
