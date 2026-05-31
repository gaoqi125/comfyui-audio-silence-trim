import json
from typing import Any

import numpy as np


DEFAULT_KEEP_RATIO = 0.5
DEFAULT_MAX_KEEP_SECONDS = 0.30
DEFAULT_SILENCE_THRESHOLD_DB = -38.0
DEFAULT_MIN_SILENCE_SECONDS = 0.18
DEFAULT_MIN_CUT_SECONDS = 0.05


def db_to_amplitude(db_value: float) -> float:
    return 10 ** (float(db_value) / 20.0)


def normalize_waveform(waveform: Any):
    torch = _torch_module()

    if torch is not None:
        result = torch.as_tensor(waveform)
    else:
        result = np.asarray(waveform)
    if result.ndim == 1:
        result = result.reshape(1, 1, -1)
    elif result.ndim == 2:
        result = result.unsqueeze(0)
    if result.ndim != 3:
        raise ValueError(f"ComfyUI AUDIO waveform must be 1D, 2D, or 3D, got shape {tuple(result.shape)}")
    return result


def _torch_module():
    try:
        import torch

        return torch
    except ImportError:
        return None


def detection_amplitude(waveform):
    if waveform.shape[-1] == 0:
        torch = _torch_module()
        if torch is not None and isinstance(waveform, torch.Tensor):
            return waveform.new_zeros((0,), dtype=waveform.float().dtype)
        return np.zeros(0, dtype=np.float32)
    axes = tuple(range(waveform.ndim - 1))
    torch = _torch_module()
    if torch is not None and isinstance(waveform, torch.Tensor):
        return waveform.float().abs().mean(dim=axes)
    return np.abs(waveform.astype(np.float32, copy=False)).mean(axis=axes)


def find_silence_runs(amplitude, *, threshold: float, min_samples: int) -> list[tuple[int, int]]:
    silent = amplitude <= float(threshold)
    length = int(silent.numel()) if hasattr(silent, "numel") else int(np.asarray(silent).size)
    if length == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = None
    values = silent.detach().cpu().tolist() if hasattr(silent, "detach") else np.asarray(silent).tolist()
    for index, is_silent in enumerate(values):
        if is_silent and start is None:
            start = index
        elif not is_silent and start is not None:
            if index - start >= min_samples:
                runs.append((start, index))
            start = None
    if start is not None and length - start >= min_samples:
        runs.append((start, length))
    return runs


def build_keep_ranges(
    *,
    sample_count: int,
    silence_runs: list[tuple[int, int]],
    keep_ratio: float,
    max_keep_samples: int,
    min_cut_samples: int,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    ranges: list[tuple[int, int]] = []
    silences: list[dict[str, Any]] = []
    cursor = 0

    for start, end in silence_runs:
        start = max(0, min(int(start), sample_count))
        end = max(start, min(int(end), sample_count))
        if start > cursor:
            ranges.append((cursor, start))

        original = end - start
        keep = max(0, min(original, int(round(original * keep_ratio)), max_keep_samples))
        removed = original - keep
        trimmed = removed >= min_cut_samples
        if not trimmed:
            keep = original
            removed = 0
        if keep > 0:
            ranges.append((start, start + keep))

        silences.append(
            {
                "start_sample": start,
                "end_sample": end,
                "original_samples": original,
                "kept_samples": keep,
                "removed_samples": removed,
                "trimmed": trimmed,
            }
        )
        cursor = end

    if cursor < sample_count:
        ranges.append((cursor, sample_count))
    elif not silence_runs:
        ranges.append((0, sample_count))

    return [(start, end) for start, end in ranges if end > start], silences


def trim_audio(
    audio: dict[str, Any],
    *,
    keep_ratio: float = DEFAULT_KEEP_RATIO,
    max_keep_seconds: float = DEFAULT_MAX_KEEP_SECONDS,
    silence_threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
    min_silence_seconds: float = DEFAULT_MIN_SILENCE_SECONDS,
    min_cut_seconds: float = DEFAULT_MIN_CUT_SECONDS,
) -> tuple[dict[str, Any], str]:
    sample_rate = int(audio["sample_rate"])
    waveform = normalize_waveform(audio["waveform"])
    sample_count = int(waveform.shape[-1])

    keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
    max_keep_samples = max(1, int(round(sample_rate * max(0.0, float(max_keep_seconds)))))
    min_silence_samples = max(1, int(round(sample_rate * max(0.0, float(min_silence_seconds)))))
    min_cut_samples = max(1, int(round(sample_rate * max(0.0, float(min_cut_seconds)))))
    threshold = db_to_amplitude(float(silence_threshold_db))

    amplitude = detection_amplitude(waveform)
    silence_runs = find_silence_runs(amplitude, threshold=threshold, min_samples=min_silence_samples)
    keep_ranges, silences = build_keep_ranges(
        sample_count=sample_count,
        silence_runs=silence_runs,
        keep_ratio=keep_ratio,
        max_keep_samples=max_keep_samples,
        min_cut_samples=min_cut_samples,
    )

    if keep_ranges:
        pieces = [waveform[..., start:end] for start, end in keep_ranges]
        torch = _torch_module()
        if torch is not None and isinstance(waveform, torch.Tensor):
            trimmed_waveform = torch.cat(pieces, dim=-1).contiguous()
        else:
            trimmed_waveform = np.concatenate(pieces, axis=-1)
    else:
        trimmed_waveform = waveform[..., :0]
        if hasattr(trimmed_waveform, "contiguous"):
            trimmed_waveform = trimmed_waveform.contiguous()

    removed_samples = sample_count - int(trimmed_waveform.shape[-1])
    report = {
        "input_seconds": round(sample_count / sample_rate, 6) if sample_rate else 0.0,
        "output_seconds": round(int(trimmed_waveform.shape[-1]) / sample_rate, 6) if sample_rate else 0.0,
        "removed_seconds": round(removed_samples / sample_rate, 6) if sample_rate else 0.0,
        "detected_silences": len(silence_runs),
        "trimmed_silences": sum(1 for item in silences if item["trimmed"]),
        "settings": {
            "formula": "kept_silence_seconds = min(original_silence_seconds * keep_ratio, max_keep_seconds)",
            "keep_ratio": keep_ratio,
            "max_keep_seconds": float(max_keep_seconds),
            "silence_threshold_db": float(silence_threshold_db),
            "min_silence_seconds": float(min_silence_seconds),
            "min_cut_seconds": float(min_cut_seconds),
        },
        "silences": silences,
    }
    return {"waveform": trimmed_waveform, "sample_rate": sample_rate}, json.dumps(report, ensure_ascii=False, indent=2)


class AudioSilenceTrim:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"display_name": "Audio"}),
                "keep_ratio": (
                    "FLOAT",
                    {"default": DEFAULT_KEEP_RATIO, "min": 0.0, "max": 1.0, "step": 0.05, "display_name": "Silence keep ratio"},
                ),
                "max_keep_seconds": (
                    "FLOAT",
                    {"default": DEFAULT_MAX_KEEP_SECONDS, "min": 0.01, "max": 2.0, "step": 0.01, "display_name": "Max keep seconds"},
                ),
                "silence_threshold_db": (
                    "FLOAT",
                    {"default": DEFAULT_SILENCE_THRESHOLD_DB, "min": -80.0, "max": -10.0, "step": 1.0, "display_name": "Silence threshold dB"},
                ),
                "min_silence_seconds": (
                    "FLOAT",
                    {"default": DEFAULT_MIN_SILENCE_SECONDS, "min": 0.01, "max": 2.0, "step": 0.01, "display_name": "Min silence seconds"},
                ),
                "min_cut_seconds": (
                    "FLOAT",
                    {"default": DEFAULT_MIN_CUT_SECONDS, "min": 0.0, "max": 1.0, "step": 0.01, "display_name": "Min cut seconds"},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    FUNCTION = "trim"
    CATEGORY = "Audio"
    DESCRIPTION = "Shortens long silent sections while preserving a configurable portion of each silence."

    def trim(
        self,
        audio: dict[str, Any],
        keep_ratio: float,
        max_keep_seconds: float,
        silence_threshold_db: float,
        min_silence_seconds: float,
        min_cut_seconds: float,
    ):
        return trim_audio(
            audio,
            keep_ratio=keep_ratio,
            max_keep_seconds=max_keep_seconds,
            silence_threshold_db=silence_threshold_db,
            min_silence_seconds=min_silence_seconds,
            min_cut_seconds=min_cut_seconds,
        )


NODE_CLASS_MAPPINGS = {
    "AudioSilenceTrim": AudioSilenceTrim,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSilenceTrim": "Audio Silence Trim",
}
