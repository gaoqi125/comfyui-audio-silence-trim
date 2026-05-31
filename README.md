# ComfyUI Audio Silence Trim

A small ComfyUI custom node that shortens long silent sections in `AUDIO` inputs while keeping a configurable portion of each detected silence.

## What It Does

- Detects silence from the mean absolute waveform amplitude.
- Keeps part of each silence using `keep_ratio`.
- Caps retained silence with `max_keep_seconds`.
- Leaves short cuts untouched when the removed duration is below `min_cut_seconds`.
- Returns both the trimmed `AUDIO` object and a JSON report.

## Install

Clone this repository into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/gaoqi125/comfyui-audio-silence-trim.git
```

Restart ComfyUI. The node appears as `Audio Silence Trim` in the `Audio` category.

## Test

```bash
python3 -m unittest discover -s tests -v
```

The tests use NumPy arrays and do not require a running ComfyUI instance.

## Defaults

- `keep_ratio`: `0.5`
- `max_keep_seconds`: `0.30`
- `silence_threshold_db`: `-38.0`
- `min_silence_seconds`: `0.18`
- `min_cut_seconds`: `0.05`
