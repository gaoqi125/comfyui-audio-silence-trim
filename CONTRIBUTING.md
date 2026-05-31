# Contributing

Thanks for improving this ComfyUI audio utility.

## Development

Run the test suite before opening a pull request:

```bash
python3 -m unittest discover -s tests -v
```

Keep changes focused on the audio silence trimming node. Avoid adding provider-specific services, credentials, private workflow paths, or generated media assets to this repository.

## Pull Requests

Please include:

- A short description of the behavior change.
- Tests for new behavior or bug fixes.
- Notes about ComfyUI compatibility if the node API changes.
