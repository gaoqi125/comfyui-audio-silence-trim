# Support

Use GitHub Issues for reproducible bugs and GitHub Discussions for usage questions.

Before opening an issue, run:

```bash
python3 -m unittest discover -s tests -v
```

Supported maintenance scope:

- ComfyUI `AUDIO` input handling
- silence detection behavior
- trimming parameters and JSON reports
- compatibility with NumPy and optional PyTorch tensors

Out of scope:

- provider-specific voice services
- private media workflow debugging
- uploading sample recordings that contain personal data
