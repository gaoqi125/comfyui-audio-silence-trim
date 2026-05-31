# Security Policy

This project is a local ComfyUI custom node and should not require credentials.

Please do not open issues that include private media, voice recordings, generated assets, local absolute paths, API keys, or provider tokens.

Safe reports should include:

- ComfyUI version
- Python version
- input waveform shape
- sample rate
- node settings
- redacted error logs

If a future change introduces file IO, network access, or provider integration, it should be discussed before implementation and covered by tests.
