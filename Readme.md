
# Qwen3

**Disclaimer:** This isn’t my original code, it’s copied from [LLMs from scratch](https://github.com/rasbt/LLMs-from-scratch) by @rasbt with a few tweaks.

This is my modified version of a Qwen3 implementation, originally from [LLMs from scratch](https://github.com/rasbt/LLMs-from-scratch) repo. I ripped out PyTorch and swapped in MLX because I wanted to peek under the hood and see what’s actually going on inside transformer.

## What I changed

- Swapped PyTorch for **MLX**. I started with NumPy, but it was way too slow, had to switch to MLX.
- Messed the code structure to make it clearer (at least to me) what each part actually does.
- Added **KV caching.**
- Added support for **X-bit quantization** (2 to 8 bits).
- Added proper output sampling so the model can generate diverse text, not just deterministic outputs.

## Important note

This is **inference** only, no training code here. Just the forward pass.

## Credits
  [@rasbt](https://github.com/rasbt/LLMs-from-scratch)

