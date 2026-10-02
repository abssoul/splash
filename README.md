# Splash for M1 and M2 Macs

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/paperniuk/splash)](https://github.com/paperniuk/splash/releases/latest)

An unofficial port of [Inco's Splash](https://github.com/incoai/splash) to
Macs with an M1 or M2 chip. Splash is a local inference engine that serves
Qwen3.8-27B and Qwen3.6-35B-A3B to coding agents and to any OpenAI or
Anthropic compatible client. Official Splash needs an M3 or newer; this fork
runs the same engine, version 1.1.0, on M1 and M2.

**On an M3 or newer, use official Splash:** `brew install incoai/tap/splash`.
Nothing here is faster there.

The engine, the models and the draft models are Inco's work. What this fork
adds:

- **It starts on M1/M2.** Upstream refuses GPUs older than the M3 family.
- **Metal kernels written for these GPUs.** Upstream's kernels use Apple's
  matrix library, which reaches 2.4 to 3.5 TFLOPS on an M1 Max and hangs the
  GPU on some GGUF shapes. The kernels here reach 7.3 TFLOPS on the same
  projections, and decode is about twice as fast as upstream's kernels on the
  same machine. They cover the 4-bit packages, GGUF files, attention and the
  vision encoder.
- **Long prompts in short GPU commands.** macOS aborts a GPU command that
  holds the GPU for too long, and on M1/M2 a full prefill chunk does. Prefill
  is split, so agent sessions do not end with an engine failure. See
  [Long prompts](#long-prompts).
- **A prebuilt release.** One line installs it, with no Xcode, Homebrew or pip.

Measured on an M1 Max with 64 GB. M1 Pro, M1 Ultra and M2 run the same code,
and owners of an M1 Ultra and an M2 Max have reported it working.

## Install

Needs macOS 26.4 or later.

```bash
curl -fsSL https://github.com/paperniuk/splash/releases/latest/download/install-m1.sh | sh
splash-m1 serve --model incoai/Qwen3.8-27B-Splash
```

This installs the command `splash-m1` and leaves an official `splash` alone.
The first run downloads the model and its draft and prepares the weights;
later starts reuse them. Once it prints `Ready`, open
<http://127.0.0.1:8000>, or start an installed agent from another terminal:

```bash
splash-m1 opencode    # or: splash-m1 claude / codex / hermes / pi
```

Running the same `curl` line again upgrades. Models live in the Hugging Face
cache and survive upgrades.

## Which model for your Mac

| Memory | Model | `--model` | Weights |
|---|---|---|---|
| 36 GB or more | Qwen3.8-27B, dense | `incoai/Qwen3.8-27B-Splash` | 16 GB |
| 36 GB or more | Qwen3.6-35B-A3B, MoE, fastest | `incoai/Qwen3.6-35B-A3B-Splash` | 21 GB |
| 36 GB or more | Qwen3.8-27B as GGUF, other quants | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` | 17 GB |
| 24 GB | a smaller GGUF quant | `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS` | |
| 16 GB | Ternary Bonsai 2, a 27B in 7 GB | `prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0` | 7 GB |

Everything here was measured on a 64 GB Mac. The memory column for the first
four rows is upstream's requirement. Bonsai was run with memory capped to
11 GB, where it starts with a 28K context; a stock 16 GB Mac gives the GPU
about 10.7 GB, slightly too little, so raise the limit first:

```bash
sudo sysctl iogpu.wired_limit_mb=12288   # resets at reboot
splash-m1 serve --model prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0 --language-only
```

`--language-only` skips vision and saves its memory. Reports from 16, 24 and
32 GB Macs are welcome.

MLX checkpoints (`mlx-community/Qwen3.8-27B-4bit` and the 35B one) load
through the same kernels as the packages, but have not been run end to end
on M1 yet. Every Unsloth GGUF of the two models loads, from UD-IQ1_S up,
except UD-Q8_K_XL and BF16.

## Speed on an M1 Max

M1 Max, 32-core GPU, 64 GB. Decode speed is the mean over the five prompts
of [npanj's benchmark](https://github.com/npanj/splash-plus), 250 output
tokens each, thinking on.

| Model | Decode | First token, cold 8K prompt |
|---|---:|---:|
| `incoai/Qwen3.6-35B-A3B-Splash` | 145 tok/s | not measured |
| `incoai/Qwen3.8-27B-Splash` | 38 tok/s | 61 s |
| `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` | 32 tok/s | 66 s |
| `prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0` | 28 tok/s | 63 s |

Decode depends on the text: on the 27B, code runs at 43 to 54 tok/s and
dense prose at 25 to 36. Upstream's kernels on the same Mac average 19 tok/s
on the 27B, against 38 here.

A cold prompt on the 27B is read at 142 tok/s for 2K tokens, 133 for 8K and
110 for 32K (290 s). It slows with length because attention over the prompt
grows. Splash keeps the prompt in a prefix cache, so an agent pays this once
per session: in a 25-request OpenCode session, 95% of prompt tokens came from
the cache.

On an M2 Max, [a user measured](https://github.com/incoai/splash/issues/131)
27B prefill at 151 to 157 tok/s with 10 to 20K of history, 111 to 122 at
40 to 50K and 82 to 90 at 80 to 95K.

Each model in the table answers all 54 problems of a physics and arithmetic
set correctly. Method and the other numbers are in the
[release notes](https://github.com/paperniuk/splash/releases).

## Flags that behave differently on M1/M2

Upstream's README describes every flag. These matter more here.

### Long prompts

`--prefill-mode bounded` is the default: long-context prefill is sent to the
GPU in commands of about 5 seconds. When one command holds the GPU for too
long while the display needs it, macOS aborts it (`ImpactingInteractivity`)
and the engine restarts. With the split, a 7-hour OpenCode session on an
M1 Max reached 229K tokens of context.

On an M2 Max with the desktop in use, 5 seconds was still too long at about
43K tokens. The [beta](https://github.com/paperniuk/splash/releases/tag/1.1.0-m1.1-beta1)
lowers the bound to 2 seconds, adds `--prefill-bound-ms`, and halves the
bound by itself after such an abort. On an M1 Max the shorter commands cost
under 2% of prefill speed. If you see engine failures at long context, use
the beta.

Do not pass `--prefill-mode full` on M1/M2. It restores upstream's behaviour,
which is what gets aborted.

### `--kv-format bf16` is slower here

The fast attention kernels of this fork exist for the default INT8 cache
only. With `--kv-format bf16` the engine uses upstream's attention kernel,
which is 1.7 to 1.9 times slower on M1/M2, and at long context attention is
most of the prefill time. Keep the default unless you are comparing quality.

### Memory

Model weights stay resident in GPU memory on M1/M2. Without that, memory
swings by the size of the weights between idle and busy, and on a 32 GB Mac
the prefix cache was refused (`cached 0`).

## API and settings

The API is upstream's: OpenAI Chat Completions, OpenAI Responses and
Anthropic Messages on `127.0.0.1:8000`, with streaming, tool calls, images
and PDFs. `splash-m1 serve --help` lists the options. For the rest, read
[upstream's README](https://github.com/incoai/splash/blob/1.1.0/README.md)
with `splash-m1` in place of `splash`, and [DEVELOPMENT.md](DEVELOPMENT.md).

Reasoning is turned off per request with `"reasoning_effort": "none"`.

## Build from source

Needs Xcode with the Metal Toolchain.

```bash
git clone https://github.com/paperniuk/splash.git
cd splash
make -j4
./splash serve --model incoai/Qwen3.8-27B-Splash
```

The default branch, `m1`, is the port. `main` follows upstream and does not
run on M1/M2.

## Status

- Stable: [1.1.0-m1](https://github.com/paperniuk/splash/releases/latest),
  upstream 1.1.0 plus the M1/M2 work.
- Beta: [1.1.0-m1.1-beta1](https://github.com/paperniuk/splash/releases/tag/1.1.0-m1.1-beta1),
  the shorter prefill commands above.
- Known gaps: no fast attention kernel for the BF16 cache; MLX checkpoints
  untested end to end; nothing below 64 GB measured on real hardware.

Problems on M1/M2 belong in [this fork's issues](https://github.com/paperniuk/splash/issues),
not upstream's.

## Credits

Splash is built by [Inco](https://github.com/incoai/splash); the
[launch post](https://inco.ai/blog/splash/) explains its design. If this port
is useful to you, please star the original too. The installer is based on
npanj's `install-q8.sh`.

Apache-2.0, see [LICENSE](LICENSE); the GGUF kernels include MIT-licensed
material from llama.cpp, see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
Model weights keep their own licenses.
