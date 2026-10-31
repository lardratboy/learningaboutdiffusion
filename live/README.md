# Diffusion, live

Stable Diffusion 1.5 runs locally on your Mac's GPU. Every tensor of every denoising step is
streamed to a 3D browser viewer while it is computed. This is the follow-up to
[`../diffusion-walk.html`](../diffusion-walk.html): the same visual language (one cube per
number), but the data comes from a real trained model instead of a toy.

```bash
cd live
uv run server.py          # first run downloads ~2 GB of weights into ~/.cache/huggingface
open http://127.0.0.1:8765
```

Type a prompt and press **Generate** (or space). On an M-series Mac a 25-step image takes about
12 s: roughly 0.4 s per UNet step plus about 35 ms for the preview decode.

## What streams per step

| tensor | shape | what it is |
|---|---|---|
| `z` | 4 × 64 × 64 | the noisy latent z_t, the sampler's only state |
| `eps_u`, `eps_c` | 4 × 64 × 64 | predicted noise for the negative prompt and for your prompt (one batched UNet pass) |
| `eps_g` | 4 × 64 × 64 | after classifier-free guidance: ε̂∅ + w·(ε̂ − ε̂∅) |
| `x0`, `x0_u`, `x0_c` | 4 × 64 × 64 | the model's guess of the clean latent, (z − √(1−ᾱ)·ε̂)/√ᾱ |
| `z_next` | 4 × 64 × 64 | after `scheduler.step` |
| `attn` | tokens × 32 × 32 | cross-attention probabilities image→word, averaged over all heads of all 16 cross-attention layers |
| `tfeat`, `temb` | 320, 1280 | sinusoidal timestep features, and the timestep embedding after the MLP |
| `mid` | 64 × 8 × 8 | first 64 of the 1,280 channels at the UNet bottleneck |
| `zimg`, `x0img` | JPEG 512² | TAESD previews of z_t and x̂₀ |

The prompt and negative-prompt CLIP embeddings (tokens × 768) are sent once per run. The final
image goes through the full VAE decoder at the end.

## How it works

- **Device.** PyTorch's `mps` backend runs on Apple's Metal GPU API, in fp16. Core ML
  (`apple/ml-stable-diffusion`) is faster, but it compiles the network into a sealed graph,
  so intermediate tensors can't be read out.
- **`sd_probe.py`** writes out the sampling loop that `StableDiffusionPipeline` normally hides.
  It swaps every `attn2` (cross-attention) processor for one that records `softmax(QKᵀ/√d)`,
  and adds forward hooks on `time_proj`, `time_embedding` and `mid_block`.
- **`server.py`** is a FastAPI app. It serves the page and a WebSocket at `/ws`. Each event is
  one binary frame: `[u32 header length][JSON header][float32 tensors…][JPEG blobs…]`. The
  browser wraps each tensor in a `Float32Array` without copying.
- **`web/index.html`** is a single file, Three.js from a CDN, no build step.

## Options

```bash
uv run server.py --model <hf-repo>   # any SD 1.x/2.x checkpoint in diffusers format (ε- or v-prediction)
uv run server.py --port 9000 --size 512
```

The page also accepts URL parameters: `?prompt=…&steps=30&guidance=7.5&seed=42&scheduler=dpm%2B%2B&auto=1`.

## How SD 1.5 differs from the walk-through

The walk-through teaches the newer recipe (SD3/Flux): a DiT transformer, rectified flow and
velocity prediction. SD 1.5 uses a convolutional **UNet** that predicts **noise**, on a
variance-preserving schedule `z_t = √ᾱ·z₀ + √(1−ᾱ)·ε`. Text enters through cross-attention
rather than a joint token sequence. Classifier-free guidance, x̂₀ and the sampling loop work
the same way. Running SD3 would need a gated Hugging Face licence and about 4× the memory.
