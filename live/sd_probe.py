"""
An instrumented Stable Diffusion sampler.

This is the same loop `StableDiffusionPipeline.__call__` runs, written out by hand so that
every intermediate tensor can be captured and streamed to the browser:

    text  ──CLIP──►  prompt embeddings ─┐
                                        ▼
    z_T ~ N(0, I) ──► for each timestep t:
                          ε̂∅, ε̂ = UNet(z_t, t, [negative, prompt])     (one batched pass)
                          ε̂_cfg  = ε̂∅ + w · (ε̂ − ε̂∅)                   classifier-free guidance
                          x̂₀     = (z_t − √(1−ᾱ_t) · ε̂_cfg) / √ᾱ_t        the model's current guess
                          z_{t−1} = scheduler.step(ε̂_cfg, t, z_t)
                  ──VAE──► pixels

Hooks inside the UNet also record the timestep embedding, a slice of the bottleneck
activations and every cross-attention map (which image regions look at which word).
"""

from __future__ import annotations

import io
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ATTN_RES = 32  # every cross-attention layer is resampled to this grid before averaging
MID_CHANNELS = 64  # how many of the 1280 bottleneck channels to send

Emit = Callable[[str, dict, dict, dict], None]  # (kind, meta, float32 tensors, binary blobs)


def pick_device() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def near_square(n: int) -> tuple[int, int]:
    """Factor n = h·w with h ≤ w as close to square as possible (768 → 24×32)."""
    h = int(math.isqrt(n))
    while n % h:
        h -= 1
    return h, n // h


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def to_jpeg(img: torch.Tensor, quality: int = 88) -> bytes:
    """[3,H,W] in [-1, 1] → JPEG bytes."""
    arr = ((img.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float32, copy=False)


# ---------------------------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------------------------


class Probe:
    """Collects tensors from inside one UNet forward pass."""

    def __init__(self) -> None:
        self.enabled = False
        self.latent_hw = (64, 64)
        self.cond_index = 1  # batch is [negative, prompt]
        self.keep_tokens = 77
        self.reset()

    def reset(self) -> None:
        self.attn_sum: torch.Tensor | None = None
        self.attn_layers = 0
        self.tfeat = self.temb = self.mid = None

    def add_cross_attention(self, probs: torch.Tensor, batch: int, heads: int) -> None:
        """probs: [batch·heads, Nq, Nk]  →  accumulate a [keep_tokens, ATTN_RES, ATTN_RES] map."""
        nq, nk = probs.shape[1], probs.shape[2]
        lh, lw = self.latent_hw
        f = math.sqrt(lh * lw / nq)
        h, w = round(lh / f), round(lw / f)
        p = probs.view(batch, heads, nq, nk)[self.cond_index].float().mean(0)  # [Nq, Nk], avg over heads
        p = p[:, : self.keep_tokens].T.reshape(1, -1, h, w)
        mode = "area" if h > ATTN_RES else "bilinear"
        p = F.interpolate(p, size=(ATTN_RES, ATTN_RES), mode=mode, **({} if mode == "area" else {"align_corners": False}))
        self.attn_sum = p[0] if self.attn_sum is None else self.attn_sum + p[0]
        self.attn_layers += 1

    def attention(self) -> torch.Tensor | None:
        return None if self.attn_sum is None else self.attn_sum / self.attn_layers


class CrossAttnProbe:
    """Drop-in for diffusers' AttnProcessor that also hands the softmax(QKᵀ/√d) matrix to the probe."""

    def __init__(self, probe: Probe) -> None:
        self.probe = probe

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if encoder_hidden_states is not None and attn.norm_cross:
            context = attn.norm_encoder_hidden_states(context)
        attention_mask = attn.prepare_attention_mask(attention_mask, context.shape[1], batch_size)

        q = attn.head_to_batch_dim(attn.to_q(hidden_states))  # [B·heads, image tokens, d]
        k = attn.head_to_batch_dim(attn.to_k(context))  # [B·heads, 77 text tokens, d]
        v = attn.head_to_batch_dim(attn.to_v(context))
        probs = attn.get_attention_scores(q, k, attention_mask)  # softmax(QKᵀ/√d)
        if self.probe.enabled and encoder_hidden_states is not None:
            self.probe.add_cross_attention(probs, batch_size, attn.heads)

        out = attn.batch_to_head_dim(torch.bmm(probs, v))
        out = attn.to_out[1](attn.to_out[0](out))
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


# ---------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------


@dataclass
class GenParams:
    prompt: str = "a lighthouse on a grassy hill by the sea at sunset"
    negative: str = ""
    steps: int = 25
    guidance: float = 7.5
    seed: int = 1
    scheduler: str = "ddim"  # "ddim" | "dpm++"

    @classmethod
    def from_dict(cls, d: dict) -> "GenParams":
        p = cls()
        p.prompt = str(d.get("prompt", p.prompt))[:1000]
        p.negative = str(d.get("negative", p.negative))[:1000]
        p.steps = max(1, min(100, int(d.get("steps", p.steps))))
        p.guidance = max(0.0, min(30.0, float(d.get("guidance", p.guidance))))
        p.seed = int(d.get("seed", p.seed)) & 0x7FFFFFFF
        p.scheduler = d.get("scheduler", p.scheduler) if d.get("scheduler") in ("ddim", "dpm++") else "ddim"
        return p


class LiveDiffusion:
    def __init__(self, model_id: str, preview_id: str = "madebyollin/taesd", size: int = 512) -> None:
        self.model_id, self.preview_id, self.size = model_id, preview_id, size
        self.device, self.dtype = pick_device()
        self.ready = False
        self.status = "starting"
        self.lock = threading.Lock()  # one generation at a time on the GPU

    # -- loading ---------------------------------------------------------------------------

    def load(self) -> None:
        from diffusers import AutoencoderKL, AutoencoderTiny, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        dev, dt, mid = self.device, self.dtype, self.model_id
        fp16 = {"variant": "fp16"} if dt == torch.float16 else {}

        def step(msg: str) -> None:
            self.status = msg
            print(f"[load] {msg}", flush=True)

        step("tokenizer + CLIP text encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained(mid, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(mid, subfolder="text_encoder", torch_dtype=dt, **fp16).to(dev)
        step("UNet")
        self.unet = UNet2DConditionModel.from_pretrained(mid, subfolder="unet", torch_dtype=dt, **fp16).to(dev)
        step("VAE (full decoder)")
        # fp16 weights, but decode in fp32: the full VAE occasionally overflows in half precision
        self.vae = AutoencoderKL.from_pretrained(mid, subfolder="vae", torch_dtype=dt, **fp16).to(dev, torch.float32)
        step("TAESD (fast preview decoder)")
        self.taesd = AutoencoderTiny.from_pretrained(self.preview_id, torch_dtype=dt).to(dev)
        self.sched_config = DDIMScheduler.load_config(mid, subfolder="scheduler")

        for m in (self.text_encoder, self.unet, self.vae, self.taesd):
            m.eval().requires_grad_(False)

        self.prediction_type = self.sched_config.get("prediction_type", "epsilon")
        self.latent_channels = self.unet.config.in_channels
        self.vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.latent_size = self.size // self.vae_scale

        # probes: cross-attention on every attn2, forward hooks on the time embedding and the bottleneck
        self.probe = Probe()
        self.probe.latent_hw = (self.latent_size, self.latent_size)
        procs = {}
        for name, proc in self.unet.attn_processors.items():
            procs[name] = CrossAttnProbe(self.probe) if name.endswith("attn2.processor") else proc
        self.n_cross_layers = sum(1 for n in procs if n.endswith("attn2.processor"))
        self.unet.set_attn_processor(procs)  # note: diffusers pops entries out of `procs`

        def hook(attr: str, fn: Callable[[torch.Tensor], torch.Tensor]):
            def _h(_m, _inp, out):
                if self.probe.enabled:
                    setattr(self.probe, attr, fn(out))
            return _h

        self.unet.time_proj.register_forward_hook(hook("tfeat", lambda o: o[0].detach()))
        self.unet.time_embedding.register_forward_hook(hook("temb", lambda o: o[0].detach()))
        self.unet.mid_block.register_forward_hook(
            hook("mid", lambda o: o[self.probe.cond_index, :MID_CHANNELS].detach())
        )

        step("warm-up pass")
        self._warmup()
        self.ready = True
        step("ready")

    @torch.no_grad()
    def _warmup(self) -> None:
        z = torch.zeros(2, self.latent_channels, self.latent_size, self.latent_size, device=self.device, dtype=self.dtype)
        ctx = self.encode_text("")[0].repeat(2, 1, 1)
        self.unet(z, 999, encoder_hidden_states=ctx)
        self.taesd.decode(z[:1])
        sync(self.device)

    def info(self) -> dict:
        info = {
            "model": self.model_id,
            "preview": self.preview_id,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "ready": self.ready,
            "status": self.status,
        }
        if self.ready:
            emb = self.text_encoder.config.hidden_size
            temb = self.unet.config.block_out_channels[0] * 4
            info.update(
                size=self.size,
                latent=[self.latent_channels, self.latent_size, self.latent_size],
                vae_scale=self.vae_scale,
                text_dim=emb,
                text_grid=near_square(emb),
                tfeat_grid=near_square(self.unet.config.block_out_channels[0]),
                temb_grid=near_square(temb),
                mid=[MID_CHANNELS, self.latent_size // 8, self.latent_size // 8],
                mid_total=self.unet.config.block_out_channels[-1],
                attn_res=ATTN_RES,
                cross_layers=self.n_cross_layers,
                prediction_type=self.prediction_type,
                train_timesteps=self.sched_config.get("num_train_timesteps", 1000),
            )
        return info

    # -- pieces ----------------------------------------------------------------------------

    def tokenize(self, text: str) -> tuple[torch.Tensor, list[str], int]:
        tok = self.tokenizer
        ids = tok(text, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
        row = ids[0].tolist()
        n = row.index(tok.eos_token_id) + 1 if tok.eos_token_id in row else len(row)
        words = []
        for i, t in enumerate(tok.convert_ids_to_tokens(row[:n])):
            if i == 0:
                words.append("⟨start⟩")
            elif row[i] == tok.eos_token_id:
                words.append("⟨end⟩")
            else:
                words.append(t.replace("</w>", ""))
        return ids, words, n

    @torch.no_grad()
    def encode_text(self, text: str) -> tuple[torch.Tensor, list[str], int]:
        ids, words, n = self.tokenize(text)
        emb = self.text_encoder(ids.to(self.device))[0]  # [1, 77, 768]
        return emb, words, n

    def make_scheduler(self, name: str):
        from diffusers import DDIMScheduler, DPMSolverMultistepScheduler

        if name == "dpm++":
            return DPMSolverMultistepScheduler.from_config(self.sched_config, algorithm_type="dpmsolver++")
        return DDIMScheduler.from_config(self.sched_config)

    def split_prediction(self, out: torch.Tensor, z: torch.Tensor, abar: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Model output → (ε̂, x̂₀), for both ε- and v-prediction checkpoints."""
        a, s = math.sqrt(abar), math.sqrt(1 - abar)
        if self.prediction_type == "v_prediction":
            return s * z + a * out, a * z - s * out
        return out, (z - s * out) / a

    # -- the loop --------------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, p: GenParams, emit: Emit, cancel: threading.Event) -> None:
        with self.lock:
            self._generate(p, emit, cancel)

    def _generate(self, p: GenParams, emit: Emit, cancel: threading.Event) -> None:
        dev, dt = self.device, self.dtype
        t_start = time.perf_counter()

        cond, words, n_cond = self.encode_text(p.prompt)
        uncond, neg_words, n_neg = self.encode_text(p.negative)
        context = torch.cat([uncond, cond])  # batch row 0 = negative/empty, row 1 = prompt
        self.probe.keep_tokens = n_cond

        sched = self.make_scheduler(p.scheduler)
        sched.set_timesteps(p.steps, device=dev)
        timesteps = [int(t) for t in sched.timesteps]
        abar_all = sched.alphas_cumprod
        gen = torch.Generator("cpu").manual_seed(p.seed)
        shape = (1, self.latent_channels, self.latent_size, self.latent_size)
        z = torch.randn(shape, generator=gen, dtype=torch.float32).to(dev) * sched.init_noise_sigma

        gh, gw = near_square(cond.shape[-1])
        emit(
            "start",
            {
                "params": p.__dict__,
                "tokens": words,
                "neg_tokens": neg_words,
                "timesteps": timesteps,
                "abar": [float(abar_all[t]) for t in timesteps],
                "prediction_type": self.prediction_type,
            },
            {
                "text": f32(cond[0, :n_cond].reshape(n_cond, gh, gw)),
                "text_neg": f32(uncond[0, :n_neg].reshape(n_neg, gh, gw)),
                "noise": f32(z[0]),
            },
            {},
        )

        for k, t in enumerate(sched.timesteps):
            if cancel.is_set():
                emit("cancelled", {"k": k}, {}, {})
                return
            ts = time.perf_counter()
            abar = float(abar_all[int(t)])

            # 1. one UNet call on a batch of two: [negative, prompt]
            self.probe.reset()
            self.probe.enabled = True
            z_in = sched.scale_model_input(torch.cat([z, z]), t).to(dt)
            out = self.unet(z_in, t, encoder_hidden_states=context).sample.float()
            self.probe.enabled = False
            out_u, out_c = out.chunk(2)

            # 2. classifier-free guidance, then convert to noise / clean-image estimates
            out_g = out_u + p.guidance * (out_c - out_u)
            eps_u, x0_u = self.split_prediction(out_u, z, abar)
            eps_c, x0_c = self.split_prediction(out_c, z, abar)
            eps_g, x0 = self.split_prediction(out_g, z, abar)

            # 3. the scheduler moves z one step toward t = 0
            z_next = sched.step(out_g, t, z).prev_sample
            sync(dev)
            unet_ms = (time.perf_counter() - ts) * 1000

            # 4. cheap previews of where we are and where the model thinks we're going
            tp = time.perf_counter()
            imgs = self.taesd.decode(torch.cat([z, x0]).to(dt)).sample
            blobs = {"zimg": to_jpeg(imgs[0]), "x0img": to_jpeg(imgs[1])}
            preview_ms = (time.perf_counter() - tp) * 1000

            attn = self.probe.attention()
            tensors = {
                "z": f32(z[0]),
                "eps_u": f32(eps_u[0]),
                "eps_c": f32(eps_c[0]),
                "eps_g": f32(eps_g[0]),
                "x0": f32(x0[0]),
                "x0_u": f32(x0_u[0]),
                "x0_c": f32(x0_c[0]),
                "z_next": f32(z_next[0]),
                "tfeat": f32(self.probe.tfeat.reshape(1, *near_square(self.probe.tfeat.numel()))),
                "temb": f32(self.probe.temb.reshape(1, *near_square(self.probe.temb.numel()))),
                "mid": f32(self.probe.mid),
            }
            if attn is not None:
                tensors["attn"] = f32(attn)
            emit(
                "step",
                {
                    "k": k,
                    "n": len(timesteps),
                    "t": int(t),
                    "abar": abar,
                    "abar_next": float(abar_all[timesteps[k + 1]]) if k + 1 < len(timesteps) else 1.0,
                    "unet_ms": round(unet_ms, 1),
                    "preview_ms": round(preview_ms, 1),
                    "cross_layers": self.probe.attn_layers,
                },
                tensors,
                blobs,
            )
            z = z_next

        # 5. the full VAE decoder turns the final latent into the real image
        td = time.perf_counter()
        img = self.vae.decode(z.float() / self.vae.config.scaling_factor).sample
        sync(dev)
        emit(
            "done",
            {"decode_ms": round((time.perf_counter() - td) * 1000, 1), "total_s": round(time.perf_counter() - t_start, 2)},
            {"z_final": f32(z[0])},
            {"final": to_jpeg(img[0], quality=93)},
        )
