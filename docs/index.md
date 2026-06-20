# Sol-LTX-Infer

**Accelerated video-diffusion inference — SANA-Video · Cosmos3-Super · LTX-2.3.**

Three production video-diffusion models, each reduced to one clean acceleration
line (plus a dense baseline), composed from five reusable acceleration methods.

## Documentation

- **Pipelines** — the three optimization pipelines (designs + speedups):
  [human](acceleration/human/pipelines.html) · [agents](acceleration/agents/pipelines.html)
- **Methods** — the five acceleration building blocks:
  [human](acceleration/human/methods.html) · [agents](acceleration/agents/methods.html)

## Run

See `QUICKSTART.md` (human) and `AGENTS.md` (agent/portable deploy) in the repo root.

| Model | Acceleration line | Speedup (GB200, warm) |
|---|---|---|
| SANA-Video (2B) | EasyCache + fusion + compile | 2.1× / 2.56× |
| Cosmos3-Super (64B) | TeaCache + step-selective NVFP4 | ~2.26× |
| LTX-2.3 (1080p/10s) | KWL fusion + cache + PISA + NVFP4 + token-prune | ~2.4× |
