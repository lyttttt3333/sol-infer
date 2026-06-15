#!/usr/bin/env python3
"""Data-parallel (multi-node) SANA-Video generation over the VBench prompt set.
Each worker (1 GPU) builds the model once and generates its slice
prompts[rank::world] as VBench-named videos ("<prompt>-0.mp4"), 1/prompt, seed 42.

Config via GEN_CONFIG:
  dense   -> faithful baseline (fp32 linattn, no cache, no compile)
  fullopt -> EasyCache 0.1 + compile + linattn-bf16 (the 2.77x deliverable)

Multi-node: srun sets SLURM_PROCID (global rank) / SLURM_NTASKS / SLURM_LOCALID.
VBench matches videos to prompts by filename, so the filename is the RAW prompt;
SANA's motion-score convention is appended only to the generation prompt."""
import os
import time
import traceback

RANK = int(os.environ.get("SLURM_PROCID", "0"))
WORLD = int(os.environ.get("SLURM_NTASKS", "1"))
LOCAL = int(os.environ.get("SLURM_LOCALID", "0"))
os.environ["CUDA_VISIBLE_DEVICES"] = str(LOCAL)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Per-rank TMPDIR: 16 concurrent workers must not share a fixed temp dir
# (documented crossed-output race). Node-local /tmp keyed by global rank.
_tmp = f"/tmp/sana_vbench_{RANK}"
os.makedirs(_tmp, exist_ok=True)
os.environ["TMPDIR"] = _tmp

CONFIG = os.environ.get("GEN_CONFIG", "dense")
PROMPTS_FILE = os.environ.get("VBENCH_PROMPTS", "/home/yitongl/vbench_prompts.txt")
OUTDIR = os.environ.get("GEN_OUTDIR", f"/home/yitongl/code/vbench_sana/named/sana_{CONFIG}")
MODEL = os.environ.get("SANA_MODEL", "Efficient-Large-Model/SANA-Video_2B_480p_diffusers")
MOTION = os.environ.get("VBENCH_MOTION", "30")
LIMIT = int(os.environ.get("VBENCH_LIMIT", "0") or 0)  # smoke: cap prompts (0=all)

COMPILE = False
if CONFIG == "fullopt":
    os.environ["SGLANG_SANA_EASYCACHE_THRESH"] = "0.1"
    os.environ["SGLANG_SANA_LINATTN_BF16"] = "1"
    COMPILE = True
# dense: leave all SANA accel envs unset -> fp32 linattn, no cache, no compile.


def _resolve_local(model):
    import glob
    if os.path.isdir(model):
        return model
    cache = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    safe = "models--" + model.replace("/", "--")
    snaps = sorted(glob.glob(os.path.join(cache, safe, "snapshots", "*")))
    return snaps[-1] if snaps else model


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    prompts = [l.strip() for l in open(PROMPTS_FILE) if l.strip()]
    mine = prompts[RANK::WORLD]
    if LIMIT:
        mine = mine[:LIMIT]
    print(f"[rank {RANK}/{WORLD} gpu {LOCAL}] config={CONFIG} compile={COMPILE} "
          f"{len(mine)} prompts -> {OUTDIR}", flush=True)
    try:
        from sglang.multimodal_gen.registry import _discover_and_register_pipelines
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
        _discover_and_register_pipelines()
        gen = DiffGenerator.from_pretrained(
            model_path=_resolve_local(MODEL),
            pipeline_class_name="SanaVideoPipeline",
            num_gpus=1,
            dit_cpu_offload=False,
            enable_torch_compile=COMPILE,
            warmup=False,
            output_path=OUTDIR,
            master_port=30000 + LOCAL,  # per-node-unique: 4 workers/node must not collide
        )
    except Exception:
        traceback.print_exc()
        print(f"[rank {RANK}] BUILD_FAIL", flush=True)
        raise

    import glob
    done = skipped = 0
    t0 = time.time()
    for idx, p in enumerate(mine):
        fn = p.replace("/", " ")[:200] + "-0.mp4"  # EXACT VBench filename (spaces kept)
        dst = os.path.join(OUTDIR, fn)
        if os.path.exists(dst):
            skipped += 1
            done += 1
            continue
        tmp = f"gtmp{RANK}x{idx}"  # sanitization-safe stem (sglang mangles spaces/punct)
        try:
            gen.generate(sampling_params_kwargs=dict(
                prompt=f"{p} motion score: {MOTION}.",
                num_frames=81, num_inference_steps=50, height=480, width=832,
                guidance_scale=6.0, seed=42, output_file_name=tmp, save_output=True, fps=16))
            # sglang sanitizes output_file_name -> locate the produced file and rename
            # to the exact VBench prompt filename (VBench matches by exact prompt text).
            cand = sorted(glob.glob(os.path.join(OUTDIR, tmp + "*.mp4")))
            if cand:
                os.replace(cand[0], dst)
                done += 1
            else:
                print(f"[rank {RANK}] NO_OUTPUT {tmp}", flush=True)
            if done % 10 == 0:
                print(f"[rank {RANK}] {done}/{len(mine)} ({(time.time()-t0)/max(done-skipped,1):.1f}s/vid)", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"[rank {RANK}] GEN_FAIL {fn[:60]}", flush=True)
    print(f"[rank {RANK}] DONE {done}/{len(mine)} (skipped existing {skipped})", flush=True)


if __name__ == "__main__":
    main()
