import os, traceback
RANK = int(os.environ.get("SLURM_PROCID", "0")); WORLD = int(os.environ.get("SLURM_NTASKS", "1"))
LOCAL = int(os.environ.get("SLURM_LOCALID", "0"))
os.environ["CUDA_VISIBLE_DEVICES"] = str(LOCAL)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Default = the 11 dimensions the LTX-2 VBench run actually scored (had eval_results);
# scene + the 4 GRIT dims were env-blocked. Override via VBENCH_DIMS.
_DEFAULT_DIMS = ["imaging_quality", "aesthetic_quality", "subject_consistency",
                 "background_consistency", "temporal_flickering", "motion_smoothness",
                 "dynamic_degree", "overall_consistency", "temporal_style",
                 "human_action", "appearance_style"]
DIMS = (os.environ.get("VBENCH_DIMS", "").replace("+", ",").split(",")
        if os.environ.get("VBENCH_DIMS") else _DEFAULT_DIMS)
BATCHES = os.environ.get("VBENCH_BATCHES", "sana_dense,sana_fullopt").split(",")
NAMED_ROOT = os.environ.get("VBENCH_NAMED_ROOT", "/home/yitongl/code/vbench_sana/named")
SCORES_ROOT = os.environ.get("VBENCH_SCORES_ROOT", "/home/yitongl/code/vbench_sana/scores")
PAIRS = [(b, d) for b in BATCHES for d in DIMS]


def main():
    import sys, types, torch
    # fix human_action (UMT): numpy 2.x removed numpy.lib.function_base (only `disp` imported)
    if "numpy.lib.function_base" not in sys.modules:
        _m = types.ModuleType("numpy.lib.function_base")
        _m.disp = lambda mesg=None, device=None, linefeed=True: None
        sys.modules["numpy.lib.function_base"] = _m
    # fix motion_smoothness (AMT): torch 2.11 defaults weights_only=True
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": k.get("weights_only", False)})
    # fix scene (Tag2Text/med.py): transformers 5.x relocated/removed these utils
    import transformers.modeling_utils as _mu, transformers.pytorch_utils as _pu
    for _n in dir(_pu):
        if not _n.startswith("__") and not hasattr(_mu, _n):
            setattr(_mu, _n, getattr(_pu, _n))
    if not hasattr(_mu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index
        _mu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    _mu.PreTrainedModel.all_tied_weights_keys = property(lambda self: {})

    from vbench import VBench
    import vbench
    FI = os.path.join(os.path.dirname(vbench.__file__), "VBench_full_info.json")
    mine = PAIRS[RANK::WORLD]
    print(f"[rank {RANK}/{WORLD} gpu {LOCAL}] {len(mine)} (batch,dim) pairs: {mine}", flush=True)
    for b, d in mine:
        out = f"{SCORES_ROOT}/{b}"; os.makedirs(out, exist_ok=True)
        vids = f"{NAMED_ROOT}/{b}"
        try:
            import time; t = time.time()
            VBench(torch.device("cuda"), FI, out).evaluate(
                videos_path=vids, name=f"{b}__{d}", dimension_list=[d], mode="vbench_standard")
            print(f"[rank {RANK}] OK {b}/{d} {round(time.time()-t)}s", flush=True)
        except Exception as e:
            print(f"[rank {RANK}] FAIL {b}/{d}: {type(e).__name__} {str(e)[:200]}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
