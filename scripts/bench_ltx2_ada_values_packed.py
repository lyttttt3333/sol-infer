import argparse
import json
from pathlib import Path

import torch

from sglang.jit_kernel.diffusion.triton.ltx2_ada_values import (
    ltx2_ada_values9,
    ltx2_ada_values9_packed,
)


def time_cuda(fn, repeats, warmup):
    last=None
    for _ in range(warmup): last=fn()
    torch.cuda.synchronize(); st=torch.cuda.Event(True); en=torch.cuda.Event(True); ts=[]
    for _ in range(repeats):
        torch.cuda.synchronize(); st.record(); last=fn(); en.record(); torch.cuda.synchronize(); ts.append(float(st.elapsed_time(en)))
    return last, ts


def stats(ts): return {"avg_ms":sum(ts)/len(ts),"min_ms":min(ts),"ms":ts}


def cmp(a,b):
    out={}
    for i,(x,y) in enumerate(zip(a,b)):
        d=(x.float()-y.float()).abs(); out[f'out{i}_max_abs_diff']=float(d.max()); out[f'out{i}_mean_abs_diff']=float(d.mean()); out[f'out{i}_packed_contiguous']=bool(y.is_contiguous())
    return out


def bench(name,b,s,h,repeats,warmup):
    torch.manual_seed(7+b+s+h)
    table=torch.randn((9,h),device='cuda',dtype=torch.bfloat16)
    timestep=torch.randn((b,s,9*h),device='cuda',dtype=torch.bfloat16)
    base,bt=time_cuda(lambda: ltx2_ada_values9(table,timestep),repeats,warmup)
    packed,pt=time_cuda(lambda: ltx2_ada_values9_packed(table,timestep),repeats,warmup)
    ba=sum(bt)/len(bt); pa=sum(pt)/len(pt)
    return {"shape":{"batch":b,"seq":s,"hidden":h},"current_all9":stats(bt),"packed_all9":stats(pt),"speedup":ba/pa,**cmp(base,packed)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='outputs/ltx23-ada-values-packed-microbench/result.json'); ap.add_argument('--repeats',type=int,default=8); ap.add_argument('--warmup',type=int,default=4); args=ap.parse_args()
    torch.cuda.set_device(0)
    payload={"device":torch.cuda.get_device_name(0),"results":{"stage1_video_b3":bench('stage1_video_b3',3,15810,4096,args.repeats,args.warmup),"stage1_video_b6":bench('stage1_video_b6',6,15810,4096,args.repeats,args.warmup),"stage2_video":bench('stage2_video',1,63240,4096,args.repeats,args.warmup),"stage1_audio":bench('stage1_audio',3,251,2048,args.repeats,args.warmup),"stage2_audio":bench('stage2_audio',1,251,2048,args.repeats,args.warmup)}}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2),flush=True)
if __name__=='__main__': main()
