import argparse
import json
from pathlib import Path

import torch


def time_cuda(fn, repeats, warmup):
    last=None
    for _ in range(warmup): last=fn()
    torch.cuda.synchronize(); st=torch.cuda.Event(True); en=torch.cuda.Event(True); ts=[]
    for _ in range(repeats):
        torch.cuda.synchronize(); st.record(); last=fn(); en.record(); torch.cuda.synchronize(); ts.append(float(st.elapsed_time(en)))
    return last, ts


def stats(ts): return {"avg_ms":sum(ts)/len(ts),"min_ms":min(ts),"ms":ts}


def bench(name,b,s,h,d,repeats,warmup):
    torch.manual_seed(123)
    out_base=torch.randn((b,s,h,d),device='cuda',dtype=torch.bfloat16)
    gate_logits=torch.randn((b,s,h),device='cuda',dtype=torch.bfloat16)
    def eager():
        out=out_base.clone()
        return out * (2.0 * torch.sigmoid(gate_logits).unsqueeze(-1))
    def inplace_gate():
        out=out_base.clone()
        gate=torch.sigmoid(gate_logits).unsqueeze(-1)
        gate.mul_(2.0)
        out.mul_(gate)
        return out
    def sigmoid_out_variant():
        out=out_base.clone()
        gate=torch.sigmoid(gate_logits).mul_(2.0).unsqueeze(-1)
        return out.mul_(gate)
    eo,et=time_cuda(eager,repeats,warmup)
    io,it=time_cuda(inplace_gate,repeats,warmup)
    so,st=time_cuda(sigmoid_out_variant,repeats,warmup)
    di=(eo.float()-io.float()).abs(); ds=(eo.float()-so.float()).abs()
    return {"shape":{"b":b,"s":s,"h":h,"d":d},"eager":stats(et),"inplace_gate":stats(it),"inplace_speedup":(sum(et)/len(et))/(sum(it)/len(it)),"inplace_max_abs_diff":float(di.max()),"inplace_mean_abs_diff":float(di.mean()),"sigmoid_out_variant":stats(st),"sigmoid_out_speedup":(sum(et)/len(et))/(sum(st)/len(st)),"sigmoid_out_max_abs_diff":float(ds.max()),"sigmoid_out_mean_abs_diff":float(ds.mean())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='outputs/ltx23-attention-gate-apply-microbench/result.json'); ap.add_argument('--repeats',type=int,default=8); ap.add_argument('--warmup',type=int,default=4); args=ap.parse_args()
    torch.cuda.set_device(0)
    payload={"device":torch.cuda.get_device_name(0),"results":{"stage1_video":bench('stage1_video',3,15810,32,128,args.repeats,args.warmup),"stage2_video":bench('stage2_video',1,63240,32,128,args.repeats,args.warmup),"stage1_audio":bench('stage1_audio',3,251,32,64,args.repeats,args.warmup),"stage2_audio":bench('stage2_audio',1,251,32,64,args.repeats,args.warmup)}}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2),flush=True)
if __name__=='__main__': main()
