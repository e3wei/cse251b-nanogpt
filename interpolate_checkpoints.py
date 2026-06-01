import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--a", required=True)
parser.add_argument("--b", required=True)
parser.add_argument("--alpha", type=float, required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

ckpt_a = torch.load(args.a, map_location="cpu", weights_only=False)
ckpt_b = torch.load(args.b, map_location="cpu", weights_only=False)

sa = ckpt_a["model"]
sb = ckpt_b["model"]

out_state = {}
for k in sa.keys():
    va, vb = sa[k], sb[k]
    if torch.is_tensor(va) and va.is_floating_point():
        out_state[k] = ((1 - args.alpha) * va.float() + args.alpha * vb.float()).to(dtype=va.dtype)
    else:
        out_state[k] = vb

torch.save({"model": out_state, "config": ckpt_b["config"]}, args.out)
print(f"saved {args.out}, alpha={args.alpha}")
