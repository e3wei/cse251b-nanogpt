import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("ckpts", nargs="+")
args = parser.parse_args()

loaded = [torch.load(p, map_location="cpu", weights_only=False) for p in args.ckpts]

states = []
for ckpt in loaded:
    if not (isinstance(ckpt, dict) and "model" in ckpt and "config" in ckpt):
        raise ValueError("Expected checkpoint dict with keys: model, config")
    states.append(ckpt["model"])

avg = {}
for k in states[0].keys():
    v = states[0][k]
    if torch.is_tensor(v) and v.is_floating_point():
        avg[k] = sum(s[k].float() for s in states) / len(states)
        avg[k] = avg[k].to(dtype=v.dtype)
    else:
        avg[k] = states[-1][k]

out = {
    "model": avg,
    "config": loaded[-1]["config"],
}
torch.save(out, args.out)
print(f"saved {args.out} from {len(args.ckpts)} checkpoints")
