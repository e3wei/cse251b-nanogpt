import importlib.util
import inspect
import math
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# default config values (override with config file)
out_dir = "out"
eval_interval = 100
eval_iters = 40
log_interval = 10
always_save_checkpoint = False
init_from = "scratch"

dataset_dir = "."
train_bin = "train.bin"
val_bin = "val.bin"

gradient_accumulation_steps = 8
batch_size = 8
block_size = 1024

n_layer = 22
n_head = 8
n_embd = 512
bias = False
norm_type = "rmsnorm"
norm_eps = 1e-5
activation = "swiglu"
ffn_mult = 2.5
ffn_dim_multiple_of = 64
qk_norm = True
rope_base = 10000

learning_rate = 3e-4
min_lr = 3e-5
max_iters = 2000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
optimizer_type = "adamw"

warmup_iters = 100
lr_decay_iters = 2000
lr_schedule = "cosine"
wsd_cooldown_frac = 0.2
wsd_final_lr = 0.0
muon_lr = 0.02
muon_momentum = 0.95
muon_ns_steps = 5

seq_len_schedule = [(0, 256), (500, 512), (1200, 1024)]
early_stop_patience = 0
early_stop_min_delta = 0.0

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16"
compile = False
seed = 1337
# -----------------------------------------------------------------------------


def load_config_from_file(config_path):
    config_globals = {}
    with open(config_path, "r", encoding="utf-8") as f:
        exec(f.read(), {}, config_globals)
    for key, value in config_globals.items():
        if key in globals():
            globals()[key] = value


def apply_cli_overrides(args):
    for arg in args:
        if "=" not in arg:
            continue
        key, raw_val = arg.split("=", 1)
        if key not in globals():
            raise ValueError(f"unknown config key: {key}")
        current = globals()[key]
        if isinstance(current, bool):
            val = raw_val.lower() in {"1", "true", "yes"}
        elif isinstance(current, int):
            val = int(raw_val)
        elif isinstance(current, float):
            val = float(raw_val)
        else:
            val = raw_val
        globals()[key] = val


def cuda_arch_in_this_torch_build() -> bool:
    """True when this PyTorch binary likely has SASS for the default CUDA device.

    Prefer ``torch.cuda.get_arch_list()`` (e.g. includes ``sm_120`` for Blackwell on cu128 wheels)
    so we do not force CPU on newer GPUs that *are* supported. If the list is missing or empty,
    fall back to a conservative capability check (wheels that only shipped through sm_90).
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    gencode = f"sm_{major}{minor}"
    get_list = getattr(torch.cuda, "get_arch_list", None)
    if callable(get_list):
        built = get_list()
        if built:
            return gencode in built
    if major > 9 or (major == 9 and minor > 0):
        return False
    return True


def triton_importable() -> bool:
    """Inductor (default ``torch.compile`` backend) needs Triton for CUDA; often missing on Windows."""
    return importlib.util.find_spec("triton") is not None


def _newton_schulz_orthogonalize(update: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
    """Approximate orthogonalization used by Muon-style updates."""
    x = update.float()
    if x.numel() == 0:
        return update
    transposed = False
    if x.size(0) < x.size(1):
        x = x.t()
        transposed = True
    x = x / (x.norm() + eps)
    for _ in range(steps):
        xtx = x.transpose(0, 1) @ x
        x = 1.5 * x - 0.5 * (x @ xtx)
    if transposed:
        x = x.t()
    return x.to(dtype=update.dtype)


class Muon(Optimizer):
    """Minimal Muon optimizer for 2D matrix parameters."""

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, eps=1e-7):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g, alpha=1.0 - momentum)
                orth = _newton_schulz_orthogonalize(buf, ns_steps, eps=eps)
                p.add_(orth, alpha=-lr)
        return loss


extra_args = sys.argv[1:]
if extra_args and extra_args[0].endswith(".py"):
    load_config_from_file(extra_args[0])
    extra_args = extra_args[1:]
apply_cli_overrides(extra_args)

_dev = str(device).lower()
if _dev.startswith("cuda"):
    if not torch.cuda.is_available():
        print("warning: device=cuda requested but torch.cuda.is_available() is False; using cpu")
        device = "cpu"
        compile = False
    elif not cuda_arch_in_this_torch_build():
        major, minor = torch.cuda.get_device_capability()
        gencode = f"sm_{major}{minor}"
        print(
            f"warning: CUDA device is {gencode} but this PyTorch install is not treated as having "
            "kernels for it (check torch.cuda.get_arch_list(); for GPUs past sm_90 use a recent CUDA wheel, "
            "e.g. cu128+ for sm_120). Falling back to cpu."
        )
        device = "cpu"
        compile = False
        dtype = "float32"

torch.manual_seed(seed)
if device.startswith("cuda"):
    torch.cuda.manual_seed(seed)
device_type = "cuda" if device.startswith("cuda") else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type=device_type, dtype=ptdtype)

os.makedirs(out_dir, exist_ok=True)


class PackedDataLoader:
    def __init__(self, data, batch_size, world_size=1, rank=0):
        self.data = data
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank

    def get_batch(self, seq_len, device):
        max_start = len(self.data) - seq_len - 1
        if max_start <= 0:
            raise ValueError(f"dataset too small for seq_len={seq_len}")
        ix = torch.randint(0, max_start, (self.batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i : i + seq_len].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in ix])
        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y


train_path = os.path.join(dataset_dir, train_bin)
val_path = os.path.join(dataset_dir, val_bin)
if not os.path.exists(train_path):
    print(f"warning: {train_path} not found, using {val_path} for train sanity run")
    train_path = val_path
if not os.path.exists(val_path):
    raise FileNotFoundError(f"missing validation data: {val_path}")

train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
train_loader = PackedDataLoader(train_data, batch_size=batch_size)
val_loader = PackedDataLoader(val_data, batch_size=batch_size)

model_args = dict(
    block_size=block_size,
    vocab_size=50257,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    bias=bias,
    norm_type=norm_type,
    norm_eps=norm_eps,
    rope_base=rope_base,
    activation=activation,
    ffn_mult=ffn_mult,
    ffn_dim_multiple_of=ffn_dim_multiple_of,
    qk_norm=qk_norm,
)

resume_iter_num = 0
resume_best_val_loss = float("inf")
resume_optimizer_state = None

if init_from == "scratch":
    model = GPT(GPTConfig(**model_args))
elif init_from == "resume":
    ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"cannot resume: checkpoint not found at {ckpt_path}"
        )
    print(f"resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # allow the config file to override stored model_args
    ckpt_model_args = checkpoint["config"]
    ckpt_model_args.update({k: v for k, v in model_args.items() if k in ckpt_model_args})
    model = GPT(GPTConfig(**ckpt_model_args))
    model.load_state_dict(checkpoint["model"])
    resume_iter_num = checkpoint.get("iter_num", 0) + 1
    resume_best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    resume_optimizer_state = checkpoint.get("optimizer", None)
    print(f"  resumed at iter {resume_iter_num}, best_val_loss={resume_best_val_loss:.4f}")
else:
    raise ValueError(f"unsupported init_from={init_from}")

model.to(device)
if compile and device_type == "cuda" and not triton_importable():
    print(
        "warning: torch.compile is on but Triton is not importable; Inductor needs it for CUDA "
        "(common on Windows). Disabling compile; install triton or set compile=False in config."
    )
    compile = False
def make_adamw_optimizer(params, lr):
    adamw_kwargs = dict(lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    if "fused" in inspect.signature(torch.optim.AdamW).parameters:
        adamw_kwargs["fused"] = (device_type == "cuda")
    return torch.optim.AdamW(params, **adamw_kwargs)


named_params = list(model.named_parameters())

if optimizer_type == "adamw":
    optimizer = make_adamw_optimizer([p for _, p in named_params], learning_rate)
    if resume_optimizer_state is not None:
        if isinstance(resume_optimizer_state, dict) and "adamw" in resume_optimizer_state:
            optimizer.load_state_dict(resume_optimizer_state["adamw"])
        else:
            optimizer.load_state_dict(resume_optimizer_state)
elif optimizer_type == "muon_adamw":
    muon_params = []
    adamw_params = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if param.ndim == 2 and "transformer" in name and ".h." in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    optimizer = {
        "muon": Muon(muon_params, lr=muon_lr, momentum=muon_momentum, ns_steps=muon_ns_steps),
        "adamw": make_adamw_optimizer(adamw_params, learning_rate),
    }
    if resume_optimizer_state is not None:
        if isinstance(resume_optimizer_state, dict) and "muon" in resume_optimizer_state:
            optimizer["muon"].load_state_dict(resume_optimizer_state["muon"])
            optimizer["adamw"].load_state_dict(resume_optimizer_state["adamw"])
        else:
            print("warning: resume optimizer state is not muon_adamw; restarting optimizer state")
else:
    raise ValueError(f"unsupported optimizer_type={optimizer_type}")

print(
    "train setup: "
    f"optimizer_type={optimizer_type}, "
    f"lr_schedule={lr_schedule}, "
    f"device={device}, "
    f"dtype={dtype}"
)


def optimizer_zero_grad(opt):
    if isinstance(opt, dict):
        for sub_opt in opt.values():
            sub_opt.zero_grad(set_to_none=True)
    else:
        opt.zero_grad(set_to_none=True)


def optimizer_step(opt):
    if isinstance(opt, dict):
        for sub_opt in opt.values():
            sub_opt.step()
    else:
        opt.step()


def set_optimizer_lr(opt, lr):
    if isinstance(opt, dict):
        for param_group in opt["adamw"].param_groups:
            param_group["lr"] = lr
        muon_scale = muon_lr / max(learning_rate, 1e-12)
        muon_effective_lr = lr * muon_scale
        for param_group in opt["muon"].param_groups:
            param_group["lr"] = muon_effective_lr
    else:
        for param_group in opt.param_groups:
            param_group["lr"] = lr


def optimizer_state_dict(opt):
    if isinstance(opt, dict):
        return {"type": optimizer_type, "muon": opt["muon"].state_dict(), "adamw": opt["adamw"].state_dict()}
    return opt.state_dict()


if compile:
    model = torch.compile(model)


def get_seq_len(it):
    current = block_size
    for start_step, size in seq_len_schedule:
        if it >= start_step:
            current = min(size, block_size)
    return current


def get_lr(it):
    if lr_schedule == "cosine":
        if it < warmup_iters:
            return learning_rate * (it + 1) / max(1, warmup_iters)
        if it >= lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
    if lr_schedule == "wsd":
        warmup = max(0, warmup_iters)
        cooldown_iters = max(1, int(max_iters * wsd_cooldown_frac))
        stable_end = max(warmup, max_iters - cooldown_iters)
        if it < warmup:
            return learning_rate * (it + 1) / max(1, warmup)
        if it < stable_end:
            return learning_rate
        if it >= max_iters:
            return wsd_final_lr
        decay_ratio = (it - stable_end) / max(1, max_iters - stable_end)
        return learning_rate + decay_ratio * (wsd_final_lr - learning_rate)
    raise ValueError(f"unsupported lr_schedule={lr_schedule}")


@torch.no_grad()
def estimate_val_loss(it):
    model.eval()
    losses = torch.zeros(eval_iters)
    seq_len = get_seq_len(it)
    for k in range(eval_iters):
        x, y = val_loader.get_batch(seq_len=seq_len, device=device)
        with ctx:
            _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


best_val_loss = resume_best_val_loss
evals_without_improvement = 0
t0 = time.time()
for iter_num in range(resume_iter_num, max_iters):
    lr = get_lr(iter_num)
    set_optimizer_lr(optimizer, lr)

    seq_len = get_seq_len(iter_num)
    optimizer_zero_grad(optimizer)
    train_loss_accum = 0.0
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = train_loader.get_batch(seq_len=seq_len, device=device)
        with ctx:
            _, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps
        train_loss_accum += loss.item()
        loss.backward()

    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer_step(optimizer)

    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        val_loss = estimate_val_loss(iter_num)
        print(
            f"iter {iter_num:5d} | train_loss {train_loss_accum:.4f} | val_loss {val_loss:.4f} "
            f"| lr {lr:.3e} | seq_len {seq_len}"
        )
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer_state_dict(optimizer),
            "config": model_args,
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, os.path.join(out_dir, "ckpt_last.pt"))
        if val_loss < (best_val_loss - early_stop_min_delta):
            best_val_loss = val_loss
            evals_without_improvement = 0
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, os.path.join(out_dir, "ckpt_best.pt"))
            torch.save(checkpoint, os.path.join(out_dir, "checkpoint.pt"))
            print(f"new best checkpoint saved at iter {iter_num}: val_loss={val_loss:.4f}")
        else:
            evals_without_improvement += 1
            if always_save_checkpoint:
                torch.save(checkpoint, os.path.join(out_dir, f"ckpt_{iter_num:06d}.pt"))
            if early_stop_patience > 0 and evals_without_improvement >= early_stop_patience:
                print(
                    f"early stopping at iter {iter_num}: no val improvement for "
                    f"{evals_without_improvement} evals (best={best_val_loss:.4f}, current={val_loss:.4f})"
                )
                break
    elif iter_num % log_interval == 0:
        dt = time.time() - t0
        tokens_per_step = batch_size * seq_len * gradient_accumulation_steps
        print(
            f"iter {iter_num:5d} | train_loss {train_loss_accum:.4f} | lr {lr:.3e} "
            f"| seq_len {seq_len} | dt {dt*1000:.1f}ms | toks/step {tokens_per_step}"
        )
    t0 = time.time()
