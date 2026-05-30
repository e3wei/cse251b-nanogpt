# Rank-seeking final candidate
# 14x640 + WSD + bf16, designed for 24GB Blackwell MIG

out_dir = "out_final_14x640_wsd_4B_bf16"

# data
dataset_dir = "data/fineweb_4B"
train_bin = "train.bin"
val_bin = "val.bin"

# batching
# Try to keep effective batch = 128 sequences, same as milestone setup.
batch_size = 8
gradient_accumulation_steps = 16
block_size = 1024

# model: expanded A1a, still should be <100M
n_layer = 14
n_embd = 640
n_head = 10
bias = False
norm_type = "rmsnorm"
norm_eps = 1e-5
rope_base = 10000
activation = "swiglu"
ffn_mult = 2.5
ffn_dim_multiple_of = 64
qk_norm = True

# optimizer
optimizer_type = "adamw"
learning_rate = 3e-4
min_lr = 3e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# WSD schedule
lr_schedule = "wsd"
warmup_iters = 400
wsd_cooldown_frac = 0.2
wsd_final_lr = 0.0
max_iters = 20000
lr_decay_iters = 20000

# curriculum
# Original schedule, but with a longer 1024 phase because max_iters is 20k.
seq_len_schedule = [
    (0, 256),
    (1500, 512),
    (4000, 768),
    (7000, 1024),
]

# logging / eval
eval_interval = 200
eval_iters = 50
log_interval = 20
always_save_checkpoint = False
early_stop_patience = 0
early_stop_min_delta = 0.0

# runtime
device = "cuda"
dtype = "bfloat16"
compile = False
seed = 1337
