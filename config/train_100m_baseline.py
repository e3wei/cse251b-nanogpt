# ~92M parameter baseline
out_dir = "out_100m_baseline"

# data and batching
dataset_dir = "."
train_bin = "train.bin"
val_bin = "val.bin"
batch_size = 8
gradient_accumulation_steps = 16
block_size = 1024

# model
n_layer = 22
n_embd = 512
n_head = n_embd // 64
bias = False
norm_type = "rmsnorm"
norm_eps = 1e-5
rope_base = 10000
activation = "swiglu"      # options: swiglu, relu2
ffn_mult = 2.5
ffn_dim_multiple_of = 64
qk_norm = True

# optimizer / schedule
learning_rate = 3e-4
min_lr = 3e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
warmup_iters = 200
lr_decay_iters = 12000
max_iters = 12000
grad_clip = 1.0

# sequence length scheduling
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
early_stop_patience = 15
early_stop_min_delta = 0.0

# runtime
device = "cuda"
dtype = "bfloat16"
compile = False
seed = 1337
