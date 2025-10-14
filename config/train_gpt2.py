# Byte-level LM config overrides for NanoGPT on single RTX 3070 (8GB)

# --- logging / I-O ---
out_dir = 'out/bytelm-medium'
eval_interval = 1000
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'resume'

# --- Weights & Biases ---
wandb_log = True
wandb_project = 'byteLM'
wandb_run_name = 'rtx3070-medium'

# --- data ---
dataset = 'train'

# --- batching & context ---
batch_size = 16                 # lower if OOM (e.g., 32/24/16)
gradient_accumulation_steps = 1
block_size = 128                # sentence-level context

# --- model size ---
n_layer = 20
n_head  = 8
n_embd  = 512
dropout = 0.1
bias = False

# --- optimizer & schedule ---
learning_rate = 2e-4
max_iters = 150_000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 2000
lr_decay_iters = 100_000
min_lr = 2e-5

# --- system ---
backend = 'nccl'
device = 'cuda'
dtype = 'float16'
compile = True
