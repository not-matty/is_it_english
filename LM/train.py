import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import wandb

from model import GPTConfig, GPT

# I/O
out_dir = 'out_lm'
eval_interval = 200
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = 'resume'
wandb_log = True
wandb_project = 'lm'
wandb_run_name = 'run' + str(time.time())
early_stop_patience = 4
# data
dataset = 'train_lm'
batch_size = 16
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 3e-4
max_iters = 20000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# lr decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 3e-5
# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 'float16'
compile = True

# derived
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loader
data_dir = os.path.join('data', dataset)
train_bin = os.path.join(data_dir, 'train.bin')
val_bin = os.path.join(data_dir, 'val.bin')
meta_path = os.path.join(data_dir, 'meta.pkl')

meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

def get_batch(split):
    path = train_bin if split == 'train' else val_bin
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init
iter_num = 0
best_val_loss = 1e9

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("warning: meta vocab_size not found, defaulting to 50304")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)
if compile:
    print("compiling the model...")
    model = torch.compile(model)

# GradScaler (use torch.amp.GradScaler with device type)
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

raw_model = model
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

if wandb_log:
    wandb.init(project=wandb_project, name=wandb_run_name, config={
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "block_size": block_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    })

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
no_improve_evals = 0
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": mfu if 'mfu' in locals() else 0.0,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            no_improve_evals = 0
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                os.makedirs(out_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        else:
            no_improve_evals += 1
            if early_stop_patience > 0 and no_improve_evals >= early_stop_patience:
                print(f"Early stopping after {no_improve_evals} evals without val loss improvement.")
                break
    if iter_num == 0 and eval_only:
        break

    optimizer.zero_grad(set_to_none=True)
    with ctx:
        logits, loss = model(X, Y)
    scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

    X, Y = get_batch('train')

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        L, H, Q, T = n_layer, n_head, n_embd // n_head, block_size
        N = model.get_num_params()
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T * batch_size
        flops_achieved = flops_per_fwdbwd * (1.0 / dt)
        flops_peak = 4.19e14
        mfu = flops_achieved / flops_peak
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, mfu {mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break
