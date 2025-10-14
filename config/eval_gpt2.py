# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters
out_dir = 'out/bytelm3070'
batch_size = 32                 # lower if OOM (e.g., 32/24/16)
block_size = 128   
eval_iters = 500 # use more iterations to get good estimate
eval_only = True
wandb_log = True
init_from = 'resume'