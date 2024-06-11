out_dir = 'out-ER'
eval_interval = 500 
eval_iters = 1
log_interval = 100 # don't print too too often

always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'twocycles'
wandb_run_name = 'ER'


dataset = 'twocycles'
gradient_accumulation_steps = 2 # See below. 
batch_size = 2048 # Have to use a large batch size to get stable and good convergence even on the in-distribution samples. 
test_batch_size = 512
block_size = 128 
scratchpad_type = 'none'
mode = 'ER'
alphabet_size = 1000
threshold_acc = 1.0

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 3e-4 
max_iters = 8000
lr_decay_iters = 8000 # make equal to max_iters usually
min_lr = 3e-5 # learning_rate / 10 usually
beta2 = 0.99
warmup_iters = 0
weight_decay = 0.1
