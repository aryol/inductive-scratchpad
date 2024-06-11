out_dir = 'out-twocycles-cur'
eval_interval = 1000
eval_iters = 1
log_interval = 500 # don't print too too often


always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'twocycles'
wandb_run_name = 'two-cycles-cur'


dataset = 'twocycles'
gradient_accumulation_steps = 1
batch_size = 512
test_batch_size = 512
block_size = 72 
scratchpad_type = 'none'
cycle_size = 7
alphabet_size = 1000
threshold_acc = 0.95 

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 3e-4 
max_iters = 100000
lr_decay_iters = 150000 
min_lr = 3e-5 # learning_rate / 10 usually
beta2 = 0.99 
warmup_iters = 0
