out_dir = 'out-twocycles'
eval_interval = 500
eval_iters = 4
log_interval = 100 # don't print too too often


always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'twocycles'
wandb_run_name = 'two-cycles-xs-no-sp'


dataset = 'twocycles'
gradient_accumulation_steps = 1
batch_size = 512
test_batch_size = 256
block_size = 128
small_size = 3 
scratchpad_type = 'none'
mode = 'single_size'
cycle_size = 5
alphabet_size = 1000
threshold_acc = 1.0

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 3e-4 
max_iters = 120000
lr_decay_iters = 120000 # make equal to max_iters usually
min_lr = 3e-5 # learning_rate / 10 usually
beta2 = 0.99 
warmup_iters = 0
