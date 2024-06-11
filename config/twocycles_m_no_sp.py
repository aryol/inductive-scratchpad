out_dir = 'out-twocycles'
eval_interval = 200 
eval_iters = 4
log_interval = 100 # don't print too too often

always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'twocycles'
wandb_run_name = 'two-cycles-m-no-sp'


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

n_layer=12
n_head=12
n_embd=768
dropout = 0.2

learning_rate = 3e-4 
max_iters = 100000
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr = 3e-5
beta2 = 0.99 
warmup_iters = 0
