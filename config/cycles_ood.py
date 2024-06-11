out_dir = 'out-twocycles-ood'
eval_interval = 500
eval_iters = 4
log_interval = 100 # don't print too too often

always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'twocycles'
wandb_run_name = 'two-cycles-ood'


dataset = 'twocycles'
gradient_accumulation_steps = 1
batch_size = 512
test_batch_size = 256
block_size = 192 
scratchpad_type = 'none'
mode = 'ood1'
alphabet_size = 1000
threshold_acc = 1.0

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 3e-4 
max_iters = 20000
lr_decay_iters = 20000 # make equal to max_iters usually
min_lr = 3e-5 # learning_rate / 10 usually
beta2 = 0.99 
warmup_iters = 0
