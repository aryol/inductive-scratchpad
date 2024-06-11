out_dir = 'out-addition-shift'
eval_interval = 250 
eval_iters = 4
log_interval = 20 # don't print too too often

always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'addition'
wandb_run_name = 'shift'


min_step_eval = 2000
dataset = 'addition'
N_train = 4
gradient_accumulation_steps = 2
batch_size = 256
test_batch_size = 256
block_size = 960
# block_size = 1152
max_token_generation = 3072
scratchpad_type = 'shift'

top_k = 100
temperature = 0.25

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 3e-4 
max_iters = 2000
lr_decay_iters = 3000 
min_lr = 3e-5 # learning_rate / 10 usually
beta2 = 0.99
weight_decay = 1e-3
warmup_iters = 0