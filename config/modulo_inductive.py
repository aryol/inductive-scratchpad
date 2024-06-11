out_dir = 'out-modulo-length'
eval_interval = 500
eval_iters = 2
log_interval = 50

always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'modulo'
wandb_run_name = 'modulo-length-gen'

dataset = 'modulo'
gradient_accumulation_steps = 1
batch_size = 512
test_batch_size = 512
block_size = 720

# task params
samples_train = 512 * 3000
samples_test = 512 * 2
mod = 2
degree = 30
dim = 30
embedding_dim = 60
inductive = True
uniform_generation=True
modulo_ood = True
scratchpad_type = 'induct-random-space'

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 3e-4 
max_iters = 3000
lr_decay_iters = 3000 # make equal to max_iters usually
min_lr = 3e-5 # learning_rate / 10 usually
beta2 = 0.99 
warmup_iters = 0 
weight_decay = 1e-3