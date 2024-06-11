out_dir = 'out-modulo-sp'
eval_interval = 500 
eval_iters = 1
log_interval = 100 

always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'modulo'
wandb_run_name = 'modulo-sp'

dataset = 'modulo'
scratchpad_type = 'full'
gradient_accumulation_steps = 1
batch_size = 1024
test_batch_size = 256
block_size = 64

# task params
samples_train = 0 # using fresh samples
samples_test = 1024
mod = 2
degree = 12
dim = 24
embedding_dim = 24
threshold_acc = 1.0
inductive = False
uniform_generation=False
modulo_ood = False
scratchpad_type = 'full'

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
weight_decay = 1e-2