# This code is mainly borrowed from Andrei Karpathy's GPT-2 implementation named nanoGPT.

"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from data.addition.prepare import AdditionTokenizer, AdditionDataset
from data.twocycles.prepare import TwoCyclesTokenizer, TwoCyclesDataset
from data.modulo.prepare import ModuloTokenizer, ModuloDataset, ModuloIterDataset
from model import GPTConfig, GPT
import random


# Graph task hyperparameters
dim=10
degree=5
mode = 'single_size'
scratchpad_type = 'induction'
seed = 1337
cycle_size = 5
alphabet_size = 1000
meta_info = False

# Addition task hyperparameters
N_train = 10 # 10 for random space and 4 for shift. 
N_test = 30
N_emb = 30

# Modulo task hyperparameters
samples_train = 51200 * 2
samples_test = 512
mod = 10
degree = 20
dim = 20
embedding_dim = 30
inductive = True
uniform_generation = False
modulo_ood = False

# Dataset dependent hyperparameters
out_dir = 'out-addition'
dataset = 'twocycles'
curriculum = False
first_valid = True

# Logging
min_step_eval = 0
name = ''
logs = {'accs': [], 'training': [], 'curriculum': []}
def save_logs():
    with open(os.path.join(out_dir, f'logs_{name}.pkl'), 'wb') as f:
        pickle.dump(logs, f)

# Training and logging settings
temperature = 1.0
top_k = 1
max_token_generation = None
PE = 'default'
num_workers = 1
print_scratchpad_acc = False
compute_valid_loss = False
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
threshold_acc = 1.01
curriculum_threshold_acc = 0.95
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# Batch size and gradient accumulation settings
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 128 # if gradient_accumulation_steps > 1, this is the micro-batch size
test_batch_size = 64
block_size = 32
# Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# Optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# Learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# System
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
inductive = ("induct" in scratchpad_type)
# Generating the data sets
corresponding_vals_for_cur = None # Only used for curriculum learning -- presents a validation set for each training set that is used to infer the time of changing the dataset in the curriculum. 
dataset_lists = {"train": []}
# dataset[valid] is always used for determining when to end the program. 
if dataset == 'twocycles':
    if mode == 'single_size':
        # Main setting checking how the model perform on the cycle task where the train and test distribution are the same. 
        dataset_lists["train"].append([([cycle_size, cycle_size], -1), ([cycle_size * 2], cycle_size)])
        dataset_lists["valid"] = [([cycle_size, cycle_size], -1), ([cycle_size * 2], cycle_size)]
    elif mode == 'ood1':
        # OOD experiment presented in the paper. 
        dataset_lists["train"].append([([6, 18], -1), ([24], 6)])
        dataset_lists["valid"] = [([12, 12], -1), ([24], 12)]
        dataset_lists["valid_iid"] = [([6, 18], -1), ([24], 6)]
    elif mode == 'multiple_sizes':
        # Mixed distribution -- Fixing the number of nodes/edges and changing the cycle sizes. -- not included in the paper.
        dataset_lists["train"].append([([s, 2 * cycle_size - s], -1) for s in range(2, cycle_size + 1)] + [([2 * cycle_size], s) for s in range(2, cycle_size + 1)])
        dataset_lists["valid"] = [([cycle_size, cycle_size], -1), ([cycle_size * 2], cycle_size)]
        dataset_lists["valid_mixed"] = [([s, 2 * cycle_size - s], -1) for s in range(2, cycle_size + 1)] + [([2 * cycle_size], s) for s in range(2, cycle_size + 1)]
    elif mode == 'multiple_nodes':
        # Mixed distribution -- Always the two cycle task with equal cycles but different number of nodes.
        dataset_lists["train"].append([([s, s], -1) for s in range(2, cycle_size + 1)] + [([2 * s], s) for s in range(2, cycle_size + 1)])
        for s in range(2, cycle_size + 1):
            dataset_lists[f"valid{s}"] = [([s, s], -1), ([s * 2], s)]
        dataset_lists["valid"] = [([cycle_size, cycle_size], -1), ([cycle_size * 2], cycle_size)]
    elif mode == 'multiple_nodes_cur':
        # Curriculum learning on the mixed distribution above, i.e., learning the task for $n=1$ and climbing up without forgetting.
        curriculum = True
        # dataset_lists["train"] = [[([s, s], -1), ([2 * s], s)] for s in range(2, cycle_size + 1)]
        dataset_lists["train"] = [[([s, s], -1) for s in range(2, cycle_size + 1)] + [([2 * s], s) for s in range(2, S + 1)] for S in range(2, cycle_size + 1)]
        corresponding_vals_for_cur = [f"valid{s}" for s in range(2, cycle_size + 1)]
        for s in range(2, cycle_size + 1):
            dataset_lists[f"valid{s}"] = [([s, s], -1), ([s * 2], s)]
        dataset_lists["valid"] = [([cycle_size, cycle_size], -1), ([cycle_size * 2], cycle_size)]
    elif mode == 'multiple_nodes_cur_forget':
        # Curriculum learning on the mixed distribution above, i.e., learning the task for $n=1$ and climbing up with forgetting. 
        curriculum = True
        dataset_lists["train"] = [[([s, s], -1), ([2 * s], s)] for s in range(2, cycle_size + 1)]
        corresponding_vals_for_cur = [f"valid{s}" for s in range(2, cycle_size + 1)]
        for s in range(2, cycle_size + 1):
            dataset_lists[f"valid{s}"] = [([s, s], -1), ([s * 2], s)]
        dataset_lists["valid"] = [([cycle_size, cycle_size], -1), ([cycle_size * 2], cycle_size)]
    elif mode == 'ER':
        # Doing the experiments on the Erdos-Renyi random graphs.
        dataset_lists["train"].append([("ER", 24, 24, [1,-1,2,-1,3,-1,4,-1])])
        dataset_lists["valid"] = [("ER", 24, 24, [1,-1,2,-1,3,-1,4,-1])]
        dataset_lists["valid1"] = [("ER", 24, 24, [1])]
        dataset_lists["valid2"] = [("ER", 24, 24, [2])]
        dataset_lists["valid3"] = [("ER", 24, 24, [3])]
        dataset_lists["valid4"] = [("ER", 24, 24, [4])]
        # dataset_lists["valid5"] = [("ER", 24, 24, [5])]
        dataset_lists["valid_NC"] = [("ER", 24, 24, [-1])]
        dataset_lists["valid_OOD_2"] = [([2, 2, 10, 10], -1), ([4, 10, 10], 2)]
        dataset_lists["valid_OOD_3"] = [([3, 3, 6, 6, 6], -1), ([6, 6, 6, 6], 3)]
        dataset_lists["valid_OOD_4"] = [([4, 4, 8, 8], -1), ([8, 8, 8], 4)]
    else:
        raise ValueError("Invalid mode for the two cycles task.")  
elif dataset == 'addition':
    curriculum = False
    dist_train = np.arange(N_train, 0, -1) ** 1
    dist_train = dist_train / dist_train.sum()
    dist_test = np.arange(N_test - 1, 0, -1)
    dist_test = dist_test / dist_test.sum()

    dataset_lists["train"] = []
    dataset_lists["train"].append((dist_train, N_emb))
    dataset_lists["valid iid"] = (dist_train, N_emb)
    dataset_lists["valid"] = (np.eye(N_test)[0], N_emb)
    dataset_lists[f"valid{N_test - 2}"] = (np.eye(N_test)[2], N_test)
    dataset_lists[f"valid{N_test - 4}"] = (np.eye(N_test)[4], N_test)
    dataset_lists[f"valid{N_test - 6}"] = (np.eye(N_test)[6], N_test)
    dataset_lists[f"valid{N_test - 8}"] = (np.eye(N_test)[8], N_test)
    dataset_lists[f"valid{N_test - 10}"] = (np.eye(N_test)[10], N_test)
    dataset_lists[f"valid{N_test - 12}"] = (np.eye(N_test)[12], N_test)
    dataset_lists[f"valid{N_test - 14}"] = (np.eye(N_test)[14], N_test)
    dataset_lists[f"valid{N_test - 16}"] = (np.eye(N_test)[16], N_test)
    dataset_lists[f"valid{N_test - 18}"] = (np.eye(N_test)[18], N_test)
    samples_train = 512 * 1000
    samples_test = 1024
    inductive = True
    tokenizer = AdditionTokenizer(block_length=block_size)
    dataset_torch = {'train_list': [AdditionDataset(samples_train, distribution, max_dist, scratchpad_type=scratchpad_type, tokenizer=tokenizer) for distribution, max_dist in dataset_lists['train']]}
    for k, v in dataset_lists.items():
        if k != 'train':
            dataset_torch[k] = AdditionDataset(samples_test, v[0], v[1], scratchpad_type=scratchpad_type, tokenizer=tokenizer if compute_valid_loss else None)
elif dataset == 'modulo':
    tokenizer = ModuloTokenizer()
    dataset_lists["train"] = []
    dataset_lists["train"].append((mod, dim, degree, embedding_dim))
    dataset_lists["valid"] = (mod, dim, degree, embedding_dim)
    uniform_generation = uniform_generation and modulo_ood
    inductive = inductive and modulo_ood
    if modulo_ood:
        dataset_lists["valid1"] = (mod, dim + 1, degree + 1, embedding_dim)
        dataset_lists["valid5"] = (mod, dim + 5, degree + 5, embedding_dim)
        dataset_lists["valid10"] = (mod, dim + 10, degree + 10, embedding_dim)
        dataset_lists["valid15"] = (mod, dim + 15, degree + 15, embedding_dim)
        dataset_lists["valid20"] = (mod, dim + 20, degree + 20, embedding_dim)
        dataset_lists["valid25"] = (mod, dim + 25, degree + 25, embedding_dim)
        dataset_lists["valid30"] = (mod, dim + 30, degree + 30, embedding_dim)
    if modulo_ood:
        dataset_torch = {'train_list': [ModuloDataset(M, d, k, samples_train, scratchpad=scratchpad_type, tokenizer=tokenizer if inductive else None, block_length=block_size, embedding_dim=emb_dim, uniform=uniform_generation) for M, d, k, emb_dim in dataset_lists['train']]}
    else:
        dataset_torch = {'train_list': [ModuloIterDataset(M, d, k, scratchpad=scratchpad_type, tokenizer=tokenizer if inductive else None, block_length=block_size, embedding_dim=emb_dim) for M, d, k, emb_dim in dataset_lists['train']]}
    for k, v in dataset_lists.items():
        if k != 'train':
            M, d, deg, emb_dim = v
            dataset_torch[k] = ModuloDataset(M, d, deg, samples_test, scratchpad=scratchpad_type, tokenizer=tokenizer if compute_valid_loss and inductive else None, block_length=block_size, embedding_dim=emb_dim, uniform=(k=='valid' and uniform_generation))
print(dataset_lists)
# DDP setup. 
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"Tokens per iteration will be: {tokens_per_iter:,} (assuming that block size is entirely used).")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
np.random.seed(seed + seed_offset)
random.seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loading for parity (modulo) and addition tasks. 
data_dir = os.path.join('data', dataset)
if dataset == 'addition' or dataset == 'modulo':
    curr_index = 0
    data_loader = {}
    data_loader_iterator = {}
    for k, v in dataset_torch.items():
        if k == 'train_list':
            for i in range(len(v)):
                data_loader[f'train{i}'] = torch.utils.data.DataLoader(v[i], batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=not isinstance(v[i], torch.utils.data.IterableDataset))
                data_loader_iterator[f'train{i}'] = iter(data_loader[f'train{i}'])
        data_loader[k] = torch.utils.data.DataLoader(v, batch_size=batch_size if k=='train' else test_batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
        data_loader_iterator[k] = iter(data_loader[k])
    data_loader['train'] = data_loader['train0']
    data_loader_iterator['train'] = iter(data_loader['train'])
    dataset_torch['train'] = dataset_torch['train_list'][0]
    pad_idx = tokenizer.stoi[tokenizer.pad_token]
    def get_batch(split, eval=False):
        meta_data = None
        if dataset_torch[split].tokenizer is not None:
            try:
                z, t, t_ind, pos_idx, att_mask, loss_mask, max_len = next(data_loader_iterator[split])
            except StopIteration:
                data_loader_iterator[split] = iter(data_loader[split])
                z, t, t_ind, pos_idx, att_mask, loss_mask, max_len = next(data_loader_iterator[split])
            t = torch.tensor(tokenizer.encode_batch(t), dtype=torch.int16)
            z = torch.tensor(tokenizer.encode_batch(z), dtype=torch.int16)
        else:
            try:
                z, t = next(data_loader_iterator[split])
            except StopIteration:
                data_loader_iterator[split] = iter(data_loader[split])
                z, t = next(data_loader_iterator[split])
            t = torch.tensor(tokenizer.encode_batch(t), dtype=torch.int16)
            z = torch.tensor(tokenizer.encode_batch(z), dtype=torch.int16)
            if inductive and not eval:
                t_ind, pos_idx, att_mask, loss_mask, max_len = tokenizer.to_inductive_form(t, block_length=block_size)
                t_ind, pos_idx, att_mask, loss_mask = torch.tensor(t_ind, dtype=torch.int16), torch.tensor(pos_idx, dtype=torch.int16), torch.tensor(att_mask, dtype=torch.bool), torch.tensor(loss_mask, dtype=torch.int16)
            else:
                t_ind = t
                pos_idx = None
                att_mask = None
                loss_mask = None

        x = torch.zeros(t.shape[0], block_size, dtype=torch.int64) + pad_idx
        y = torch.zeros(t.shape[0], max(block_size, t.shape[-1]), dtype=torch.int64) + pad_idx
        if inductive and not eval:
            # max_len = int(np.ceil(max_len.max().item() / 64) * 64)
            max_len = block_size
            t_ind = t_ind[:, :max_len]
            pos_idx = pos_idx[:, :max_len]
            att_mask = att_mask[:, :max_len, :max_len]
            loss_mask = loss_mask[:, :max_len]    
            loss_mask[:, :-1] = loss_mask[:, 1:]
        else:
            # max_len = int(np.ceil(t.shape[-1] / 64) * 64)
            max_len = block_size
        if not eval:
            t = t_ind
            x[:, :t.shape[-1]] = t
            y[:, :t.shape[-1] - 1] = t[:, 1:t.shape[-1]]
            x = x[:, :max_len]
            y = y[:, :max_len]
        else:
            x = z
            y[:, :t.shape[-1]] = t[:, 0:t.shape[-1]]   
        if device_type == 'cuda':
            if inductive and not eval:
                x, y, pos_idx, att_mask, loss_mask = x.long().to(device, non_blocking=True), y.long().to(device, non_blocking=True), pos_idx.long().to(device, non_blocking=True), att_mask.bool().to(device, non_blocking=True), loss_mask.long().to(device, non_blocking=True)
            else:
                x, y = x.long().to(device, non_blocking=True), y.long().to(device, non_blocking=True)
        else:
            if inductive and not eval:
                x, y, pos_idx, att_mask, loss_mask = x.long().to(device), y.long().to(device), pos_idx.long().to(device), att_mask.bool().to(device), loss_mask.long().to(device)
            else:
                x, y = x.long().to(device), y.long().to(device)
        return x, y, pos_idx, att_mask, loss_mask, meta_data
elif dataset == 'twocycles':
    curr_index = 0
    inductive = "induct" in scratchpad_type
    tokenizer = TwoCyclesTokenizer()
    dataset_torch = {'train_list': [(TwoCyclesDataset(cycle_sizes=l, scratchpad=scratchpad_type, alphabet_size=alphabet_size)) if l[0][0] != "ER" else (TwoCyclesDataset(scratchpad=scratchpad_type, alphabet_size=alphabet_size, ER = (l[0][1], l[0][2], l[0][3]))) for l in dataset_lists['train']]}
    for k, v in dataset_lists.items():
        if k != 'train':
            if v[0][0] != "ER":
                dataset_torch[k] = TwoCyclesDataset(cycle_sizes=v, scratchpad=scratchpad_type, alphabet_size=alphabet_size)
            else:
                dataset_torch[k] = TwoCyclesDataset(scratchpad=scratchpad_type, alphabet_size=alphabet_size, ER = (v[0][1], v[0][2], v[0][3]), meta=True)
    data_loader = {}
    for k, v in dataset_torch.items():
        if k == 'train_list':
            for i in range(len(v)):
                data_loader[f'train{i}'] = iter(torch.utils.data.DataLoader(v[i], batch_size=batch_size, num_workers=num_workers, pin_memory=True))
        data_loader[k] = iter(torch.utils.data.DataLoader(v, batch_size=batch_size if k=='train' else test_batch_size, num_workers=num_workers, pin_memory=True))
    data_loader['train'] = data_loader['train0']
    dataset_torch['train'] = dataset_torch['train_list'][0]
    pad_idx = tokenizer.stoi[tokenizer.pad_token]
    def get_batch(split, eval=False):
        meta_data = None
        x = torch.zeros(batch_size if split=='train' else test_batch_size, block_size, dtype=torch.int64) + pad_idx
        y = torch.zeros(batch_size if split=='train' else test_batch_size, block_size, dtype=torch.int64) + pad_idx
        if dataset_torch[split].tokenizer is not None:
            if dataset_torch[split].meta:
                z, t_ind, pos_idx, att_mask, loss_mask, max_len, meta_data = next(data_loader[split])
            else:
                z, t_ind, pos_idx, att_mask, loss_mask, max_len = next(data_loader[split])
            t = t_ind
        else:
            if dataset_torch[split].meta:
                z, t, meta_data = next(data_loader[split])
            else:
                z, t = next(data_loader[split])
            t = torch.tensor(tokenizer.encode_batch(t), dtype=torch.int16)
            z = torch.tensor(tokenizer.encode_batch(z), dtype=torch.int16)
            if inductive:
                t_ind, pos_idx, att_mask, loss_mask, max_len = tokenizer.to_inductive_form(t, block_length=block_size)
                t_ind, pos_idx, att_mask, loss_mask = torch.tensor(t_ind, dtype=torch.int16), torch.tensor(pos_idx, dtype=torch.int16), torch.tensor(att_mask, dtype=torch.bool), torch.tensor(loss_mask, dtype=torch.int16)
            else:
                t_ind = t
                pos_idx = None
                att_mask = None
                loss_mask = None
        if inductive:
            # max_len = int(np.ceil(max_len.max().item() / 64) * 64)
            max_len = block_size
            t_ind = t_ind[:, :max_len]
            pos_idx = pos_idx[:, :max_len]
            att_mask = att_mask[:, :max_len, :max_len]
            loss_mask = loss_mask[:, :max_len]    
            loss_mask[:, :-1] = loss_mask[:, 1:]
        else:
            # max_len = int(np.ceil(t.shape[-1] / 64) * 64)
            max_len = block_size
        if not eval:
            t = t_ind
            x[:, :t.shape[-1]] = t
            y[:, :t.shape[-1] - 1] = t[:, 1:t.shape[-1]]
            x = x[:, :max_len]
            y = y[:, :max_len]
        else:
            x = z
            y[:, :t.shape[-1]] = t[:, 0:t.shape[-1]]   
        if device_type == 'cuda':
            if inductive:
                x, y, pos_idx, att_mask, loss_mask = x.long().to(device, non_blocking=True), y.long().to(device, non_blocking=True), pos_idx.long().to(device, non_blocking=True), att_mask.bool().to(device, non_blocking=True), loss_mask.long().to(device, non_blocking=True)
            else:
                x, y = x.long().to(device, non_blocking=True), y.long().to(device, non_blocking=True)
        else:
            if inductive:
                x, y, pos_idx, att_mask, loss_mask = x.long().to(device), y.long().to(device), pos_idx.long().to(device), att_mask.bool().to(device), loss_mask.long().to(device)
            else:
                x, y = x.long().to(device), y.long().to(device)
        return x, y, pos_idx, att_mask, loss_mask, meta_data

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, pad_token_id=tokenizer.stoi[tokenizer.pad_token], PE=PE)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, f'{name}_ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in dataset_lists.keys():
        if compute_valid_loss or split.startswith('train'):
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y, pos_idx, attn_mask, loss_mask, _ = get_batch(split, eval=False)
                with ctx:
                    _, loss = model(X, Y, pos=pos_idx, att_mask=attn_mask, loss_mask=loss_mask)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

# Computing the accuracy of the model on validation sets. 
@torch.no_grad()
def estimate_acc_with_sampling():
    model.eval()
    accs = {}
    scratchpad_accs = {} 
    is_correct = {} # Only used for the random graph experiment.
    meta_data_list = {} # Only used for the random graph experiment.
    for split in dataset_lists.keys():
        if split != "train":
            print("Split: ", split)
            scratchpad_accs[split] = np.zeros(block_size)
            accs[split] = 0
            counter = 0
            for eval_iter_counter in range(eval_iters):
                X, Y_orig, _, _, _, meta_data = get_batch(split, eval=True)
                if Y_orig.shape[1] > scratchpad_accs[split].shape[0]:
                    scratchpad_accs[split] = np.concatenate([scratchpad_accs[split], counter * np.ones(Y_orig.shape[1] - scratchpad_accs[split].shape[0])])
                counter += X.shape[0]
                with ctx:
                    Y = model.generate(X, block_size - X.size(1) if max_token_generation is None else max_token_generation, start_token_id=tokenizer.stoi[tokenizer.start_char], state_token_id=tokenizer.stoi[tokenizer.state_char], eos_token_id=tokenizer.stoi[tokenizer.eos_token], temperature=temperature, top_k=top_k)
                Y = Y.cpu().numpy()
                Y_orig = Y_orig.cpu().numpy()
                Y_temp = np.zeros_like(Y_orig) + tokenizer.stoi[tokenizer.pad_token]
                Y = Y[:, :Y_orig.shape[1]]
                Y_temp[:, :Y.shape[1]] = Y
                Y = Y_temp
                for i in range(Y.shape[0]):
                    eos = Y[i, :] == tokenizer.stoi[tokenizer.eos_token]
                    if np.any(eos):
                        Y[i, eos.argmax() + 1:] = tokenizer.stoi[tokenizer.pad_token]
                if eval_iter_counter == 0:
                    print(f"For input {tokenizer.decode(X[0].tolist())} and \nexpected output {tokenizer.decode(Y_orig[0, X.size(1):].tolist())} \nwe got {tokenizer.decode(Y[0, X.size(1):].tolist())}")
                if dataset != 'addition':
                    i_possibly_wrong = (Y_orig[:, X.size(1):] == Y[:, X.size(1):]).all(axis=1).argmin()
                    if not (Y_orig[i_possibly_wrong, X.size(1):] == Y[i_possibly_wrong, X.size(1):]).all() and eval_iter_counter == 0:
                        print("### A wrong example:")
                        print(f"For input {tokenizer.decode(X[i_possibly_wrong].tolist())} and \nexpected output {tokenizer.decode(Y_orig[i_possibly_wrong, X.size(1):].tolist())} \nwe got {tokenizer.decode(Y[i_possibly_wrong, X.size(1):].tolist())}")
                    scratchpad_accs[split][X.size(1):] += (Y_orig[:, X.size(1):] == Y[:, X.size(1):]).sum(axis=0)
                    if dataset == 'twocycles' and mode == 'ood1':
                        for s in range(Y.shape[0]):
                            output = tokenizer.decode(Y[s, X.size(1):].tolist())
                            expected_output = tokenizer.decode(Y_orig[s, X.size(1):].tolist())
                            index_of_eos = expected_output.find(tokenizer.eos_token)
                            index_of_eos_output = output.find(tokenizer.eos_token)
                            if output[index_of_eos_output - 1:index_of_eos_output + 1] == expected_output[index_of_eos - 1:index_of_eos + 1]:
                                accs[split] += 1
                    else:
                        accs[split] += (Y_orig[:, X.size(1):] == Y[:, X.size(1):]).all(axis=1).sum()
                else: # for addition there's randomness in the scratchpad which we should handle. 
                    wrong_printed = False
                    for s in range(Y.shape[0]):
                        output = tokenizer.decode(Y[s, X.size(1):].tolist())
                        expected_output = tokenizer.decode(Y_orig[s, X.size(1):].tolist())
                        index_of_eos = expected_output.find(tokenizer.eos_token)
                        real_sum = expected_output[index_of_eos - N_emb - 2:index_of_eos].split("$")[0]
                        if output[index_of_eos] == tokenizer.eos_token and output[index_of_eos - N_emb - 2:index_of_eos].split("$")[0] == real_sum:
                            accs[split] += 1
                            scratchpad_accs[split][X.size(1):] += 1 # This will not be precise here. 
                        elif not wrong_printed:
                            print("### A wrong example:")
                            print(f"For input {tokenizer.decode(X[s].tolist())} and \nexpected output {expected_output} \nwe got {output}")
                            print("Real sum", real_sum, "and got instead", output[index_of_eos - N_emb - 2:index_of_eos].split("$")[0] if '$' in output[index_of_eos - N_emb - 2:index_of_eos] else output[index_of_eos - N_emb - 2:index_of_eos])
                            wrong_printed = True
            if dataset == 'twocycles' and mode == 'ER':
                is_correct[split] = (Y_orig[:, X.size(1):] == Y[:, X.size(1):]).all(axis=1)
                meta_data_list[split] = meta_data
                result_file = os.path.join(out_dir, f'ER_metadata_{name}_iter_{iter_num}.pkl')
                results = {
                    'acc': accs,
                    'scratchpad_accs': scratchpad_accs,
                    'is_correct': is_correct,
                    'meta_data': meta_data_list
                }
                with open(result_file, 'wb') as f:
                    pickle.dump(results, f)


            scratchpad_accs[split] /= counter
            scratchpad_accs[split] = scratchpad_accs[split][X.size(1):]
            accs[split] /= counter
    model.train()
    return scratchpad_accs, accs

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, pos_idx, att_mask, loss_mask, _ = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process and iter_num >= min_step_eval:
        losses = estimate_loss()
        scratchpad, accs = estimate_acc_with_sampling()
        if print_scratchpad_acc:
            for split in scratchpad.keys():
                print(f"scratchpad acc for {split}: {scratchpad[split]}")
        print(f"step: {iter_num}, " + ", ".join([f"{k} acc: {accs[k]}" for k in accs.keys()] + [f"{k} loss: {losses[k]:.4f}" for k in losses.keys()]), flush=True)
        logs['accs'].append([iter_num, accs, losses, lr])
        save_logs()
        if curriculum and accs[corresponding_vals_for_cur[curr_index]] >= curriculum_threshold_acc:
            logs['curriculum'].append([iter_num, curr_index, accs[corresponding_vals_for_cur[curr_index]]])
            save_logs()
            curr_index += 1
            if curr_index >= len(dataset_lists['train']) and accs['valid'] >= threshold_acc:
                break
            elif curr_index >= len(dataset_lists['train']):
                pass # We have to continue training on the current dataset as accs['valid'] < threshold_acc and there is no other train dataset.
            else:
                print("Going to the next dataset in the curriculum.")
                data_loader['train'] = data_loader[f'train{curr_index}']
                dataset_torch['train'] = dataset_torch['train_list'][curr_index]
                try:
                    data_loader_iterator['train'] = data_loader_iterator[f'train{curr_index}'] # Going to the next data loader iterator if it is defined. 
                except Exception:
                    pass
        if accs['valid'] >= threshold_acc:
            break
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['valid'],
                "train/acc": accs['train'],
                "val/acc": accs['valid'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['train'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['train'] # NOTE: one may want to change this. 
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_train_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f'{name}_ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y, pos=pos_idx, att_mask=att_mask, loss_mask=loss_mask)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, pos_idx, att_mask, loss_mask, _ = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%", flush=True)
        logs['training'].append([iter_num, lossf, dt, running_mfu])
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
if ddp:
    destroy_process_group()
