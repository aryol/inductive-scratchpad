import os
import pickle
import torch
import numpy as np
from math import ceil
from tqdm import tqdm


alphabet = [f"{i:01}" for i in range(10)]

def spacify(text):
    # put a space between each two chars of the text
    return " ".join(list(text))

# A function to get the inductive form of the training data, i.e., the inductive input (with copies), positional indices, attention mask, loss mask, and lengths of the new inputs. 
def get_inductive_form_train(X, pad_token_id, state_token_id, start_token_id, terminal_tokens, block_length, compute_loss_before_start=True, return_list=False):
    def _att_mask_from_vector(vecs):
        # the input must be a list of vectors
        b, n = vecs.shape
        x = vecs.reshape((b, n, 1))
        y = vecs.reshape((b, 1, n))
        return (x * (x-y) * y == 0).astype(np.int16)

    def _process_row(row):
        new_row = np.zeros(block_length, dtype=np.int16) + pad_token_id
        positional_indices = np.zeros(block_length, dtype=np.int16)
        att_mask = np.zeros(block_length, dtype=np.int16)
        loss_mask = np.zeros(block_length, dtype=np.float16)
        # finding start token
        start_token_idx = np.where(row == start_token_id)[0]
        if len(start_token_idx) == 0:
            start_token_idx = -1
        else:
            start_token_idx = start_token_idx[-1]
        # getting apearance of state token
        state_token_appearances = np.where(row == state_token_id)[0]
        last_state_token_idx = state_token_appearances[-1] if len(state_token_appearances) > 0 else start_token_idx
        # doing the constructions
        copying_index = 0
        pos_emb_offset = 0
        if start_token_idx >= 0:
            new_row[:start_token_idx+1] = row[:start_token_idx+1]
            positional_indices[:start_token_idx+1] = np.arange(start_token_idx+1)
            att_mask[:start_token_idx+1] = 0
            loss_mask[:start_token_idx+1] = 1 if compute_loss_before_start else 0
        for i in range(len(state_token_appearances)):
            state_idx = state_token_appearances[i]
            prev_state_idx = state_token_appearances[i-1] if i > 0 else start_token_idx
            new_row[copying_index + prev_state_idx + 1: copying_index + state_idx + 1] = row[prev_state_idx + 1: state_idx + 1]
            positional_indices[copying_index + prev_state_idx + 1: copying_index + state_idx + 1] = np.arange(pos_emb_offset + start_token_idx + 1, pos_emb_offset + start_token_idx + state_idx - prev_state_idx + 1)
            att_mask[copying_index + prev_state_idx + 1: copying_index + state_idx + 1] = i + 1
            loss_mask[copying_index + prev_state_idx + 1: copying_index + state_idx + 1] = 1
            if i == 0 and start_token_idx == -1:
                pos_emb_offset = state_idx - prev_state_idx
            elif (i < len(state_token_appearances) - 1) or ((len(row) > state_idx + 1) and (row[state_idx + 1] not in terminal_tokens)): # state being repeated if it's not the last one or if there are more non-trivial tokens after the last state. We also don't copy the first state if there is not start token. 
                # or (len(row) > state_idx and (row[state_idx] + 1 not in terminal_tokens)) 
                # print(start_token_idx, i, (not (i == 0 and start_token_idx == -1)) and (i < len(state_token_appearances) - 1))
                copying_index += state_idx - prev_state_idx
                new_row[copying_index + prev_state_idx + 1: copying_index + state_idx + 1] = row[prev_state_idx + 1: state_idx + 1]
                positional_indices[copying_index + prev_state_idx + 1: copying_index + state_idx + 1] = np.arange(start_token_idx + 1, start_token_idx + state_idx - prev_state_idx + 1)
                att_mask[copying_index + prev_state_idx + 1: copying_index + state_idx + 1] = i + 2
                loss_mask[copying_index + prev_state_idx + 1: copying_index + state_idx + 1] = 0
                pos_emb_offset = state_idx - prev_state_idx
        
        new_length = copying_index + len(row)
        new_row[copying_index + last_state_token_idx + 1: new_length] = row[last_state_token_idx + 1:]
        positional_indices[copying_index + last_state_token_idx + 1: new_length] = np.arange(last_state_token_idx + 1, len(row)) + (pos_emb_offset + start_token_idx - last_state_token_idx)
        att_mask[copying_index + last_state_token_idx + 1: new_length] = len(state_token_appearances) + 1
        loss_mask[copying_index + last_state_token_idx + 1: new_length] = 1
        if not return_list:
            return new_row, positional_indices, att_mask, loss_mask, new_length
        else:
            return new_row[:new_length], positional_indices[:new_length], att_mask[:new_length], loss_mask[:new_length], new_length
        
    if not return_list:
        X = np.array(X, dtype=np.int16)
        X_new = np.zeros((X.shape[0], block_length), dtype=np.int16)
        positional_indices = np.zeros((X.shape[0], block_length), dtype=np.int16)
        att_mask = np.zeros((X.shape[0], block_length), dtype=np.int16)
        loss_mask = np.zeros((X.shape[0], block_length), dtype=np.int16)
        new_lengths = np.zeros(X.shape[0], dtype=np.int16)
        for i in range(X.shape[0]):
            X_new[i, :], positional_indices[i, :], att_mask[i, :], loss_mask[i, :], new_lengths[i] = _process_row(X[i])
        att_mask = _att_mask_from_vector(att_mask)
        return X_new, positional_indices, att_mask, loss_mask, new_lengths
    else:
        X_new = []
        positional_indices = []
        att_mask = []
        loss_mask = []
        new_lengths = []
        for i in range(len(X)):
            X_new_row, positional_indices_row, att_mask_row, loss_mask_row, new_length = _process_row(X[i])
            X_new.append(X_new_row)
            positional_indices.append(positional_indices_row)
            att_mask.append(_att_mask_from_vector(att_mask_row))
            loss_mask.append(loss_mask_row)
            new_lengths.append(new_length)
        att_mask = _att_mask_from_vector(att_mask)
        return X_new, positional_indices, att_mask, loss_mask, new_lengths


# Modulo (and parity) dataset with fixed samples. 
class ModuloDataset(torch.utils.data.Dataset):
    def __init__(self, M, dim, degree, N, embedding_dim = None, scratchpad="full", alphabet=alphabet, sep_char=";", state_char='#', start_char=':', eos='.', tokenizer=None, block_length=128, uniform=False):
        super(ModuloDataset, self).__init__()
        self.M = M
        self.N = N
        self.dim = dim
        self.degree = degree
        self.scratchpad = scratchpad
        self.alphabet = alphabet
        self.sep_char = sep_char
        self.state_char = state_char
        self.start_char = start_char
        self.special_char = False
        self.eos = eos
        self.tokenizer = tokenizer
        self.inductive_form = ("induct" in scratchpad)
        self.block_length = block_length
        self.data = []
        self.embedding_dim = embedding_dim
        self.uniform = uniform
        if self.embedding_dim is None:
            self.embedding_dim = self.dim
        self.samples_per_call = 1024 if N <= 1024 * 1024 else 512 * 100
        if not uniform:
            for i in tqdm(range(ceil(N/self.samples_per_call))):
                X, y = self.create_sample(self.samples_per_call, self.dim, self.degree, self.scratchpad)
                self.data.extend(list(zip(X, y)))
                # NOTE: this code generates ceil(N/self.samples_per_call) * self.samples_per_call samples.
        else:
            for _ in tqdm(range(ceil(N/256/dim))):
                for i in range(1, dim + 1):
                    X, y = self.create_sample(256, i, i, self.scratchpad)
                    self.data.extend(list(zip(X, y)))
                # NOTE: this code generates ceil(N/256/dim) * 256 * dim samples.



    def create_sample(self, N, dim, degree, scratchpad):
        X = np.random.randint(0, self.M, size=(N, dim))
        jump = 1
        if scratchpad.startswith("full"):
            if scratchpad == "full":
                scratchpad = "full1"
            jump = int(scratchpad[4:])
        assert degree % jump == 0, "degree must be divisible by jump of the scratchpad."
        if scratchpad.startswith("full") or scratchpad.startswith("induct"):
            path = X.cumsum(axis=1)[:, :degree] % self.M
            y = path[:, jump-1::jump]
        else:
            y = X[:, :degree].sum(axis=1, keepdims=True) % self.M
        X = np.apply_along_axis(lambda row: "".join(map(lambda i: alphabet[i], row)), axis=1, arr=X).tolist()
        y = np.apply_along_axis(lambda row: "".join(map(lambda i: alphabet[i], row)), axis=1, arr=y).tolist()
        
        if scratchpad == "induct-random-space":
            for i in range(len(X)):
                text = ['_'] * self.embedding_dim
                rand_idx = np.random.choice(range(self.embedding_dim), size=dim, replace=False)
                rand_idx.sort()
                for j, r, in enumerate(rand_idx):
                    text[r] = X[i][j]
                if self.embedding_dim <= 99:
                    y[i] = self.state_char.join([f"[{rand_idx[j]:02}]{X[i][j]},{y[i][j]}" for j in range(degree)])
                    y[i] += self.state_char + f"[{self.embedding_dim:02}]_,{y[i][-1]}"
                elif self.embedding_dim <= 999:
                    y[i] = self.state_char.join([f"[{rand_idx[j]:03}]{X[i][j]},{y[i][j]}" for j in range(degree)])
                    y[i] += self.state_char + f"[{self.embedding_dim:03}]_,{y[i][-1]}"
                else:
                    raise ValueError("Embedding dimension is too large.")
                start_text = self.start_char
                # if self.embedding_dim <= 100:
                #     start_text = f"[{rand_idx:02}],[{rand_idx + degree - 1:02}]" + start_text
                # elif self.embedding_dim <= 1000:
                #     start_text = f"[{rand_idx:03}],[{rand_idx + degree - 1:03}]" + start_text
                X[i] = spacify("".join(text) + "=")
                y[i] = X[i] + " " + spacify(start_text + y[i]) + f" {self.eos}"
        else:
            for i in range(len(X)):
                X[i] = spacify(X[i] + "=")
                y[i] = X[i] + " " + spacify(y[i]) + f" {self.eos}"
        return X, y

    def __getitem__(self, idx):
        if self.tokenizer is not None:
            question, ans = self.data[idx]
            if not self.inductive_form:
                text = [self.data[idx][0], self.data[idx][1], -1, -1, -1, -1, -1]
            else:
                text = [None,np.array(self.tokenizer.encode(ans), dtype=np.int16), None, None, None, None, None]
                text[2], text[3], text[4], text[5], text[6] = self.tokenizer.to_inductive_form([text[1]], self.block_length)
                text[0], text[1], text[2], text[3], text[4], text[5], text[6] = question, ans, text[2][0], text[3][0], text[4][0], text[5][0], text[6][0]
            return text
        else:
            return self.data[idx]
        
    def __len__(self):
        return len(self.data)


# Modulo (and parity) dataset with online samples.
class ModuloIterDataset(torch.utils.data.IterableDataset):
    def __init__(self, M, dim, degree, embedding_dim = None, scratchpad="full", alphabet=alphabet, sep_char=";", state_char='#', start_char=':', eos='.', tokenizer=None, block_length=128):
        super(ModuloIterDataset, self).__init__()
        self.M = M
        self.dim = dim
        self.degree = degree
        self.scratchpad = scratchpad
        self.alphabet = alphabet
        self.sep_char = sep_char
        self.state_char = state_char
        self.start_char = start_char
        self.special_char = False
        self.eos = eos
        self.tokenizer = tokenizer
        self.inductive_form = ("induct" in scratchpad)
        self.block_length = block_length
        self.embedding_dim = embedding_dim
        if self.embedding_dim is None:
            self.embedding_dim = self.dim

    def create_sample(self, N, dim, degree, scratchpad):
        X = np.random.randint(0, self.M, size=(N, dim))
        jump = 1
        if scratchpad.startswith("full"):
            if scratchpad == "full":
                scratchpad = "full1"
            jump = int(scratchpad[4:])
        assert degree % jump == 0, "degree must be divisible by jump of the scratchpad."
        if scratchpad.startswith("full") or scratchpad.startswith("induct"):
            path = X.cumsum(axis=1)[:, :degree] % self.M
            y = path[:, jump-1::jump]
        else:
            y = X[:, :degree].sum(axis=1, keepdims=True) % self.M
        X = np.apply_along_axis(lambda row: "".join(map(lambda i: alphabet[i], row)), axis=1, arr=X).tolist()
        y = np.apply_along_axis(lambda row: "".join(map(lambda i: alphabet[i], row)), axis=1, arr=y).tolist()
        
        if scratchpad == "induct-random-space":
            for i in range(len(X)):
                text = ['_'] * self.embedding_dim
                rand_idx = np.random.choice(range(self.embedding_dim), size=dim, replace=False)
                rand_idx.sort()
                for j, r, in enumerate(rand_idx):
                    text[r] = X[i][j]
                if self.embedding_dim <= 99:
                    y[i] = self.state_char.join([f"[{rand_idx[j]:02}]{X[i][j]},{y[i][j]}" for j in range(degree)])
                    y[i] += self.state_char + f"[{self.embedding_dim:02}]_,{y[i][-1]}"
                elif self.embedding_dim <= 999:
                    y[i] = self.state_char.join([f"[{rand_idx[j]:03}]{X[i][j]},{y[i][j]}" for j in range(degree)])
                    y[i] += self.state_char + f"[{self.embedding_dim:03}]_,{y[i][-1]}"
                else:
                    raise ValueError("Embedding dimension is too large.")
                start_text = self.start_char
                # if self.embedding_dim <= 100:
                #     start_text = f"[{rand_idx:02}],[{rand_idx + degree - 1:02}]" + start_text
                # elif self.embedding_dim <= 1000:
                #     start_text = f"[{rand_idx:03}],[{rand_idx + degree - 1:03}]" + start_text
                X[i] = spacify("".join(text) + "=")
                y[i] = X[i] + " " + spacify(start_text + y[i]) + f" {self.eos}"
        else:
            for i in range(len(X)):
                X[i] = spacify(X[i] + "=")
                y[i] = X[i] + " " + spacify(y[i]) + f" {self.eos}"
        return X, y

    def __iter__(self):
        while True:
            X, y = self.create_sample(512, self.dim, self.degree, self.scratchpad)
            texts = [None for _ in range(len(X))]
            if self.tokenizer is not None:
                for i in range(len(X)):
                    texts[i] = [X[i], y[i], -1, -1, -1, -1, -1]
                    if self.inductive_form:
                        texts[i][2], texts[i][3], texts[i][4], texts[i][5], texts[i][6] = self.tokenizer.to_inductive_form([texts[i][1]], self.block_length)
                        texts[i][2], texts[i][3], texts[i][4], texts[i][5], texts[i][6] = texts[i][2][0], texts[i][3][0], texts[i][4][0], texts[i][5][0], texts[i][6][0]
            else:
                for i in range(len(X)):
                    texts[i] = [X[i], y[i]]
            for text in texts:
                yield text


# Tokenizer
class ModuloTokenizer():
    def __init__(self, sep_char=';', start_char = ':', state_char='#', pad_token=' ', eos_token='.', alphabet=alphabet):
        self.sep_char = sep_char
        self.state_char = state_char
        self.pad_token = pad_token
        self.start_char = start_char
        self.eos_token = eos_token
        self.alphabet = alphabet
        chars = list(set([sep_char, start_char, state_char, pad_token, eos_token, '?', '|', '=', '[', ']', ',', '_', '$'])) + alphabet
        self.vocab_size = len(chars)
        print("Vocab size:", self.vocab_size)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def _to_str(self, x):
        # convert a tensor to a list
        return "".join(map(str, x.tolist()))

    
    def to_inductive_form(self, X, block_length):
        return get_inductive_form_train(X, self.stoi[self.pad_token], self.stoi[self.state_char], self.stoi[self.start_char], [self.stoi[self.eos_token]], block_length, return_list=False)

    
    def encode(self, s):
        # return [self.stoi[c] for c in s]
        return list(map(lambda x: self.stoi[x], s.split(" ")))
    
    def encode_batch(self, X):
        X = [self.encode(x) for x in X]
        max_len = max([len(x) for x in X])
        X = [x + [self.stoi[self.pad_token]] * (max_len - len(x)) for x in X]
        return X
        
    def decode(self, l):
        return ''.join([self.itos.get(i, "_") for i in l])
    
    def decode_batch(self, X):
        return [self.decode(x) for x in X]

# Running this part will create a small sample dataset and will also create the meta.pkl file.
if __name__ == '__main__':
    tokenizer = ModuloTokenizer()
    M = 2
    N = 20
    degree = 10
    dim = 20
    emb_dim=20
    uniform = False
    dataset = ModuloDataset(M, dim, degree, N, scratchpad='none', tokenizer=tokenizer, block_length=480, embedding_dim=emb_dim, uniform=uniform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, num_workers=1, shuffle=True)
    for X, y, y_ind, _, _, _, lengths in data_loader:
        print("Number of tokens:", len(X))
        print("Max size of the batch:", max(lengths).item())
        # y_ind = tokenizer.decode_batch(y_ind.tolist())
        for i in range(len(X)):
            print(X[i])
            print(y[i])
            print(y_ind[i])
        break
    meta = {
        'vocab_size': tokenizer.vocab_size,
        'itos': tokenizer.itos,
        'stoi': tokenizer.stoi,
        'tokens': y_ind.shape[-1]
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)