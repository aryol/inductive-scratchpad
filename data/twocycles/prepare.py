import os
import pickle
import torch
import random
import numpy as np
import networkx as nx

# alphabet = list(chr(c) for c in range(ord('a'), ord('z')+1))
alphabet = [f"v{i:03}" for i in range(1000)]

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
        
    #     # OLDER CODE
    #     # previous_state_idx = -1
    #     # state_copy = False
    #     # state_counter = 1
    #     # copying_index = 0
    #     # for i in range(row.shape[0]):
    #     #     if i <= start_token_idx:
    #     #         new_row[copying_index] = row[i]
    #     #         positional_indices[copying_index] = i
    #     #         att_mask[copying_index] = 0
    #     #         loss_mask[copying_index] = 1 if compute_loss_before_start else 0
    #     #         copying_index += 1
    #     #     elif previous_state_idx == -1:
    #     #         new_row[copying_index] = row[i]
    #     #         positional_indices[copying_index] = i
    #     #         att_mask[copying_index] = state_counter
    #     #         loss_mask[copying_index] = 1
    #     #         copying_index += 1
    #     #         if row[i] == state_token_id:
    #     #             previous_state_idx = i
    #     #             state_counter += 1
        
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


# Create online samples for the two cycles dataset
class TwoCyclesDataset(torch.utils.data.IterableDataset):
    ### Question is always asked from the smaller cycle to the bigger cycle in the current version. 
    def __init__(self, cycle_sizes=None, question_pair=True, scratchpad="full", alphabet=alphabet, file_name=None, edge_char=">", sep_char=";", alphabet_size=None, completedlabels=False, state_char='#', start_char=':', eos='.', tokenizer=None, block_length=128, ER=None, meta=False, huggingface=False):
        super(TwoCyclesDataset, self).__init__()
        self.cycle_sizes = cycle_sizes
        self.scratchpad = scratchpad
        self.alphabet = alphabet
        self.file_name = file_name
        self.edge_char = edge_char
        self.sep_char = sep_char
        self.alphabet_size = alphabet_size
        self.completedlabels = completedlabels
        self.state_char = state_char
        self.start_char = start_char
        self.question_pair = question_pair
        self.special_char = False
        self.alphabet = self.alphabet[:self.alphabet_size] if alphabet_size is not None else alphabet
        self.eos = eos
        self.tokenizer = tokenizer
        self.inductive_form = ("induct" in scratchpad)
        self.block_length = block_length
        self.ER = ER
        self.meta = meta
        self.huggingface = huggingface


    # Create Erdos-Renyi/random graphs
    def create_sample_ER(self, n, p, distance, return_meta=False):
        while True:
            if p <= 1: # Using probabilities
                g = nx.erdos_renyi_graph(n, p, directed=True)
            else: # Using number of edges
                g = nx.gnm_random_graph(n, p, directed=True)
            shortest_paths = dict(nx.all_pairs_shortest_path_length(g))
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    distances[i, j] = shortest_paths[i].get(j, -1)
            xs, ys = np.where(distances == distance)
            if len(xs) == 0:
                continue
            else:
                source = xs[0]
                dest = ys[0]
                vertices = np.random.choice(alphabet, size=n, replace=False)
                edges = [vertices[u]+f" {self.edge_char} "+vertices[v]+f" {self.sep_char} " for u, v in list(g.edges)]
                np.random.shuffle(edges)
                label = 1 if distance >= 0 else 0
                x =  "".join(edges) + vertices[source] + " ? " + vertices[dest] + f" {self.sep_char}"
                y = f"{label} {self.eos}"
                if not return_meta:
                    return [(x, x + " " + y)]
                else:
                    source_degree = g.in_degree(source), g.out_degree(source)
                    dest_degree = g.in_degree(dest), g.out_degree(dest)
                    triangles = nx.triangles(g.to_undirected())
                    source_triangles, dest_traingles = triangles[source], triangles[dest]
                    # We save some meta data on the samples to interpret the model's behavior later. 
                    meta = {'source_degree': source_degree, 'dest_degree': dest_degree, 'source_triangles': source_triangles, 'dest_triangles': dest_traingles, 'dist': distance}
                    return [(x, x + " " + y, meta)]


    def create_sample(self, cycle_sizes, dist=-1):
        # The first question node is always from the first cycle. If dist=-1 the second question node will be from the second cycle. Otherwise, it will be dist nodes away from the first question node from the first cycle.
        label = 0 if dist == -1 else 1
        total_nodes = sum(cycle_sizes)
        vertices = np.random.choice(self.alphabet, size=total_nodes, replace=False)
        edges = []
        cycles_offsets = []
        offset = 0
        for j, size in enumerate(cycle_sizes):
            for i in range(size):
                edges.append(vertices[i + offset] + " " + self.edge_char + " " + vertices[(i+1) % size + offset] + " " +self.sep_char)
            offset += size
            cycles_offsets += [j] * size
        # permuting        
        perm = np.random.permutation(len(edges))
        first_index = perm[0] if not self.question_pair else 0
        first_node = vertices[first_index]
        first_node_cycle = cycles_offsets[first_index]
        modulo = cycle_sizes[first_node_cycle]
        bias = ([0] + list(np.cumsum(cycle_sizes)))[first_node_cycle]
        edges = [edges[i] for i in perm]
        second_node = vertices[cycle_sizes[1]] if dist == -1 else vertices[dist]
        if self.question_pair:
            edges = edges + [first_node + " ? " + second_node + " " + self.sep_char]
        x = " ".join(edges)
        # Full DFS scratchpad. One can possibly have jumps (i.e., skipping nodes in the scratchpad) to make the task harder.
        if self.scratchpad.startswith("full"):
            if self.scratchpad == "full":
                self.scratchpad = "full1"
            jump = int(self.scratchpad[4:])
            assert (dist == -1 or dist % jump == 0) and (cycle_sizes[0] % jump == 0), "dist and cycle_sizes[0] must be divisible by jump of the scratchpad."
            if label == 0:
                y = (" " + self.edge_char + " ").join([vertices[(first_index + i) % modulo + bias] for i in range(0, cycle_sizes[0] + 1, jump)]) + f" {self.sep_char} {label} {self.eos}"
            else:
                y = (" " + self.edge_char + " ").join([vertices[(first_index + i) % modulo + bias] for i in range(0, dist + 1, jump)]) + f" {self.sep_char} {label} {self.eos}"
            return [(x, x + " " + y)]
        elif self.scratchpad == "induct" and self.question_pair:
            # states = []
            dist = dist if dist != -1 else modulo
            y = (" " + self.state_char + " ").join([vertices[(first_index + i) % modulo + bias] for i in range(0, dist + 1)]) + f" {self.sep_char} {label} {self.eos}"
            return [(x, x + " " + self.start_char + " " + y)]
        else:
            y = f"{label} {self.eos}"
            return [(x, x + " " + y)]


    def __iter__(self):
        while True:
            texts = []
            if self.ER is None:
                for _ in range(10):
                    for sizes, dist in self.cycle_sizes:
                        texts += self.create_sample(sizes, dist=dist)
            else:
                n, m, distances = self.ER
                for d in distances:
                    texts += self.create_sample_ER(n, m, d, return_meta=self.meta)
            
            if self.huggingface:
                X = self.tokenizer([x.replace(" ", "") for x, _ in texts], return_tensors='pt', padding="longest", max_length=self.block_length, truncation=True)
                target_encoding = self.tokenizer([y.replace(" ", "").replace(self.eos, "")[len(x.replace(" ", "")): ] for x, y in texts], padding="longest", max_length=self.block_length, truncation=True, return_tensors="pt")
                labels = target_encoding.input_ids
                # replace padding token id's of the labels by -100 so it's ignored by the loss
                labels[labels == self.tokenizer.pad_token_id] = -100
                for i in range(len(texts)):
                    yield X['input_ids'][i], X['attention_mask'][i], labels[i]
            else:   
                if self.tokenizer is not None:
                    for i in range(len(texts)):
                        if self.meta:
                            meta = texts[i][-1]
                        texts[i] = [np.array(self.tokenizer.encode(texts[i][0]), dtype=np.int16), self.tokenizer.encode(texts[i][1]), None, None, None, None]
                        if self.inductive_form:
                            texts[i][1], texts[i][2], texts[i][3], texts[i][4], texts[i][5] = self.tokenizer.to_inductive_form([texts[i][1]], self.block_length)
                            texts[i][1], texts[i][2], texts[i][3], texts[i][4], texts[i][5] = texts[i][1][0], texts[i][2][0], texts[i][3][0], texts[i][4][0], texts[i][5][0]
                        if self.return_meta:
                            texts[i].append(meta)
                random.shuffle(texts)
                for text in texts:
                    yield text


# Tokenizer for the two cycles dataset
class TwoCyclesTokenizer():
    def __init__(self, sep_char=';', edge_char='>', start_char = ':', state_char='#', pad_token=' ', eos_token='.', alphabet=alphabet, numbers=list('0123456789')):
        self.sep_char = sep_char
        self.edge_char = edge_char
        self.state_char = state_char
        self.pad_token = pad_token
        self.start_char = start_char
        self.eos_token = eos_token
        self.alphabet = alphabet
        self.numbers = numbers
        chars = list(set([sep_char, edge_char, start_char, state_char, pad_token, eos_token, '?', '|'])) + alphabet + numbers
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
    
    # def encode_dataset(self, X, Y):
    #     text = []
    #     for i in range(len(X)):
    #         text.append(self.encode("".join(self._to_str(X[i])) + self.input_output_token + self.separator.join([self._to_str(Y[i, j, :]) for j in range(Y.shape[1])]) + self.eos_token))
    #     return text
        
    def decode(self, l):
        return ''.join([self.itos.get(i, "_") for i in l])
    
    def decode_batch(self, X):
        return [self.decode(x) for x in X]

# Running this part will create a small sample dataset and will also create the meta.pkl file.
if __name__ == '__main__':
    tokenizer = TwoCyclesTokenizer()
    size = 4
    dataset = TwoCyclesDataset(cycle_sizes=[([size, size], -1), ([2 * size], size)], scratchpad='inductnew')
    # dataset = TwoCyclesDataset(cycle_sizes=[([size, size], -1), ([2 * size], size)], scratchpad='none')
    # dataset = TwoCyclesDataset(cycle_sizes=[([s, s], -1) for s in range(2, size + 1)] + [([2 * s], s) for s in range(2, size + 1)], scratchpad='none')
    # dataset = TwoCyclesDataset(cycle_sizes=(5,5), scratchpad='full')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=1)
    for _, X in data_loader:
        X = torch.tensor(tokenizer.encode_batch(X), dtype=torch.int16)
        print("Number of tokens:", X.shape[-1])
        X_inductive_form, _, _, _, lengths = get_inductive_form_train(X, tokenizer.stoi[tokenizer.pad_token], tokenizer.stoi[tokenizer.state_char], tokenizer.stoi[tokenizer.start_char], [tokenizer.stoi[tokenizer.eos_token]], size * 20, return_list=False)
        print("Inductive form length:", max(lengths))
        X_decoded = tokenizer.decode_batch(X_inductive_form.tolist())
        for x in X_decoded:
            print(x)
        break
    # print(get_inductive_form_train(X[0:1], tokenizer.stoi[tokenizer.pad_token], tokenizer.stoi[tokenizer.state_char], tokenizer.stoi[tokenizer.start_char], [tokenizer.stoi[tokenizer.eos_token]], 128, return_list=False)[-3][0][:55, :55][50])
    print(get_inductive_form_train(X[0:1], tokenizer.stoi[tokenizer.pad_token], tokenizer.stoi[tokenizer.state_char], tokenizer.stoi[tokenizer.start_char], [tokenizer.stoi[tokenizer.eos_token]], size * 20, return_list=False)[3])
    meta = {
        'vocab_size': tokenizer.vocab_size,
        'itos': tokenizer.itos,
        'stoi': tokenizer.stoi,
        'tokens': X.shape[-1]
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)