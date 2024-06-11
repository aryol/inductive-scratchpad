import os
import pickle
import torch
import torch
from torch.utils.data import IterableDataset, DataLoader
import random
import numpy as np
import networkx as nx
import tqdm
import random

alphabet = list(chr(c) for c in range(ord('a'), ord('z')+1)) + list(chr(c) for c in range(ord('0'), ord('9')+1)) + ['$']

def spacify(text):
    # put a space between each two chars of the text
    return " ".join(list(text))

def std_num(x):
    if x == -1:
        return "- 1"
    else:
        return spacify(f"{x:02}")

def num(c):
    if isinstance(c, int):
        return c
    else:
        return 0

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

# Addition dataset with fixed samples
class AdditionDataset(torch.utils.data.Dataset):
    ### Addition question generator
    def __init__(self, N, digit_dist, max_digit_length: int, scratchpad_type='random-space', equals_sign='=', emtpy_char='_', state_char='#', separator=',', start_char=':', eos='.', finish='!', separator_ans='|', tokenizer=None, block_length=128):
        super(AdditionDataset, self).__init__()
        self.N = N
        self.digit_dist = digit_dist
        self.max_digit_length = max_digit_length
        self.scratchpad_type = scratchpad_type
        self.equals_sign = equals_sign
        self.emtpy_char = emtpy_char
        self.state_char = state_char
        self.start_char = start_char
        self.eos = eos
        self.tokenizer = tokenizer
        self.separator = separator
        self.finish = finish
        self.separator_ans = separator_ans
        self.data = []
        for i in tqdm.tqdm(range(N)):
            size = np.random.choice(np.arange(len(self.digit_dist), 0, -1), 1, p=self.digit_dist)[0]
            size = int(size)
            op_1 = random.randrange(10 ** (size - 1), 10 ** size)
            op_2 = random.randrange(10 ** (size - 1), 10 ** size)
            if self.scratchpad_type == 'shift':
                text = self.create_sample_shift(op_1, op_2)[0]
            else:
                text = self.create_sample_random_space(op_1, op_2)[0]
            self.data.append(text)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        text = self.data[idx]
        if self.tokenizer is not None:
            question, ans = text
            text = [np.array(self.tokenizer.encode(text[0]), dtype=np.int16), self.tokenizer.encode(text[1]), None, None, None, None, None]
            text[2], text[3], text[4], text[5], text[6] = self.tokenizer.to_inductive_form([text[1]])
            text[0], text[1], text[2], text[3], text[4], text[5], text[6] = question, ans, text[2][0], text[3][0], text[4][0], text[5][0], text[6][0]
        return text   
    
    # Shift scratchpad
    def create_sample_shift(self, operand_1, operand_2):
        result = str(operand_1 + operand_2)
        operand_1 = str(operand_1)
        operand_2 = str(operand_2)
        size_1, size_2, size_result = len(operand_1), len(operand_2), len(result)
        if size_result == max(size_1, size_2) - 1:
            result = "0" + result
        random_string = ''.join(np.random.choice(alphabet, self.max_digit_length * 3 + 1, replace=True))
        text_a = random_string[:self.max_digit_length - size_1] + '$' + operand_1
        text_b = random_string[self.max_digit_length: 2 * self.max_digit_length - size_2] + '$' + operand_2
        text_result = '$' + random_string[self.max_digit_length * 2:-1]
        text_question = text_a + '+' + text_b + '='
        scratchpad_text = text_result + '|0' + self.state_char
        carry = 0
        for _ in range(max(size_1, size_2) + 1):
            i_a = int(text_a[-1]) if text_a[-1] != '$' else '$'
            i_b = int(text_b[-1]) if text_b[-1] != '$' else '$'
            if text_a[-1] == '$' and text_b[-1] == '$':
                # Addition is finished.
                state_new = str(carry) + text_result + self.eos
                scratchpad_text += state_new
            else:
                text_result = str((i_a + i_b + carry) % 10) + text_result[:-1]
                carry = 1 if i_a + i_b + carry >= 10 else 0
                if text_a[-1] != '$':
                    text_a = str(i_a) + text_a[:-1]
                if text_b[-1] != '$':
                    text_b = str(i_b) + text_b[:-1]
                state_new = text_a + '+' + text_b + '=' + text_result + '|' + str(carry) + self.state_char
                scratchpad_text += state_new
        return [(spacify(text_question), spacify(text_question + scratchpad_text))]

    # Random space scratchpad
    def create_sample_random_space(self, operand_1, operand_2):
        result = str(operand_1 + operand_2)
        operand_1 = str(operand_1)
        operand_2 = str(operand_2)
        size_1, size_2, size_result = len(operand_1), len(operand_2), len(result)
        if size_result == max(size_1, size_2):
            result = "0" + result # NOTE: one can remove this. 
        
        question = ['_'] * (self.max_digit_length * 2 + 1)
        question_no_space = operand_1 + '+' + operand_2
        random_positions = np.random.choice(np.arange(self.max_digit_length * 2 + 1), size_1 + size_2 + 1, replace=False)
        random_positions = np.sort(random_positions)
        for i in range(size_1 + size_2 + 1):
            question[random_positions[i]] = question_no_space[i]
        question = ''.join(question) + '='
        random_ans = ['$'] + list(np.random.choice(alphabet, self.max_digit_length + 1, replace=True))
        # question += ''.join(random_ans)
        # scratchpad_text = f"{self.start_char}"
        scratchpad_text = ' '.join(random_ans) + f" {self.start_char}"
        
        carry = 0
        for i in range(size_result):
            p_a = random_positions[size_1 - 1 - i] if size_1 - 1 - i >= 0 else -1
            p_b = random_positions[size_1 + size_2 - i] if size_1 + size_2 - i > size_1 else random_positions[size_1]
            i_a = int(operand_1[-i-1]) if i < len(operand_1) else '_'
            i_b = int(operand_2[-i-1]) if i < len(operand_2) else '_'
            carry = 1 if num(i_a) + num(i_b) + carry >= 10 else 0
            random_ans[1:] = random_ans[:-1]
            random_ans[0] = result[-i-1]
            state_text = f"[ {std_num(p_a)} ] {i_a} [ {std_num(p_b)} ] {i_b} c {carry} r {' '.join(random_ans)}"
            scratchpad_text = scratchpad_text + " " + state_text + " "
            if i < size_result - 1:
                scratchpad_text += self.state_char
            else:
                scratchpad_text += self.eos
        text_question = spacify(question)
        return [(text_question, text_question + " " + scratchpad_text)]


# Addition dataset with online samples
class AdditionIterDataset(torch.utils.data.IterableDataset):
    ### Addition question generator
    def __init__(self, digit_dist, max_digit_length: int, scratchpad_type='random-space', equals_sign='=', emtpy_char='_', state_char='#', separator=',', start_char=':', eos='.', finish='!', separator_ans='|', tokenizer=None, block_length=128, meta=False):
        super(AdditionIterDataset, self).__init__()
        self.digit_dist = digit_dist
        self.max_digit_length = max_digit_length
        self.scratchpad_type = scratchpad_type
        self.equals_sign = equals_sign
        self.emtpy_char = emtpy_char
        self.state_char = state_char
        self.start_char = start_char
        self.eos = eos
        self.tokenizer = tokenizer
        self.block_length = block_length
        self.meta = meta
        self.separator = separator
        self.finish = finish
        self.separator_ans = separator_ans

    
    # Random space scratchpad
    def create_sample_random_space(self, operand_1, operand_2):
        result = str(operand_1 + operand_2)
        operand_1 = str(operand_1)
        operand_2 = str(operand_2)
        size_1, size_2, size_result = len(operand_1), len(operand_2), len(result)
        if size_result == max(size_1, size_2):
            result = "0" + result # NOTE: one can remove this. 
        
        question = ['_'] * (self.max_digit_length * 2 + 1)
        question_no_space = operand_1 + '+' + operand_2
        random_positions = np.random.choice(np.arange(self.max_digit_length * 2 + 1), size_1 + size_2 + 1, replace=False)
        random_positions = np.sort(random_positions)
        for i in range(size_1 + size_2 + 1):
            question[random_positions[i]] = question_no_space[i]
        question = ''.join(question) + '='
        random_ans = ['$'] + list(np.random.choice(alphabet, self.max_digit_length + 1, replace=True))
        # question += ''.join(random_ans)
        # scratchpad_text = f"{self.start_char}"
        scratchpad_text = ' '.join(random_ans) + f" {self.start_char}"
        
        carry = 0
        for i in range(size_result):
            p_a = random_positions[size_1 - 1 - i] if size_1 - 1 - i >= 0 else -1
            p_b = random_positions[size_1 + size_2 - i] if size_1 + size_2 - i > size_1 else random_positions[size_1]
            i_a = int(operand_1[-i-1]) if i < len(operand_1) else '_'
            i_b = int(operand_2[-i-1]) if i < len(operand_2) else '_'
            carry = 1 if num(i_a) + num(i_b) + carry >= 10 else 0
            random_ans[1:] = random_ans[:-1]
            random_ans[0] = result[-i-1]
            state_text = f"[ {std_num(p_a)} ] {i_a} [ {std_num(p_b)} ] {i_b} c {carry} r {' '.join(random_ans)}"
            scratchpad_text = scratchpad_text + " " + state_text + " "
            if i < size_result - 1:
                scratchpad_text += self.state_char
            else:
                scratchpad_text += self.eos
        text_question = spacify(question)
        return [(text_question, text_question + " " + scratchpad_text)]
    
    # Shift scratchpad
    def create_sample_shift(self, operand_1, operand_2):
        result = str(operand_1 + operand_2)
        operand_1 = str(operand_1)
        operand_2 = str(operand_2)
        size_1, size_2, size_result = len(operand_1), len(operand_2), len(result)
        if size_result == max(size_1, size_2) - 1:
            result = "0" + result
        random_string = ''.join(np.random.choice(alphabet, self.max_digit_length * 3 + 1, replace=True))
        text_a = random_string[:self.max_digit_length - size_1] + '$' + operand_1
        text_b = random_string[self.max_digit_length: 2 * self.max_digit_length - size_2] + '$' + operand_2
        text_result = '$' + random_string[self.max_digit_length * 2:-1]
        text_question = text_a + '+' + text_b + '='
        scratchpad_text = text_result + '|0' + self.state_char
        carry = 0
        for _ in range(max(size_1, size_2) + 1):
            i_a = int(text_a[-1]) if text_a[-1] != '$' else '$'
            i_b = int(text_b[-1]) if text_b[-1] != '$' else '$'
            if text_a[-1] == '$' and text_b[-1] == '$':
                # Addition is finished.
                state_new = str(carry) + text_result + self.eos
                scratchpad_text += state_new
            else:
                text_result = str((i_a + i_b + carry) % 10) + text_result[:-1]
                carry = 1 if i_a + i_b + carry >= 10 else 0
                if text_a[-1] != '$':
                    text_a = str(i_a) + text_a[:-1]
                if text_b[-1] != '$':
                    text_b = str(i_b) + text_b[:-1]
                state_new = text_a + '+' + text_b + '=' + text_result + '|' + str(carry) + self.state_char
                scratchpad_text += state_new
        return [(spacify(text_question), spacify(text_question + scratchpad_text))]
    
    def __iter__(self):
        while True:
            texts = []
            for _ in range(10):
                # digit dist looks like [0.1 0.3 0.5 0.1] for example. take a sample according to it
                # size_1, size_2 = np.random.choice(np.arange(len(self.digit_dist), 0, -1), 2, p=self.digit_dist)
                size = np.random.choice(np.arange(len(self.digit_dist), 0, -1), 1, p=self.digit_dist)[0]
                size = int(size)
                op_1 = random.randrange(10 ** (size - 1), 10 ** size)
                op_2 = random.randrange(10 ** (size - 1), 10 ** size)
                if self.scratchpad_type == 'shift':
                    texts += self.create_sample_shift(op_1, op_2)
                else:
                    texts += self.create_sample_random_space(op_1, op_2)
            
            if self.tokenizer is not None:
                for i in range(len(texts)):
                    texts[i] = [np.array(self.tokenizer.encode(texts[i][0]), dtype=np.int16), self.tokenizer.encode(texts[i][1]), None, None, None, None]
                    if self.inductive_form:
                        texts[i][1], texts[i][2], texts[i][3], texts[i][4], texts[i][5] = self.tokenizer.to_inductive_form([texts[i][1]], self.block_length)
                        texts[i][1], texts[i][2], texts[i][3], texts[i][4], texts[i][5] = texts[i][1][0], texts[i][2][0], texts[i][3][0], texts[i][4][0], texts[i][5][0]
                    if self.return_meta:
                        texts[i].append(meta)
            random.shuffle(texts)
            for text in texts:
                yield text

# Tokenizer
class AdditionTokenizer():
    def __init__(self, block_length=512, equals_sign='=', emtpy_char='_', state_char='#', sep_char=',', start_char=':', eos_token='.', pad_token = ' ', numbers=list('0123456789')):
        self.equals_sign = equals_sign
        self.sep_char = sep_char
        self.emtpy_char = emtpy_char
        self.state_char = state_char
        self.start_char = start_char
        self.eos_token = eos_token
        self.numbers = numbers
        self.pad_token = pad_token
        self.block_length = block_length
        chars = list(set([equals_sign, sep_char, emtpy_char, start_char, state_char, pad_token, eos_token, '?', '|', '+', '!', '-', '[', ']'] + alphabet))
        self.vocab_size = len(chars)
        print("Vocab size:", self.vocab_size)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def _to_str(self, x):
        # convert a tensor to a list
        return "".join(map(str, x.tolist()))

    def to_inductive_form(self, X, block_length=None):
        if block_length is None:
            block_length = self.block_length
        return get_inductive_form_train(X, self.stoi[self.pad_token], self.stoi[self.state_char], self.stoi[self.start_char], [self.stoi[self.eos_token]], compute_loss_before_start=True, block_length=block_length, return_list=False)

    
    def encode(self, s):
        # return [self.stoi[c] for c in s]
        try:
            return list(map(lambda x: self.stoi[x], s.split(" ")))
        except:
            print(s + "$")
            exit()
    
    def encode_batch(self, X):
        X = [self.encode(x) for x in X]
        max_len = max([len(x) for x in X])
        X = [x + [self.stoi[self.pad_token]] * (max_len - len(x)) for x in X]
        return X
        
    def decode(self, l):
        return ''.join([self.itos.get(i, "x") for i in l])
    
    def decode_batch(self, X):
        return [self.decode(x) for x in X]


# Running this part will create a small sample dataset and will also create the meta.pkl file. 
if __name__ == '__main__':
    tokenizer = AdditionTokenizer()
    N = 4
    dist = np.arange(2, 0, -1)
    dataset = AdditionIterDataset(dist / dist.sum(), N, scratchpad_type='shift')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, num_workers=1)
    for _, X in data_loader:
        X = torch.tensor(tokenizer.encode_batch(X), dtype=torch.int16)
        print("Number of tokens:", X.shape[-1])
        X_inductive_form, _, _, _, lengths = get_inductive_form_train(X, tokenizer.stoi[tokenizer.pad_token], tokenizer.stoi[tokenizer.state_char], tokenizer.stoi[tokenizer.start_char], [tokenizer.stoi[tokenizer.eos_token]], 4000, return_list=False)
        print("Inductive form length:", max(lengths))
        X_decoded = tokenizer.decode_batch(X_inductive_form.tolist())
        for x in X_decoded:
            print(x)
        break
    # print(get_inductive_form_train(X[0:1], tokenizer.stoi[tokenizer.pad_token], tokenizer.stoi[tokenizer.state_char], tokenizer.stoi[tokenizer.start_char], [tokenizer.stoi[tokenizer.eos_token]], 128, return_list=False)[-3][0][:55, :55][50])
    print(get_inductive_form_train(X[0:1], tokenizer.stoi[tokenizer.pad_token], tokenizer.stoi[tokenizer.state_char], tokenizer.stoi[tokenizer.start_char], [tokenizer.stoi[tokenizer.eos_token]], 4000, return_list=False)[3])
    meta = {
        'vocab_size': tokenizer.vocab_size,
        'itos': tokenizer.itos,
        'stoi': tokenizer.stoi,
        'tokens': X.shape[-1]
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
