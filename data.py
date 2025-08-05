# Data loading and Preprocessing
import os
import random
import numpy as np
import torch
import math
import os
import torch.nn.functional as F
from utils import set_seeds

# Define vocabulary and tokens
vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '&', '*', '+'] # remeber to delete '+' for the string copy task
padding_token_index = 12  # '' is the padding token
end_token_index = 11      # '&' is the end token

# Create a mapping from string to interger
stoi = {ch:i for i, ch in enumerate(vocab)}
itos = {i:ch for i, ch in enumerate(vocab)}
encode = lambda s:[stoi[c] for c in s] # encoder: take a string, output a list of intergers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of intergers, output a string

def generate_origin_dataset(original, task, num_samples=2000000, data_dir="data"):
    
    file_path = os.path.join(data_dir, f"origin_ds_{task}.txt")
    if os.path.exists(file_path):
        print(f"File {file_path} already exists.\nSkipping original dataset generation.")
        return
    
    if task == 'copy':
        # generate 200000 sample
        a_values = np.random.randint(1, original + 1, size=num_samples)
        strings = ["".join(np.random.choice([str(i) for i in range(10)], size=a)) for a in a_values]  # random generate strings
        target = strings
        to_write = [f"{a}={b}&" for a, b in zip(strings, target)]

        # write down
        with open(file_path, "w") as f:
            f.write("\n".join(to_write))
            
    elif task == 'reverse_addition':
        exp_a = random.choices(range(1, original + 1), k=num_samples)
        exp_b = random.choices(range(1, original + 1), k=num_samples)

        # same operand length
        # exponents = random.choices(range(1, original+1), k=num_samples)

        a = [random.randint(10**(exp-1), 10**exp - 1) for exp in exp_a]
        b = [random.randint(10**(exp-1), 10**exp - 1) for exp in exp_b]
        c = [x + y for x, y in zip(a, b)]

        data_list = [
            f"{str(i)[::-1]}+{str(j)[::-1]}={str(k)[::-1]}&"
            for i, j, k in zip(a, b, c)
        ]

        with open(file_path, "w") as f:
            f.write("\n".join(data_list))

    print(f"{num_samples} original data for task {task} is saved in {file_path}")

def get_batch(data, batch_size=1024, block_size=100, device='cuda'):
    final_sample = random.sample(data, batch_size)
    final_sample = [line.strip() for line in final_sample]

    x_list, y_list = [], []
    for x_str in final_sample:
        # print(x_str)
        x_encoded = encode(x_str)
        x_padded = x_encoded + [padding_token_index] * (block_size - len(x_encoded))
        x_list.append(torch.tensor(x_padded, dtype=torch.int64))
        y_encoded = encode(x_str)[1:]
        y_encoded.append(end_token_index)
        y_padded = y_encoded + [padding_token_index] * (block_size - len(y_encoded))
        y_list.append(torch.tensor(y_padded, dtype=torch.int64))

    x_tensor = torch.stack(x_list).to(device)
    y_tensor = torch.stack(y_list).to(device)
    return x_tensor, y_tensor

def generate_prompt_OOD(si_round, task, original):
    
    if task == 'copy':
        strings = "".join(np.random.choice([str(i) for i in range(10)], size=si_round+original))
        prompt_str = f"{str(strings)}="  
        
    elif task == 'reverse_addition':
        exp = original+si_round
        # print(exp)
        a = [random.randint(10**(exp-1), 10**(exp)-1) for _ in range(1)]  
        b = [random.randint(10**(exp-1), 10**(exp)-1) for _ in range(1)] 
        prompt_str = f"{str(a[0])[::-1]}+{str(b[0])[::-1]}=" 

    return prompt_str

def generate_baseline_problems(num_digits, num_samples=10000):
    problems = []
    for _ in range(num_samples):
        a = random.randint(10**(num_digits-1), 10**num_digits - 1)
        b = random.randint(10**(num_digits-1), 10**num_digits - 1)

        problem = f"{str(a)[::-1]}+{str(b)[::-1]}="
        problems.append(problem)
    return problems

def insert_digit_at_position(number_str, digit, position):

    digits = list(number_str)
    digits.insert(position, str(digit))
    return ''.join(digits)

def create_modified_problems(baseline_problems, insertion_position, num_samples=10000):
    modified_problems = []

    for problem in baseline_problems[:num_samples]:
        # Extract operands from problem string
        parts = problem.split('+')
        x_str = parts[0]
        y_str = parts[1].split('=')[0]

        # Insert random digit in operand X at position j
        random_digit = random.randint(0, 9)
        modified_x = insert_digit_at_position(x_str, random_digit, insertion_position)

        # Y remains unchanged, create new problem
        modified_problem = f"{modified_x}+{y_str}="
        modified_problems.append(modified_problem)

    return modified_problems

# model.generate() function
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.00001, top_k=None):
  
    batch_size, seq_len = idx.shape
    idx = idx.to(model.device)

    # Track which sequences are still active (not finished)
    is_active = torch.ones(batch_size, dtype=torch.bool, device=model.device)

    for _ in range(max_new_tokens):
        if not is_active.any():
            break
        # Ensure context length does not exceed model's block size
        idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]

        # Forward pass to get logits
        logits, _ = model(idx_cond)

        # Extract logits for the last token and apply temperature scaling
        logits = logits[:, -1, :] / temperature

        # Apply top-k filtering if necessary
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample next token
        idx_next = torch.multinomial(probs, num_samples=1)

        for i in range(batch_size):
            if is_active[i] and idx_next[i].item() == encode('&')[0]:
                is_active[i] = False 

        # Stop if all sequences have reached eos
        if not is_active.any():
            break

        # Append sampled token to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    decoded_texts = []
    for seq in idx.tolist():
        text = decode(seq)
        cut_text = text.split('&')[0]  
        decoded_texts.append(cut_text)

    return decoded_texts


def save_baseline_problems(num_digits, num_samples=10000, filename=None):
    if filename is None:
        filename = f"{num_digits}+{num_digits}.txt"
    
    problems = generate_baseline_problems(num_digits, num_samples)
    
    with open(filename, 'w') as f:
        for problem in problems:
            f.write(problem + '\n')
    
    print(f"save to {filename}")
    return problems

def load_baseline_problems(filename):
    with open(filename, 'r') as f:
        problems = [line.strip() for line in f.readlines()]
    return problems

def save_modified_problems(baseline_problems, insertion_position, num_samples=10000, filename=None):
    if filename is None:
        filename = f"modified_problems_pos{insertion_position}.txt"
    
    modified_problems = create_modified_problems(baseline_problems, insertion_position, num_samples)
    
    with open(filename, 'w') as f:
        for problem in modified_problems:
            f.write(problem + '\n')
    
    print(f"save to {filename}")
    return modified_problems

def load_modified_problems(filename):
    with open(filename, 'r') as f:
        problems = [line.strip() for line in f.readlines()]
    return problems

def load_response(filename):
    with open(filename, 'r') as f:
        responses = [line.strip() for line in f.readlines()]
    return responses

if __name__ == "__main__":
    
    # generate_origin_dataset(original=10, task='reverse_addition')

    baseline_problems = save_baseline_problems(13, num_samples=10000)

    # # Case 1: Insert digit at position 5, X becomes 11 digits
    # modified_11 = save_modified_problems(baseline_problems, 5, filename="11+10.txt")

    # # Case 2: Insert another digit at position 5 of the 11-digit problems, X becomes 12 digits  
    # modified_12 = save_modified_problems(modified_11, 5, filename="12+10.txt")

    # # Case 3: Insert another digit at position 5 of the 12-digit problems, X becomes 13 digits
    # modified_13 = save_modified_problems(modified_12, 5, filename="13+10.txt")
        
    
    
