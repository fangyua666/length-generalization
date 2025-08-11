import torch
import os
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

from utils import set_seeds
# from model import GPT
from model_t5_pe import GPT
# from model_rope import GPT

from data import get_batch, load_baseline_problems
from evaluation import get_model_responses
from data import generate

set_seeds(42)

# Loading test ood data and model checkpoint
exp_a, exp_b = 11, 11  # number of digits
path_to_ood_samples = os.path.join("data", "ood-samples", f"{exp_a}+{exp_b}_responses.txt")
path_to_baseline_model = "/workspace/length-generalization/models/ra/T5_ra.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# problems = load_baseline_problems(path_to_ood_samples)  # Not used in analysis
model = GPT(vocab_size=14, block_size=100, n_embd=384, n_layer=6, n_head=6, dropout=0.0, bias=True).to(device)
ckpt = path_to_baseline_model
model.load_state_dict(torch.load(ckpt, map_location=device))

# Define vocabulary and tokens
vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '&', '*', '+']  # remember to delete '+' for the string copy task
padding_token_index = 12  # '' is the padding token
end_token_index = 11      # '&' is the end token

# Create a mapping from string to integer
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Generate test samples and evaluate
correct = 0
total = 1024
block_size = 100
batch_size = 1024
num_batches = total // batch_size

for _ in range(num_batches):
    a_list = [random.randint(10**(exp_a-1), 10**(exp_a)-1) for _ in range(batch_size)]
    b_list = [random.randint(10**(exp_b-1), 10**(exp_b)-1) for _ in range(batch_size)]
    prompt_str = [f"{str(i)[::-1]}+{str(j)[::-1]}=" for i, j in zip(a_list, b_list)]

    context = torch.tensor([encode(inp) for inp in prompt_str], dtype=torch.long, device=device)

    # output in batch
    output_batch, probs_batch, logits_batch = generate(model=model, idx=context, max_new_tokens=block_size, temperature=1, top_k=1, output_probs=True)

    answers = [str(i + j)[::-1] for i, j in zip(a_list, b_list)]
    targets = [p + ans for p, ans in zip(prompt_str, answers)]

    correct += sum([output == target for output, target in zip(output_batch, targets)])

acc = correct / total
print(f"Accuracy: {acc}")

# Display example
print(f"Example output: {output_batch[15]}")
print(f"Example target: {targets[15]}")

def calc_entropy(probs: torch.Tensor, targets: torch.Tensor, eps=1e-6):
    """
    calc_entropy calculates the entropy of prediction probabilities
    """
    n, T, V = probs.shape
    assert targets.shape == (n, T)
    entropy = torch.sum((-1) * probs * torch.log(probs + eps), dim=-1)
    return entropy

def calc_spurious_solutions(targets: List[str], shift_range=2) -> Dict[str, List[str]]:
    """
    calc_spurious_solutions calculates plausible but incorrect solutions with common mistake patterns
    """
    n = len(targets)
    keys = [f"shift_a_{shift_a}_shift_b_{shift_b}_{part}" for shift_a in range(shift_range+1) for shift_b in range(shift_range+1) for part in ["res", "carry"]]
    keys += [f"shift_a_{shift_a}_shift_b_{shift_b}_res_using_correct_carry" for shift_a in range(shift_range+1) for shift_b in range(shift_range+1)]
    spurious_solutions = {key: [None]*n for key in keys}
    
    for i, target in tqdm(enumerate(targets)):
        plus_index = target.index('+')
        equal_index = target.index('=')
        len_a, len_b, len_c = plus_index, equal_index - plus_index - 1, len(target) - equal_index - 1
        max_len = max(len_a, len_b) + 2*shift_range  # right padding with zeros for convenience
        s_a = target[:plus_index].ljust(max_len, '0')
        s_b = target[plus_index+1:equal_index].ljust(max_len, '0')
        s_c = target[equal_index+1:]
        
        for shift_a in range(shift_range+1):
            for shift_b in range(shift_range+1):
                shifted_a = s_a[shift_a:]  # shift left by shift_a digits (the same as removing some digits)
                shifted_b = s_b[shift_b:]
                res = int(shifted_a[::-1]) + int(shifted_b[::-1])  # redundant zeros are automatically removed
                res_str = str(res)[::-1]
                carry = [False]
                for k in range(1, len(res_str)):
                    carry.append(int(shifted_a[k-1]) + int(shifted_b[k-1]) + carry[-1] > 9)
                spurious_solutions[f"shift_a_{shift_a}_shift_b_{shift_b}_res"][i] = res_str
                spurious_solutions[f"shift_a_{shift_a}_shift_b_{shift_b}_carry"][i] = carry
                # addition of shifted digits, but with the correct carry
                carry = spurious_solutions["shift_a_0_shift_b_0_carry"][i]
                res_using_correct_carry = [(int(shifted_a[k]) + int(shifted_b[k]) + carry[k]) % 10 for k in range(len(res_str))]
                spurious_solutions[f"shift_a_{shift_a}_shift_b_{shift_b}_res_using_correct_carry"][i] = ''.join(map(str, res_using_correct_carry))
    
    return spurious_solutions

# Calculate spurious solutions
shift_range = 2
n = len(targets)
keys = [f"shift_a_{shift_a}_shift_b_{shift_b}_{part}" for shift_a in range(shift_range+1) for shift_b in range(shift_range+1) for part in ["res", "carry"]]
keys += [f"shift_a_{shift_a}_shift_b_{shift_b}_res_using_correct_carry" for shift_a in range(shift_range+1) for shift_b in range(shift_range+1)]
spurious_solutions = {key: [None]*n for key in keys}

for i, target in tqdm(enumerate(targets)):
    plus_index = target.index('+')
    equal_index = target.index('=')
    len_a, len_b, len_c = plus_index, equal_index - plus_index - 1, len(target) - equal_index - 1
    max_len = max(len_a, len_b) + 2*shift_range  # right padding with zeros for convenience
    s_a = target[:plus_index].ljust(max_len, '0')
    s_b = target[plus_index+1:equal_index].ljust(max_len, '0')
    s_c = target[equal_index+1:]
    for shift_a in range(shift_range+1):
        for shift_b in range(shift_range+1):
            shifted_a = s_a[shift_a:]  # shift left by shift_a digits (the same as removing some digits)
            shifted_b = s_b[shift_b:]
            res = int(shifted_a[::-1]) + int(shifted_b[::-1])  # redundant zeros are automatically removed
            res_str = str(res)[::-1]
            carry = [False]
            for k in range(1, len(res_str)):
                carry.append(int(shifted_a[k-1]) + int(shifted_b[k-1]) + carry[-1] > 9)
            spurious_solutions[f"shift_a_{shift_a}_shift_b_{shift_b}_res"][i] = res_str
            spurious_solutions[f"shift_a_{shift_a}_shift_b_{shift_b}_carry"][i] = carry
            # addition of shifted digits, but with the correct carry
            carry = spurious_solutions["shift_a_0_shift_b_0_carry"][i]
            res_using_correct_carry = [(int(shifted_a[k]) + int(shifted_b[k]) + carry[k]) % 10 for k in range(len(res_str))]
            spurious_solutions[f"shift_a_{shift_a}_shift_b_{shift_b}_res_using_correct_carry"][i] = ''.join(map(str, res_using_correct_carry))

print("First 10 targets:", targets[:10])
print("Example output:", output_batch[0])

# Calculate match results
keys_res = [key for key in keys if "carry" not in key or "using_correct_carry" in key]
match_res = {key: [None]*n for key in keys_res}
print("Keys for result matching:", keys_res)

for i, preds in tqdm(enumerate(output_batch)):
    equal_index = preds.index('=')
    len_c = len(preds) - equal_index - 1
    for j, key in enumerate(keys_res):
        match_res[key][i] = [preds[k+equal_index+1] == spurious_solutions[key][i].ljust(len_c+10, '0')[k] for k in range(len_c)]

# Calculate accuracy by position
L = max(exp_a, exp_b) + 1
accs = np.zeros((len(keys_res), L))
for k, key in enumerate(keys_res):
    for digit_index in range(L):
        accs[k, digit_index] = np.mean([match_res[key][i][digit_index] for i in range(n) if len(match_res[key][i]) > digit_index])

keys_res2 = [key for key in keys_res if "shift_a_0_shift_b_0" not in key]
accs2 = np.zeros((len(keys_res2), L))
counts2 = np.zeros((len(keys_res2), L))
for k, key in enumerate(keys_res2):
    for digit_index in range(L):
        accs2[k, digit_index] = np.mean([match_res[key][i][digit_index] for i in range(n) if len(match_res[key][i]) > digit_index and not match_res["shift_a_0_shift_b_0_res"][i][digit_index]])
        counts2[k, digit_index] = np.sum([match_res[key][i][digit_index] for i in range(n) if len(match_res[key][i]) > digit_index and not match_res["shift_a_0_shift_b_0_res"][i][digit_index]])

# Generate visualizations
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.heatmap(accs, ax=ax)
ax.set_yticklabels(keys_res, rotation=0)
ax.set_title(f"Addition accuracy with (left) shifted digits: {exp_a}-digit + {exp_b}-digit", weight="bold")
ax.set_xlabel(f"Digit index (0 means unit place)", weight="bold")
plt.savefig("accuracy_heatmap_full.png", dpi=300, bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(6*1, 6*2))
sns.heatmap(accs[1:9], ax=axs[0])
sns.heatmap(accs[-8:], ax=axs[1])
axs[0].set_yticklabels(keys_res[1:9], rotation=0)
axs[1].set_yticklabels(keys_res[-8:], rotation=0)
for k in range(2):
    axs[k].set_title(f"Addition accuracy with (left) shifted digits: {exp_a}-digit + {exp_b}-digit", weight="bold")
    axs[k].set_xlabel(f"Digit index (0 means unit place)", weight="bold")
plt.savefig("accuracy_heatmap_split.png", dpi=300, bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(6*1, 6*3))
sns.heatmap(accs2, ax=axs[0])
sns.heatmap(counts2, ax=axs[1])
sns.heatmap(np.where(counts2 > 9, accs2, 0), ax=axs[2])
for k in range(3):
    title_name = "accuracy" if k == 0 or k == 2 else "count"
    axs[k].set_yticklabels(keys_res2, rotation=0)
    axs[k].set_title(f"Conditional addition {title_name} with (left) shifted digits: {exp_a}-digit + {exp_b}-digit", weight="bold")
    axs[k].set_xlabel(f"Digit index (0 means unit place)", weight="bold")
plt.savefig("conditional_accuracy_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# Final analysis with unexplained errors
keys_res3 = keys_res2 + ["unexplained_by_shift_1", "unexplained_by_shift_2"]
accs3 = np.concatenate((np.where(counts2 > 5, accs2, 0), np.zeros((2, L))), axis=0)

for digit_index in range(L):
    key0 = "shift_a_0_shift_b_0_res"
    err_cnt = 0
    explained_shift_1 = 0
    explained_shift_2 = 0
    for i in range(n):
        if len(match_res[key0][i]) > digit_index and not match_res[key0][i][digit_index]:  # model makes a mistake at digit_index
            err_cnt += 1
            explained_shift_1 += any([match_res[key][i][digit_index] for key in keys_res if "0" in key and "1" in key])
            explained_shift_2 += any([match_res[key][i][digit_index] for key in keys_res])
    accs3[-2, digit_index] = (err_cnt - explained_shift_1) / (err_cnt + 1e-6)
    accs3[-1, digit_index] = (err_cnt - explained_shift_2) / (err_cnt + 1e-6)
    print(f"Error counts at digit index {digit_index}: {err_cnt}")

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.heatmap(accs3, ax=ax)
ax.set_yticklabels(keys_res3, rotation=0)
ax.set_title(f"Conditional addition accuracy with (left) shifted digits: {exp_a}-digit + {exp_b}-digit", weight="bold")
ax.set_xlabel(f"Digit index (0 means unit place)", weight="bold")
plt.savefig("final_analysis_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# Error pattern analysis
L = max(exp_a, exp_b) + 1
cnt_list = []
for digit_index in range(L):
    cnt = np.zeros(8)
    key0 = "shift_a_0_shift_b_0_res"
    for i in range(n):
        if len(match_res[key0][i]) > digit_index and not match_res[key0][i][digit_index]:  # model makes a mistake at digit_index
            ind1 = match_res["shift_a_1_shift_b_0_res"][i][digit_index]
            ind2 = match_res["shift_a_0_shift_b_1_res"][i][digit_index]
            ind3 = match_res["shift_a_1_shift_b_1_res"][i][digit_index]
            ind4 = match_res["shift_a_2_shift_b_0_res"][i][digit_index]
            ind5 = match_res["shift_a_0_shift_b_2_res"][i][digit_index]
            ind6 = match_res["shift_a_2_shift_b_1_res"][i][digit_index]
            ind7 = match_res["shift_a_1_shift_b_2_res"][i][digit_index]
            ind8 = match_res["shift_a_2_shift_b_2_res"][i][digit_index]
            cnt[0] += 1
            cnt[1] += ind1
            cnt[2] += ind2
            cnt[3] += ind3 and (not ind1) and (not ind2)
            cnt[4] += ind4 and (not ind1) and (not ind2)
            cnt[5] += ind5 and (not ind1) and (not ind2)
            cnt[6] += (ind3 or ind4 or ind5) and (not ind1) and (not ind2)
            cnt[7] += (ind6 or ind7 or ind8) and (not ind1) and (not ind2) and (not ind3) and (not ind4) and (not ind5)
    cnt_list.append(cnt)

for digit_index in range(L):
    print(f"Digit {digit_index}: {cnt_list[digit_index]}")

print(f"Shape of accs2: {accs2.shape}")
print("Analysis complete!")