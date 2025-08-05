# src/evaluation.py
import torch
import numpy as np
from data import generate
from data import encode
from model import GPT
import random
import os
from data import load_baseline_problems, save_baseline_problems, save_modified_problems, load_modified_problems
from utils import set_seeds

def accuracy_print_one(model, num_digits, need_print=False, device='cuda', block_size=100, batch_size=1024, save_to_file=None):
    correct = 0
    total = 1024
    num_batches = total // batch_size
    
    # Open file if save_to_file is specified
    if save_to_file:
        f = open(save_to_file, 'w')
    
    for _ in range(num_batches):
        exp = num_digits
        # print(exp)
        a_list = [random.randint(10**(exp-1), 10**(exp)-1) for _ in range(batch_size)]
        b_list = [random.randint(10**(exp-1), 10**(exp)-1) for _ in range(batch_size)]
        # prompt_str = [f"{str(i)[::-1]}+{str(j)[::-1]}=" for i, j in zip(a, b)]
        prompt_str = [f"{str(i)[::-1]}+{str(j)[::-1]}=" for i, j in zip(a_list, b_list)]
        # print(prompt_str)
        
        context = torch.tensor([encode(inp) for inp in prompt_str], dtype=torch.long, device=device)
        
        # output in batch
        output_batch = generate(model=model, idx=context, max_new_tokens=block_size, top_k=1)
        
        answers = [str(i + j)[::-1] for i, j in zip(a_list, b_list)]
        targets = [p + ans for p, ans in zip(prompt_str, answers)]
        
        correct += sum([output == target for output, target in zip(output_batch, targets)])
        
        # if needed, print wrong answer
        if need_print:
            for inp, out, target in zip(prompt_str, output_batch, targets):
                if out != target:
                    print(f"   Input: {inp}")
                    print(f"  Output: {out}")
                    print(f"Expected: {target}")
                    print("-----------")
                    
                    # Also write to file if specified
                    if save_to_file:
                        f.write(f"   Input: {inp}\n")
                        f.write(f"  Output: {out}\n")
                        f.write(f"Expected: {target}\n")
                        f.write("-----------\n")
    
    acc = correct / total
    print(f"Accuracy for {num_digits} digits: {acc}")
    
    if save_to_file:
        f.write(f"Accuracy for {num_digits} digits: {acc}\n")
        f.close()
        print(f"Output saved to: {save_to_file}")
    
    return acc

def get_avg_performance(model, num_digits):

    dict_acc = {}
    for num_dig in range(1, num_digits+1):
        dict_acc[num_dig] = accuracy_print_one(model, num_dig, need_print=False)
    return dict_acc

def test_accuracy_on_digits(model, digits):
    acc_list = []
    for i in range(10):
        acc_list.append(accuracy_print_one(model, digits, need_print=False))
    return sum(acc_list)/len(acc_list)

def get_model_responses(model, problems, max_new_tokens=100, batch_size=1024):
    responses = []


    batch_size_local = batch_size

    for i in range(0, len(problems), batch_size_local):
        batch_problems = problems[i:i + batch_size_local]

        context = torch.tensor([encode(problem) for problem in batch_problems],
                             dtype=torch.long, device='cuda')

        batch_responses = generate(model=model, idx=context,
                                 max_new_tokens=max_new_tokens, top_k=1)

        responses.extend(batch_responses)

    return responses

def save_model_responses(responses, filename_prefix):
    
    responses_filename = f"{filename_prefix}_responses.txt"
    with open(responses_filename, 'w') as f:
        for response in responses:
            f.write(response + '\n')
    
    print(f"save to {responses_filename}")

    return responses_filename

if __name__ == "__main__":
    set_seeds(42)
    model = GPT(vocab_size=14, block_size=100, n_embd=384, n_layer=6, n_head=6, dropout=0.0, bias=True).to('cuda')
    ckpt = f"/workspace/length-generalization/models/ra_model_0.pt"
    model.load_state_dict(torch.load(ckpt, map_location='cuda'))
    accuracy_print_one(model, 12)