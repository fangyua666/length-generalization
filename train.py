# Base model training
import math
import torch
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from utils import set_seeds
from model import GPT
from data import get_batch
import os
from evaluation import test_accuracy_on_digits
import argparse

def create_optimizer_and_scheduler(model, total_steps, warmup_steps=0, decay_steps=0):
    # AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,              
        betas=(0.9, 0.99),
        eps=1e-12,
        weight_decay=0.1
    )

    # Define stable steps
    stable_steps = total_steps - warmup_steps - decay_steps

    def lr_lambda(step):
        # Linear warmup from 0->1
        if step < warmup_steps:
            return step / warmup_steps
        # Stable at 1.0
        elif step < warmup_steps + stable_steps:
            return 1.0
        else:
            # Cosine decay from 1->0
            decay_ratio = (step - warmup_steps - stable_steps) / decay_steps
            return 0.5 * (1 + math.cos(math.pi * decay_ratio))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler

def estimate_loss(data, model, eval_iters=100, batch_size=1024, block_size=100, device='cuda'):

    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, block_size, device)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out['loss'] = losses.mean()
    model.train()
    return out

def train_base_model(
    vocab_size=14,
    block_size=100,
    n_embd=384,
    n_layer=6,
    n_head=6,
    dropout=0.0,
    bias=True,
    max_iters=10000, # change as needed
    eval_interval=100,
    data_path=None,
    models_dir=None,
    device='cuda',
    task='reverse_addition', # or 'copy' for copy task
):
    
    if task == 'copy':
        task_simplified = 'sc'
    elif task == 'reverse_addition':
        task_simplified = 'ra'
        
    
    model = GPT(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias=bias)
    model = model.to(device)
    
    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.readlines()
    
    optimizer, scheduler = create_optimizer_and_scheduler(model, max_iters, 1000, 2000) # 1000, 2000
    
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    loss_list = []
    
    scaler = GradScaler()
    for iter in tqdm(range(max_iters), desc="Training Progress"):
        # Sample a batch of data
        # Every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(data, model)['loss']
            print(f"step {iter}: loss {losses:.4f}")
            loss_list.append(round(losses.item(), 4))

        
        xb, yb = get_batch(data, device=device)
        
        # Evaluate the loss     
        with autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        
    # Evaluate final performance on digit addition
    acc = test_accuracy_on_digits(model, 11)
    print(f"Average accuracy: {acc}")
    
    # Save the model
    filename = f"{task_simplified}_model_0.pt"
    save_path = os.path.join(models_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Saved model at {save_path}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train a GPT model on arithmetic tasks')
    
    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=14, help='Vocabulary size')
    parser.add_argument('--block_size', type=int, default=100, help='Block size for model')
    parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--bias', action='store_true', help='Use bias in linear layers')
    
    # Training parameters
    parser.add_argument('--max_iters', type=int, default=10000, help='Maximum training iterations')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--task', type=str, default='reverse_addition', 
                       choices=['reverse_addition', 'copy'], help='Task to train on')
    
    # Data and model paths
    parser.add_argument('--data_path', type=str, default='length-generalization/data/origin_ds_reverse_addition.txt',
                       help='Path to training data')
    parser.add_argument('--models_dir', type=str, default='models', 
                       help='Directory to save models')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    set_seeds(args.seed)
    
    # Create models directory if it doesn't exist
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Train the model
    model = train_base_model(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
        bias=args.bias,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        data_path=args.data_path,
        models_dir=args.models_dir,
        device=args.device,
        task=args.task
    )

if __name__ == "__main__":
    main()
 