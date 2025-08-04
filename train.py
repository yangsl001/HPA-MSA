# train_bert.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random
import os
import time
from tqdm import tqdm

from src.config import get_config
from src.utils.metrics import calculate_metrics_reg as calculate_metrics
from src.utils.loss import HPAMSALoss
from src.datasets_bert import get_data_loaders_bert
from src.models.hpa_msa_bert import HPAMSA_BERT


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Counts the number of total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def train_epoch(model, data_loader, criterion, optimizer, config, scheduler):
    model.train()
    epoch_losses = {
        'total': [], 'task': [], 'disentangle': [],
        'regression': [], 'classification': [], 'ortho': [], 'mi': []
    }

    for i, batch in enumerate(tqdm(data_loader, desc="Training (BERT Fine-tuning)")):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        token_type_ids = batch['token_type_ids'].to(config.device)
        vision = batch['vision'].to(config.device)
        audio = batch['audio'].to(config.device)
        labels = {k: v.to(config.device) for k, v in batch['labels'].items()}

        optimizer.zero_grad()
        model_output = model(input_ids, attention_mask, token_type_ids, vision, audio)

        loss_dict = criterion(model_output, labels)
        loss = loss_dict['total']

        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        # Per-epoch scheduler step is generally more stable for CosineAnnealingLR
        # scheduler.step() # Uncomment for per-batch step

        for key in epoch_losses:
            if key in loss_dict:
                epoch_losses[key].append(loss_dict[key].item())

    # After the loop, the scheduler is stepped in the main function.

    return {key: np.mean(val) for key, val in epoch_losses.items() if val}


def evaluate(model, data_loader, criterion, config):
    model.eval()
    epoch_losses = {
        'total': [], 'task': [], 'disentangle': [],
        'regression': [], 'classification': [], 'ortho': [], 'mi': []
    }

    all_preds_reg, all_truths_reg = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating on {data_loader.dataset.mode}"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            token_type_ids = batch['token_type_ids'].to(config.device)
            vision = batch['vision'].to(config.device)
            audio = batch['audio'].to(config.device)
            labels = {k: v.to(config.device) for k, v in batch['labels'].items()}

            model_output = model(input_ids, attention_mask, token_type_ids, vision, audio)
            loss_dict = criterion(model_output, labels)

            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key].append(loss_dict[key].item())

            all_preds_reg.extend(model_output['pred_reg'].cpu().numpy())
            all_truths_reg.extend(labels['regression'].cpu().numpy())

    avg_losses = {key: np.mean(val) for key, val in epoch_losses.items() if val}
    metrics = calculate_metrics(np.array(all_preds_reg), np.array(all_truths_reg))

    return avg_losses, metrics


def main():
    config = get_config()
    set_seed(config.seed)

    run_name = f"HPAMSA-BERT_{config.dataset_name}_{int(time.time())}"
    checkpoint_path = os.path.join(config.checkpoints_path, f"{run_name}_best.pth")
    os.makedirs(config.checkpoints_path, exist_ok=True)

    print("--- HPA-MSA with BERT Fine-tuning Configuration ---")
    for key, val in config.items(): print(f"{key:<30}: {val}")
    print("---------------------------------------------------\n")

    train_loader, valid_loader, test_loader = get_data_loaders_bert(config)

    model = HPAMSA_BERT(config).to(config.device)
    criterion = HPAMSALoss(config).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=0)

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params / 1e6:.2f}M, Trainable: {trainable_params / 1e6:.2f}M")
    print(f"Note: BERT parameters are included in the count.\n")

    best_valid_f1 = -1.0

    print("--- Starting Training (BERT Fine-tuning) ---")
    for epoch in range(1, config.num_epochs + 1):
        start_time = time.time()

        train_losses = train_epoch(model, train_loader, criterion, optimizer, config, scheduler)

        # --- START OF MODIFICATION ---
        # The incorrect variable 'data_loader' has been replaced with 'valid_loader'
        valid_losses, valid_metrics = evaluate(model, valid_loader, criterion, config)
        # --- END OF MODIFICATION ---

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        epoch_duration = time.time() - start_time

        print(f"\n--- Epoch {epoch}/{config.num_epochs} | Duration: {epoch_duration:.2f}s | LR: {current_lr:.8f} ---")
        print("Train Loss:")
        for key, val in train_losses.items(): print(f"  {key:<15}: {val:.4f}", end=" |")
        print("\nValidation Loss:")
        for key, val in valid_losses.items(): print(f"  {key:<15}: {val:.4f}", end=" |")
        print("\nValidation Metrics:")
        for key, val in valid_metrics.items(): print(f"  {key:<10}: {val:.4f}", end=" |")
        print("\n--------------------------------------------------")

        current_valid_f1 = valid_metrics['F1-Score']
        if current_valid_f1 > best_valid_f1:
            best_valid_f1 = current_valid_f1
            print(f"🎉 New best model found! F1-Score: {best_valid_f1:.4f}. Saving to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)

    print("\n--- Training Complete. Loading best model for final testing. ---")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        test_losses, test_metrics = evaluate(model, test_loader, criterion, config)

        print("\n--- Final Test Results ---")
        print("Test Loss:")
        for key, val in test_losses.items(): print(f"  {key:<15}: {val:.4f}", end=" |")
        print("\nTest Metrics:")
        for key, val in test_metrics.items(): print(f"  {key:<10}: {val:.4f}", end=" |")
        print("\n--------------------------")
    else:
        print("No best model was saved during training. Skipping final test.")


if __name__ == '__main__':
    main()