
import os
import time
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def medare(y_true, y_pred):
    eps = 1e-8
    return np.median(np.abs((y_true - y_pred) / (y_true + eps))) * 100

@torch.no_grad()
def measure_end_to_end_latency(model, device, dataset, warmup=5, runs=30):
    model.eval()
    for _ in range(warmup):
        idx = np.random.randint(0, len(dataset))
        image, patch_feats, _ = dataset[idx]
        x = image.unsqueeze(0).to(device, non_blocking=True)
        pf = patch_feats.unsqueeze(0).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            _ = model(x, pf)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        idx = np.random.randint(0, len(dataset))
        image, patch_feats, _ = dataset[idx]
        x = image.unsqueeze(0).to(device, non_blocking=True)
        pf = patch_feats.unsqueeze(0).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            _ = model(x, pf)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / runs * 1000.0

def train_model(model, output_path, train_loader, valid_loader, test_loader, num_epochs, optimizer, criterion,
                scheduler=None, patience=50):
    train_losses, valid_losses, valid_r2, epoch_times = [], [], [], []
    best_r2 = -float('inf')
    epochs_no_improve = 0
    scaler = torch.cuda.amp.GradScaler()
    best_model_path = os.path.join(output_path, 'Best.pth')

    os.makedirs(output_path, exist_ok=True)
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        losses = []

        for inputs, patch_feats, targets in tqdm(train_loader, desc=f"Train {epoch + 1}/{num_epochs}"):
            inputs = inputs.to(device)
            patch_feats = patch_feats.to(device)
            targets = targets.type(torch.float).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs, patch_feats).squeeze()
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.detach().cpu())

        train_losses.append(float(torch.stack(losses).mean()))

        model.eval()
        val_losses, labels, preds = [], [], []
        with torch.no_grad():
            for inputs, patch_feats, targets in tqdm(valid_loader, desc=f"Valid {epoch + 1}/{num_epochs}"):
                inputs = inputs.to(device)
                patch_feats = patch_feats.to(device)
                targets = targets.type(torch.float).to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(inputs, patch_feats).squeeze()
                    loss = criterion(outputs, targets)
                val_losses.append(loss.detach().cpu())
                labels.append(targets.detach().cpu())
                preds.append(outputs.detach().cpu())

        valid_losses.append(float(torch.stack(val_losses).mean()))
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        r2 = r2_score(labels.numpy(), preds.numpy())
        valid_r2.append(r2)

        end_time = time.time()
        epoch_times.append(end_time - start_time)

        print(f"Epoch {epoch + 1} | Train Loss: {train_losses[-1]:.6f} | "
              f"Val Loss: {valid_losses[-1]:.6f} | R2: {r2:.4f} | Best: {best_r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs.")
                break

        if scheduler is not None:
            scheduler.step(r2)

    df_log = pd.DataFrame({
        'train_loss': train_losses,
        'val_loss': valid_losses,
        'val_r2': valid_r2,
        'epoch_times': epoch_times
    })
    df_log.to_csv(os.path.join(output_path, 'PatchPhysics_log.csv'), index=False)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_labels, test_preds = [], []
    with torch.no_grad():
        for inputs, patch_feats, targets in tqdm(test_loader, desc="[Test]"):
            inputs = inputs.to(device)
            patch_feats = patch_feats.to(device)
            targets = targets.type(torch.float).to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs, patch_feats).squeeze()
            test_labels.append(targets.detach().cpu())
            test_preds.append(outputs.detach().cpu())

    test_labels = torch.cat(test_labels, dim=0)
    test_preds = torch.cat(test_preds, dim=0)

    test_labels_exp = np.expm1(test_labels.numpy())
    test_preds_exp = np.expm1(test_preds.numpy())

    test_r2_exp = r2_score(test_labels_exp, test_preds_exp)
    test_rmse = np.sqrt(mean_squared_error(test_labels_exp, test_preds_exp))
    test_medare = medare(test_labels_exp, test_preds_exp)
    test_mse = mean_squared_error(test_labels_exp, test_preds_exp)

    print(f"[Test MSE]: {test_mse:.4f}")
    print(f"[Test R2]: {test_r2_exp:.4f}")
    print(f"[Test RMSE]: {test_rmse:.4f}")
    print(f"[Test MedARE]: {test_medare:.2f}%")

    with open(os.path.join(output_path, 'test_metrics.txt'), 'w') as f:
        f.write(f"Test Loss (MSE): {test_mse:.6f}\n")
        f.write(f"Test R2: {test_r2_exp:.6f}\n")
        f.write(f"Test RMSE: {test_rmse:.6f}\n")
        f.write(f"Test MedARE: {test_medare:.4f}\n")

    return test_r2_exp, test_rmse, test_medare, test_mse
