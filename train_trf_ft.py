import numpy as np
from model_trf import TRFNet
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
def correlation_loss(pred, target, eps=1e-8):
    """
    pred:   (B, T)
    target: (B, T)
    """
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    num = (pred * target).sum(dim=1)
    den = torch.sqrt((pred ** 2).sum(dim=1) * (target ** 2).sum(dim=1) + eps)

    corr = num / den
    return 1.0 - corr.mean()


def trf_smoothness_loss(trf):
    """
    trf: (B, 30, L)
    penalize temporal roughness
    """
    return ((trf[:, :, 1:] - trf[:, :, :-1]) ** 2).mean()

def global_residual_trf_loss(
    y_hat,
    y_true,
    trf_residual,
    trf_final,
    alpha_corr=0.5,
    alpha_res=1e-4,
    alpha_smooth=1e-4,
):
    loss_mse = F.mse_loss(y_hat, y_true)
    loss_corr = correlation_loss(y_hat, y_true)
    loss_res = (trf_residual ** 2).mean()
    loss_smooth = trf_smoothness_loss(trf_final)

    total_loss = (
        loss_mse
        + alpha_corr * loss_corr
        + alpha_res * loss_res
        + alpha_smooth * loss_smooth
    )

    return total_loss, {
        "loss_mse": loss_mse.item(),
        "loss_corr": loss_corr.item(),
        "loss_res": loss_res.item(),
        "loss_smooth": loss_smooth.item(),
    }

def reconstruct_envelope_from_trf(x, trf):
    B, C, T = x.shape
    _, _, L = trf.shape

    device = x.device
    y_hat_list = []

    for b in range(B):
        xb = x[b:b+1].to(device)     # (1, C, T)
        hb = trf[b].to(device)       # (C, L)

        hb = torch.flip(hb, dims=[-1])
        hb = hb.unsqueeze(1)         # (C, 1, L)

        yb = F.conv1d(
            xb,
            hb,
            bias=None,
            stride=1,
            padding=L - 1,
            groups=C
        )

        yb = yb.sum(dim=1)

        start = L - 1
        end = start + T
        yb = yb[:, start:end]

        y_hat_list.append(yb)

    y_hat = torch.cat(y_hat_list, dim=0)
    return y_hat

class EEGTrialDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]

        x = np.asarray(sample[0], dtype=np.float32)   # (30, T)
        y = np.asarray(sample[1], dtype=np.float32)   # (T,)
        label =  np.asarray(sample[2], dtype=np.float32)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        label = torch.from_numpy(label)

        # if np.isscalar(label):
        #     label = torch.tensor(label, dtype=torch.long)
        # else:
        #     label = np.asarray(label)
        #     if label.ndim == 0:
        #         label = torch.tensor(label.item(), dtype=torch.long)
        #     else:
        #         label = torch.tensor(label, dtype=torch.float32)

        return x, y, label

def eeg_pad_collate(batch):
    xs, ys, labels = zip(*batch)

    lengths = [x.shape[1] for x in xs]
    max_len = max(lengths)

    padded_xs = []
    padded_ys = []
    masks = []

    for x, y in zip(xs, ys):
        T = x.shape[1]
        pad_len = max_len - T

        x_pad = F.pad(x, (0, pad_len), value=0.0)   # (30, max_len)
        y_pad = F.pad(y, (0, pad_len), value=0.0)   # (max_len,)

        mask = torch.zeros(max_len, dtype=torch.float32)
        mask[:T] = 1.0

        padded_xs.append(x_pad)
        padded_ys.append(y_pad)
        masks.append(mask)

    batch_x = torch.stack(padded_xs, dim=0)   # (B, 30, max_len)
    batch_y = torch.stack(padded_ys, dim=0)   # (B, max_len)
    batch_mask = torch.stack(masks, dim=0)    # (B, max_len)

    # label
    if torch.is_tensor(labels[0]):
        batch_label = torch.stack(labels, dim=0)
    else:
        batch_label = torch.tensor(labels)

    batch_lengths = torch.tensor(lengths, dtype=torch.long)

    return batch_x, batch_y, batch_label, batch_mask, batch_lengths

def eeg_pad_collate(batch):
    xs, ys, labels = zip(*batch)

    lengths = [x.shape[1] for x in xs]
    max_len = max(lengths)

    padded_xs = []
    padded_ys = []
    masks = []

    for x, y in zip(xs, ys):
        T = x.shape[1]
        pad_len = max_len - T

        x_pad = F.pad(x, (0, pad_len), value=0.0)   # (30, max_len)
        y_pad = F.pad(y, (0, pad_len), value=0.0)   # (max_len,)

        mask = torch.zeros(max_len, dtype=torch.float32)
        mask[:T] = 1.0

        padded_xs.append(x_pad)
        padded_ys.append(y_pad)
        masks.append(mask)

    batch_x = torch.stack(padded_xs, dim=0)      # (B, 30, max_len)
    batch_y = torch.stack(padded_ys, dim=0)      # (B, max_len)
    batch_label = torch.stack(labels, dim=0)     # (B, 2)
    batch_mask = torch.stack(masks, dim=0)       # (B, max_len)
    batch_lengths = torch.tensor(lengths, dtype=torch.long)

    return batch_x, batch_y, batch_label, batch_mask, batch_lengths

def masked_mse_loss(pred, target, mask):
    """
    pred:   (B, T)
    target: (B, T)
    mask:   (B, T), valid=1, padded=0
    """
    loss = (pred - target) ** 2
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)

def masked_correlation_loss(pred, target, mask, eps=1e-8):
    """
    pred:   (B, T)
    target: (B, T)
    mask:   (B, T)
    """
    pred = pred * mask
    target = target * mask

    valid_len = mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    pred_mean = pred.sum(dim=1, keepdim=True) / valid_len
    target_mean = target.sum(dim=1, keepdim=True) / valid_len

    pred_centered = (pred - pred_mean) * mask
    target_centered = (target - target_mean) * mask

    num = (pred_centered * target_centered).sum(dim=1)
    den = torch.sqrt(
        (pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1) + eps
    )

    corr = num / den
    return 1.0 - corr.mean()

def trf_smoothness_loss(trf):
    """
    trf: (B, C, L)
    """
    return ((trf[:, :, 1:] - trf[:, :, :-1]) ** 2).mean()


def total_trf_loss(
    y_hat,
    y_true,
    mask,
    trf_residual,
    trf_final,
    alpha_corr=0.5,
    alpha_res=1e-4,
    alpha_smooth=1e-4,
):
    loss_mse = masked_mse_loss(y_hat, y_true, mask)
    loss_corr = masked_correlation_loss(y_hat, y_true, mask)
    loss_res = (trf_residual ** 2).mean()
    loss_smooth = trf_smoothness_loss(trf_final)

    total = (
        loss_mse
        + alpha_corr * loss_corr
        + alpha_res * loss_res
        + alpha_smooth * loss_smooth
    )

    return total, {
        "mse": loss_mse.item(),
        "corr": loss_corr.item(),
        "res": loss_res.item(),
        "smooth": loss_smooth.item(),
    }

def build_trial_list(subject_dataset_list):
    trial_list = []

    for subj_data in subject_dataset_list:
        x_trials = subj_data[0]   # list / object array of trials
        y_trials = subj_data[1]
        labels = subj_data[2]

        n_trials = len(x_trials)

        for i in range(n_trials):
            x_i = x_trials[i]     # (30, T_i)
            y_i = y_trials[i]     # (T_i,)
            label_i = labels[i]   # scalar

            trial_list.append((x_i, y_i, label_i))

    return trial_list

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    alpha_corr=0.5,
    alpha_res=1e-4,
    alpha_smooth=1e-4,
):
    model.train()

    running = {
        "total": 0.0,
        "mse": 0.0,
        "corr": 0.0,
        "res": 0.0,
        "smooth": 0.0,
    }
    n_batches = 0

    for batch in train_loader:
        if len(batch) >= 5:
            batch_x = batch[0]
            batch_y = batch[1]
            batch_label = batch[2]
            batch_mask = batch[3]
        else:
            batch_x = batch[0]
            batch_y = batch[1]
            batch_mask = batch[2]

        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()
        batch_mask = batch_mask.to(device).float()
        batch_label = batch_label.to(device).long() 

        optimizer.zero_grad()

        trf_global, trf_residual, trf_final = model(batch_x, context=batch_label)
        y_hat = reconstruct_envelope_from_trf(batch_x, trf_final)

        total_loss, loss_dict = total_trf_loss(
            y_hat=y_hat,
            y_true=batch_y,
            mask=batch_mask,
            trf_residual=trf_residual,
            trf_final=trf_final,
            alpha_corr=alpha_corr,
            alpha_res=alpha_res,
            alpha_smooth=alpha_smooth,
        )

        total_loss.backward()
        optimizer.step()

        running["total"] += total_loss.item()
        running["mse"] += loss_dict["mse"]
        running["corr"] += loss_dict["corr"]
        running["res"] += loss_dict["res"]
        running["smooth"] += loss_dict["smooth"]
        n_batches += 1

    for k in running:
        running[k] /= max(n_batches, 1)

    return running

@torch.no_grad()
def validate_one_epoch(
    model,
    val_loader,
    device,
    alpha_corr=0.5,
    alpha_res=1e-4,
    alpha_smooth=1e-4,
):
    model.eval()

    running = {
        "total": 0.0,
        "mse": 0.0,
        "corr": 0.0,
        "res": 0.0,
        "smooth": 0.0,
    }
    n_batches = 0

    for batch in val_loader:
        if len(batch) >= 5:
            batch_x = batch[0]
            batch_y = batch[1]
            batch_label = batch[2]
            batch_mask = batch[3]
        else:
            batch_x = batch[0]
            batch_y = batch[1]
            batch_mask = batch[2]

        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()
        batch_mask = batch_mask.to(device).float()
        batch_label = batch_label.to(device).long() 
        batch_context = F.one_hot(batch_label, num_classes=2).float() 

        trf_global, trf_residual, trf_final = model(batch_x, context=batch_label)
        y_hat = reconstruct_envelope_from_trf(batch_x, trf_final)

        total_loss, loss_dict = total_trf_loss(
            y_hat=y_hat,
            y_true=batch_y,
            mask=batch_mask,
            trf_residual=trf_residual,
            trf_final=trf_final,
            alpha_corr=alpha_corr,
            alpha_res=alpha_res,
            alpha_smooth=alpha_smooth,
        )

        running["total"] += total_loss.item()
        running["mse"] += loss_dict["mse"]
        running["corr"] += loss_dict["corr"]
        running["res"] += loss_dict["res"]
        running["smooth"] += loss_dict["smooth"]
        n_batches += 1

    for k in running:
        running[k] /= max(n_batches, 1)

    return running

def fit_trf_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    lr=1e-3,
    weight_decay=1e-5,
    device=None,
    alpha_corr=0.5,
    alpha_res=1e-4,
    alpha_smooth=1e-4,
    save_dir="./outputs_trf",
    save_name="trfnet_best.pt",
    hist_name="trfnet_hist.npy",
):
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    hist = {
        "train_total": [],
        "train_mse": [],
        "train_corr": [],
        "train_res": [],
        "train_smooth": [],
        "val_total": [],
        "val_mse": [],
        "val_corr": [],
        "val_res": [],
        "val_smooth": [],
        "best_epoch": None,
        "best_val_total": None,
    }

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        train_log = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha_corr=alpha_corr,
            alpha_res=alpha_res,
            alpha_smooth=alpha_smooth,
        )

        val_log = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            alpha_corr=alpha_corr,
            alpha_res=alpha_res,
            alpha_smooth=alpha_smooth,
        )

        hist["train_total"].append(train_log["total"])
        hist["train_mse"].append(train_log["mse"])
        hist["train_corr"].append(train_log["corr"])
        hist["train_res"].append(train_log["res"])
        hist["train_smooth"].append(train_log["smooth"])

        hist["val_total"].append(val_log["total"])
        hist["val_mse"].append(val_log["mse"])
        hist["val_corr"].append(val_log["corr"])
        hist["val_res"].append(val_log["res"])
        hist["val_smooth"].append(val_log["smooth"])

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train total={train_log['total']:.4f}, mse={train_log['mse']:.4f}, "
            f"corr={train_log['corr']:.4f}, res={train_log['res']:.6f}, smooth={train_log['smooth']:.6f} | "
            f"Val total={val_log['total']:.4f}, mse={val_log['mse']:.4f}, "
            f"corr={val_log['corr']:.4f}, res={val_log['res']:.6f}, smooth={val_log['smooth']:.6f}"
        )

        if val_log["total"] < best_val:
            best_val = val_log["total"]
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_total": best_val,
                },
                os.path.join(save_dir, save_name),
            )
            print(f"  -> Best model saved at epoch {best_epoch}, val_total={best_val:.4f}")

        hist["best_epoch"] = best_epoch
        hist["best_val_total"] = best_val

        np.save(os.path.join(save_dir, hist_name), hist)

    print(f"Training finished. Best epoch={best_epoch}, best val_total={best_val:.4f}")
    print(f"History saved to {os.path.join(save_dir, hist_name)}")

    return hist

def fit_trf_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    lr=1e-3,
    weight_decay=1e-5,
    device=None,
    alpha_corr=0.5,
    alpha_res=1e-4,
    alpha_smooth=1e-4,
    save_dir="./outputs_trf",
    save_name="trfnet_best.pt",
    hist_name="trfnet_hist.npy",
):
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    hist = {
        "train_total": [],
        "train_mse": [],
        "train_corr": [],
        "train_res": [],
        "train_smooth": [],
        "val_total": [],
        "val_mse": [],
        "val_corr": [],
        "val_res": [],
        "val_smooth": [],
        "best_epoch": None,
        "best_val_total": None,
    }

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        train_log = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha_corr=alpha_corr,
            alpha_res=alpha_res,
            alpha_smooth=alpha_smooth,
        )

        val_log = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            alpha_corr=alpha_corr,
            alpha_res=alpha_res,
            alpha_smooth=alpha_smooth,
        )

        hist["train_total"].append(train_log["total"])
        hist["train_mse"].append(train_log["mse"])
        hist["train_corr"].append(train_log["corr"])
        hist["train_res"].append(train_log["res"])
        hist["train_smooth"].append(train_log["smooth"])

        hist["val_total"].append(val_log["total"])
        hist["val_mse"].append(val_log["mse"])
        hist["val_corr"].append(val_log["corr"])
        hist["val_res"].append(val_log["res"])
        hist["val_smooth"].append(val_log["smooth"])

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train total={train_log['total']:.4f}, mse={train_log['mse']:.4f}, "
            f"corr={train_log['corr']:.4f}, res={train_log['res']:.6f}, smooth={train_log['smooth']:.6f} | "
            f"Val total={val_log['total']:.4f}, mse={val_log['mse']:.4f}, "
            f"corr={val_log['corr']:.4f}, res={val_log['res']:.6f}, smooth={val_log['smooth']:.6f}"
        )

        if val_log["total"] < best_val:
            best_val = val_log["total"]
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_total": best_val,
                },
                os.path.join(save_dir, save_name),
            )
            print(f"  -> Best model saved at epoch {best_epoch}, val_total={best_val:.4f}")

        hist["best_epoch"] = best_epoch
        hist["best_val_total"] = best_val

        np.save(os.path.join(save_dir, hist_name), hist)

    print(f"Training finished. Best epoch={best_epoch}, best val_total={best_val:.4f}")
    print(f"History saved to {os.path.join(save_dir, hist_name)}")

    return hist

if __name__ == '__main__':
    path='/workspace/535505/'
    sid=['training-1','training-2','training-3','training-4','training-5']
    train_dataset = [np.load('%s/data/%s_dataset_250.npy'%(path,i), allow_pickle=True) for i in sid]
    sid_val=['testing']
    val_dataset = [np.load('%s/data/%s_dataset_250.npy'%(path,i), allow_pickle=True) for i in sid_val]
    train_trials = build_trial_list(train_dataset)
    val_trials = build_trial_list(val_dataset)
    train_dataset = EEGTrialDataset(train_trials)
    val_dataset = EEGTrialDataset(val_trials)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=eeg_pad_collate,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=eeg_pad_collate,
        num_workers=0
    )

    save_dir = '/workspace/535505/save_model/trf_2_250'
    os.makedirs(save_dir, exist_ok=True)
    num_epochs = 500
    device = torch.device("cuda")
    model =TRFNet(
        in_channels=30,
        trf_len=125,
        hidden_dim=64,
        latent_dim=128,
        context_dim=0,
        dropout=0.2,
        residual_scale=0.1,
    )

    hist = fit_trf_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=1e-4,
        weight_decay=1e-5,
        device=device,
        alpha_corr=0.5,
        alpha_res=1e-4,
        alpha_smooth=1e-4,
        save_dir= save_dir,
        save_name="trfnet_best.pt",
        hist_name="trfnet_hist.npy",
    )
    