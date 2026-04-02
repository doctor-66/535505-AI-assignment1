import numpy as np
from model_trf import *
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset

def trf_smoothness_loss(trf):
    """
    trf: (B, 30, L)
    penalize temporal roughness
    """
    return ((trf[:, :, 1:] - trf[:, :, :-1]) ** 2).mean()

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

def multitask_loss(y_hat, y_true, logits, labels, alpha_rec=1.0, alpha_corr=0.5, alpha_cls=1.0):
    """
    y_hat:   (B, 1000)
    y_true:  (B, 1000)
    logits:  (B, 2)
    labels:  (B,)
    """
    loss_mse = F.mse_loss(y_hat, y_true)
    loss_corr = correlation_loss(y_hat, y_true)
    # loss_cls = F.cross_entropy(logits, labels)

    # total_loss = alpha_rec * loss_mse + alpha_corr * loss_corr + alpha_cls * loss_cls
    total_loss = loss_mse + alpha_corr * loss_corr 
    return total_loss, loss_mse, loss_corr#, loss_cls
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


def build_trial_list(subject_dataset_list):
    """
    subject_dataset_list:
        [
            np.array([x_trials, y_trials, labels], dtype=object),   # one subject
            ...
        ]

    return:
        trial_list = [
            (x_trial, y_trial, label_trial),
            ...
        ]
    """
    trial_list = []

    for subj_data in subject_dataset_list:
        x_trials = subj_data[0]
        y_trials = subj_data[1]
        labels = subj_data[2]

        n_trials = len(x_trials)

        for i in range(n_trials):
            x_i = x_trials[i]     # (30, T_i)
            y_i = y_trials[i]     # (T_i,)
            label_i = labels[i]   # e.g. (2,) one-hot

            trial_list.append((x_i, y_i, label_i))

    return trial_list

def segment_trial_to_windows(x, y, label, window_size=1024, stride=1024):
    """
    x: (30, T)
    y: (T,)
    label: (2,) one-hot or scalar

    return:
        Xw: (Nw, 30, window_size)
        Yw: (Nw, window_size)
        Lw: (Nw, 2) or (Nw,)
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    label = np.asarray(label, dtype=np.float32)

    T = x.shape[1]
    X_list, Y_list, L_list = [], [], []

    if T < window_size:
        return None, None, None

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        X_list.append(x[:, start:end])   # (30, window_size)
        Y_list.append(y[start:end])      # (window_size,)
        L_list.append(label.copy())      # (2,) or scalar

    Xw = np.stack(X_list, axis=0)
    Yw = np.stack(Y_list, axis=0)

    if label.ndim == 0:
        Lw = np.array(L_list, dtype=np.int64)
    else:
        Lw = np.stack(L_list, axis=0).astype(np.float32)

    return Xw, Yw, Lw

def build_window_dataset(trial_list, window_size=1000, stride=1000):
    all_x, all_y, all_label = [], [], []

    for x, y, label in trial_list:
        Xw, Yw, Lw = segment_trial_to_windows(
            x, y, label,
            window_size=window_size,
            stride=stride
        )

        if Xw is None:
            continue

        all_x.append(Xw)
        all_y.append(Yw)
        all_label.append(Lw)

    X = np.concatenate(all_x, axis=0)   # (N, 30, window_size)
    Y = np.concatenate(all_y, axis=0)   # (N, window_size)
    L = np.concatenate(all_label, axis=0)

    return X, Y, L

def build_window_dataset(trial_list, window_size=1000, stride=1000):
    all_x, all_y, all_label = [], [], []

    for x, y, label in trial_list:
        Xw, Yw, Lw = segment_trial_to_windows(
            x, y, label,
            window_size=window_size,
            stride=stride
        )

        if Xw is None:
            continue

        all_x.append(Xw)
        all_y.append(Yw)
        all_label.append(Lw)

    X = np.concatenate(all_x, axis=0)   # (N, 30, window_size)
    Y = np.concatenate(all_y, axis=0)   # (N, window_size)
    L = np.concatenate(all_label, axis=0)

    return X, Y, L

def train_one_epoch_window(
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

    for batch_x, batch_y, batch_label in train_loader:
        batch_x = batch_x.to(device).float()      # (B, 30, W)
        batch_y = batch_y.to(device).float()      # (B, W)
        batch_label = batch_label.to(device).float()  # (B,2) if one-hot

        if model.context_dim == 0:
            batch_context = None
        elif model.context_dim == 2:
            batch_context = batch_label
        else:
            raise ValueError(f"Unsupported context_dim={model.context_dim}")

        optimizer.zero_grad()

        trf_global, trf_residual, trf_final = model(batch_x, context=batch_context)
        y_hat = reconstruct_envelope_from_trf(batch_x, trf_final)

        total_loss, loss_dict = global_residual_trf_loss(
            y_hat=y_hat,
            y_true=batch_y,
            trf_residual=trf_residual,
            trf_final=trf_final,
            alpha_corr=alpha_corr,
            alpha_res=alpha_res,
            alpha_smooth=alpha_smooth,
        )

        total_loss.backward()
        optimizer.step()

        running["total"] += total_loss.item()
        running["mse"] += loss_dict["loss_mse"]
        running["corr"] += loss_dict["loss_corr"]
        running["res"] += loss_dict["loss_res"]
        running["smooth"] += loss_dict["loss_smooth"]
        n_batches += 1

    for k in running:
        running[k] /= max(n_batches, 1)

    return running

@torch.no_grad()
def validate_one_epoch_window(
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

    for batch_x, batch_y, batch_label in val_loader:
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()
        batch_label = batch_label.to(device).float()

        if model.context_dim == 0:
            batch_context = None
        elif model.context_dim == 2:
            batch_context = batch_label
        else:
            raise ValueError(f"Unsupported context_dim={model.context_dim}")

        trf_global, trf_residual, trf_final = model(batch_x, context=batch_context)
        y_hat = reconstruct_envelope_from_trf(batch_x, trf_final)

        total_loss, loss_dict = global_residual_trf_loss(
            y_hat=y_hat,
            y_true=batch_y,
            trf_residual=trf_residual,
            trf_final=trf_final,
            alpha_corr=alpha_corr,
            alpha_res=alpha_res,
            alpha_smooth=alpha_smooth,
        )

        running["total"] += total_loss.item()
        running["mse"] += loss_dict["loss_mse"]
        running["corr"] += loss_dict["loss_corr"]
        running["res"] += loss_dict["loss_res"]
        running["smooth"] += loss_dict["loss_smooth"]
        n_batches += 1

    for k in running:
        running[k] /= max(n_batches, 1)

    return running

def fit_trf_model_window(
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
    save_dir="./outputs_trf_window",
    save_name="trfnet_window_best.pt",
    hist_name="trfnet_window_hist.npy",
):
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
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
        train_log = train_one_epoch_window(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha_corr=alpha_corr,
            alpha_res=alpha_res,
            alpha_smooth=alpha_smooth,
        )

        val_log = validate_one_epoch_window(
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

class EEGWindowDataset(Dataset):
    def __init__(self, X, Y, L):
        self.X = torch.tensor(X, dtype=torch.float32)   # (N, 30, W)
        self.Y = torch.tensor(Y, dtype=torch.float32)   # (N, W)

        # one-hot label: (N,2)
        if len(L.shape) == 2:
            self.L = torch.tensor(L, dtype=torch.float32)
        else:
            self.L = torch.tensor(L, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.L[idx]

if __name__ == '__main__':
    path='/workspace/535505/'
    sid=['training-1','training-2','training-3','training-4','training-5']
    dataset = [np.load('%s/data/%s_dataset_250.npy'%(path,i), allow_pickle=True) for i in sid]
    sid = ['training-1','training-2','training-3','training-4','training-5']
    sid_val = ['testing']

    train_subject_dataset = [np.load(f'{path}/data/{i}_dataset_250.npy', allow_pickle=True) for i in sid]
    val_subject_dataset = [np.load(f'{path}/data/{i}_dataset_250.npy', allow_pickle=True) for i in sid_val]

    train_trials = build_trial_list(train_subject_dataset)
    val_trials = build_trial_list(val_subject_dataset)

    window_size = 250
    stride = 250

    X_train, Y_train, L_train = build_window_dataset(train_trials, window_size=window_size, stride=stride)
    X_val, Y_val, L_val = build_window_dataset(val_trials, window_size=window_size, stride=stride)
    train_dataset = EEGWindowDataset(X_train, Y_train, L_train)
    val_dataset = EEGWindowDataset(X_val, Y_val, L_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    save_dir = '/workspace/535505/save_model/trf_tw_250'
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

    hist = fit_trf_model_window(
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