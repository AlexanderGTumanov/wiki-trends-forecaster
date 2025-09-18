import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

def slug(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")

def norm(xs: list[str] | None) -> list[str] | None:
    if xs is None:
        return None
    return [slug(x).lower() for x in xs]

class ReturnForecaster(nn.Module):
    def __init__(self, window: int, horizon: int, hidden_sizes = (64, 64), dropout_rate= 0.2):
        super().__init__()
        layers = []
        prev = window
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p = dropout_rate))
            prev = h
        self.shared_net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, horizon)
        self.logvar_head = nn.Linear(prev, horizon)
        self.window = window
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.shared_net(x)
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        return mean, logvar

def gaussian_nll_loss(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(0.5 * (math.log(2 * math.pi) + logvar) + 0.5 * ((target - mean) ** 2) * torch.exp(-logvar))

def create_dataset(df: pd.DataFrame, window: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    cats = df["category"].astype(str).unique()
    seqs, targs = [], []
    for cat in cats:
        sub = df[df["category"] == cat].sort_values("week_start")
        v = np.log1p(sub["weekly_views"].to_numpy())
        for i in range(len(v) - window - horizon + 1):
            seqs.append(v[i:i + window].astype(np.float32))
            targs.append(v[i + window:i + window + horizon].astype(np.float32))
    return np.asarray(seqs, dtype = np.float32), np.asarray(targs, dtype = np.float32)

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = sequences
        self.targets = targets

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = torch.tensor(self.sequences[idx], dtype = torch.float32)
        targ = torch.tensor(self.targets[idx], dtype = torch.float32)
        baseline = seq[-1]
        seq = seq - baseline
        targ = targ - baseline
        return seq, targ

def prepare_dataloaders(sequences: np.ndarray, targets: np.ndarray, batch_size = 32, val_ratio = 0.2, shuffle = True) -> tuple[DataLoader, DataLoader]:
    ds = TimeSeriesDataset(sequences, targets)
    n = len(ds)
    idx = np.random.permutation(n)
    v = int(n * val_ratio)
    val_idx, train_idx = idx[:v], idx[v:]
    train_loader = DataLoader(Subset(ds, train_idx), batch_size = batch_size, shuffle = shuffle)
    valid_loader = DataLoader(Subset(ds, val_idx), batch_size = batch_size, shuffle = False)
    return train_loader, valid_loader

def train_model(model: ReturnForecaster, train_loader: DataLoader, valid_loader: DataLoader, epochs: int = 100, lr: float = 1e-3) -> tuple[ReturnForecaster, pd.DataFrame]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_metric_history, valid_metric_history = [], []
    import math
    for _ in range(epochs):
        model.train()
        train_se, train_n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            mean, logvar = model(xb)
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            loss = gaussian_nll_loss(mean, logvar, yb)
            loss.backward()
            opt.step()
            train_se += torch.sum((mean - yb)**2).item()
            train_n += yb.numel()
        train_rmse = math.sqrt(train_se / train_n) if train_n > 0 else float("nan")
        model.eval()
        valid_se, valid_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mean, logvar = model(xb)
                logvar = torch.clamp(logvar, min=-10.0, max=10.0)
                valid_se += torch.sum((mean - yb)**2).item()
                valid_n += yb.numel()
        valid_rmse = math.sqrt(valid_se / valid_n) if valid_n > 0 else float("nan")
        train_metric_history.append(train_rmse)
        valid_metric_history.append(valid_rmse)
    history = pd.DataFrame({"train_metric": train_metric_history, "valid_metric": valid_metric_history})
    return model, history

def forecast(data: pd.DataFrame, model: ReturnForecaster, horizon = 12, categories: list[str] | None = None) -> pd.DataFrame:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    df = data.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["cat_norm"] = df["category"].str.lower()
    if categories is not None:
        target = norm(categories)
        if target is not None:
            df = df[df["cat_norm"].isin(target)]
    cats = df["category"].astype(str).unique().tolist()
    out = []
    for cat in cats:
        sub = df[df["category"] == cat].sort_values("week_start")
        v_orig = sub["weekly_views"].to_numpy(dtype = np.float32)
        v_log = np.log1p(v_orig)
        baseline = v_log[-1]
        x = torch.tensor(v_log[-model.window:], dtype = torch.float32, device = device).unsqueeze(0)
        x = x - baseline
        model.eval()
        with torch.no_grad():
            mean, logvar = model(x)
            mean = mean.squeeze(0).cpu().numpy()
            std = np.exp(0.5 * logvar.squeeze(0).cpu().numpy())
        last = sub["week_start"].max()
        ds = [last] + [last + pd.Timedelta(days = 7 * (i + 1)) for i in range(horizon)]
        y0 = v_orig[-1]
        pred_log = mean[:horizon] + baseline
        y = np.concatenate(([y0], np.expm1(pred_log)))
        y_lower = np.concatenate(([y0], np.expm1((mean[:horizon] - 1.96 * std[:horizon]) + baseline)))
        y_upper = np.concatenate(([y0], np.expm1((mean[:horizon] + 1.96 * std[:horizon]) + baseline)))
        out.append(pd.DataFrame({"category": cat, "ds": ds, "yhat": y, "yhat_lower": y_lower, "yhat_upper": y_upper}))
    return pd.concat(out, ignore_index = True) if out else pd.DataFrame(columns = ["category", "ds", "yhat", "yhat_lower", "yhat_upper"])

def plot_training_history(history: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize = (10, 5))
    plt.plot(history.index + 1, history["train_metric"], label = "train")
    plt.plot(history.index + 1, history["valid_metric"], label = "valid")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (log-delta)")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
