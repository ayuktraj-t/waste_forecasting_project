"""Trains the LSTM in its own process (so PyTorch never shares a process with xgboost/sklearn;
that combination can freeze on macOS). Usage: python lstm_job.py in.npz out.npz"""
import sys
import numpy as np
import torch
import torch.nn as nn

WINDOW, HIDDEN, EPOCHS, BATCH, SEED = 7, 48, 15, 256, 42


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.lstm = nn.LSTM(n_in, HIDDEN, batch_first=True)
        self.head = nn.Sequential(nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def windows(X, y, zones):
    Xs, ys, idx = [], [], []
    for z in np.unique(zones):
        ii = np.where(zones == z)[0]
        for k in range(WINDOW, len(ii)):
            Xs.append(X[ii[k - WINDOW + 1:k + 1]])
            idx.append(ii[k]); ys.append(y[ii[k]] if y is not None else 0.0)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(idx)


def main(src, dst):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    d = np.load(src)
    mx, sx = d["X"].mean(0), d["X"].std(0) + 1e-9
    my, sy = float(d["y"].mean()), float(d["y"].std()) + 1e-9
    Xw, yw, _ = windows((d["X"] - mx) / sx, (d["y"] - my) / sy, d["zones"])
    print(f"      LSTM: {len(Xw):,} training windows", flush=True)
    net = Net(Xw.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt, yt = torch.from_numpy(Xw), torch.from_numpy(yw)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), BATCH):
            ib = perm[b:b + BATCH]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xt[ib]), yt[ib])
            loss.backward(); opt.step()
        print(f"      LSTM epoch {ep + 1}/{EPOCHS}  loss={float(loss):.4f}", flush=True)
    Xa, _, idx = windows((d["X_all"] - mx) / sx, None, d["zones_all"])
    net.eval()
    with torch.no_grad():
        p = np.concatenate([net(torch.from_numpy(Xa[i:i + 4096])).numpy() for i in range(0, len(Xa), 4096)])
    np.savez(dst, pred=p * sy + my, idx=idx)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
