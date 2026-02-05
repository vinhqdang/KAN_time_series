import torch
import torch.nn as nn
from .config import DEVICE

def train_once(model, Xtr, ytr, Xte, yte, steps, lr, name):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for i in range(steps):
        opt.zero_grad()
        out = model(Xtr)
        loss = loss_fn(out, ytr)
        loss.backward()
        opt.step()
        if (i+1) % 50 == 0:
            print(f"{name} step {i+1}/{steps} | loss={loss.item():.6f}")
    model.eval()
    with torch.no_grad():
        tr = loss_fn(model(Xtr), ytr).item()
        te = loss_fn(model(Xte), yte).item()
    print(f"{name} final train MSE={tr:.6f} | test MSE={te:.6f}")
