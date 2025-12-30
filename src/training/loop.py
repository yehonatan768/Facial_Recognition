import torch
from torch import nn


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    n = 0

    for x1, x2, y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        logits = model(x1, x2)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

    return total_loss / n


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    n = 0

    for x1, x2, y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device).float().view(-1, 1)

        logits = model(x1, x2)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

    return total_loss / n
