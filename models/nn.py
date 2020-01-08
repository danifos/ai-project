from __future__ import division, print_function
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lib.config as cfg
from lib.loader import OneDataset, One2OneDataset, ManyDataset, Many2OneDataset, get_loader


class FullyConnectedNetwork(nn.Module):
    def __init__(self, num_features, hidden_units, num_classes=1, len_sequence=1):
        super(FullyConnectedNetwork, self).__init__()
        self.mlps = nn.ModuleList()
        self.mlps.append(nn.Linear(num_features * len_sequence, hidden_units[0]))
        for i in range(len(hidden_units)-1):
            in_features, out_features = hidden_units[i:i+2]
            mlp = nn.Linear(in_features, out_features)
            self.mlps.append(mlp)
        self.num_classes = num_classes
        self.fc = nn.Linear(hidden_units[-1], num_classes)
        self.use_seq = (len_sequence > 1)

    def forward(self, x):
        """
        Inputs:
            x: N x F  /  N x L x F
        Return:
            x: N x 1
        """
        if self.use_seq:
            N = x.shape[0]
            x = x.view((N, -1))
        for mlp in self.mlps:
            x = mlp(x)
            x = F.relu(x)
        x = self.fc(x)
        return x

    def fit(self, *args, **kwargs):
        fit_nn(self, *args, **kwargs)

    def predict(self, data):
        loader = get_loader(OneDataset(data), batch_size=1024, shuffle=False)
        return raw_predict(self, loader)

    def validate(self, loader):
        return check_reg(self, loader)


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, num_channels, hidden_units, kernel_size,
            len_sequence=10, num_classes=1):
        super(ConvolutionalNeuralNetwork, self).__init__()
        if np.isscalar(kernel_size):
            kernel_size = [kernel_size for _ in range(len(hidden_units))]
        else:
            assert len(kernel_size) == len(hidden_units), 'Number of Conv layers not matched'

        self.cnns = nn.ModuleList()
        num_features = len_sequence
        hidden_units.insert(0, num_channels)
        for i in range(len(hidden_units) - 1):
            in_channels, out_channels = hidden_units[i:i+2]
            size = kernel_size[i]
            cnn = nn.Conv1d(in_channels, out_channels, size)
            self.cnns.append(cnn)
            num_features = num_features - size + 1

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channels * num_features, num_classes)

    def forward(self, x):
        """
        Inputs:
            x: N x L x F
        Return:
            out: N x 1
        """
        x = x.permute((0, 2, 1))  # N x F x L
        feats = x
        for cnn in self.cnns:
            feats = cnn(feats)
        feats = self.flatten(feats)
        score = self.fc(feats)
        return score

    def predict(self, data):
        loader = get_loader(ManyDataset(data), batch_size=32, shuffle=False)
        return raw_predict(self, loader)

    def fit(*args, **kwargs):
        return fit_nn(*args, **kwargs)

    def validate(self, loader):
        return check_reg(self, loader)


class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, num_features, hidden_features, num_classes=1):
        super(RecurrentNeuralNetwork, self).__init__()
        self.backbone = nn.LSTM(num_features, hidden_features)
        self.mlp = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        """
        Inputs:
            x: N x L x F
        Return:
            out: N x 1
        """
        x = x.permute((1, 0, 2))  # L x N x F
        out, (h_t, c_t) = self.backbone(x)
        out = self.mlp(out[-1])
        return out

    def predict(self, data):
        loader = get_loader(ManyDataset(data), batch_size=32, shuffle=False)
        return raw_predict(self, loader)

    def fit(*args, **kwargs):
        return fit_nn(*args, **kwargs)

    def validate(self, loader):
        return check_reg(self, loader)


def raw_predict(model, loader):
    outputs = []
    for x in loader:
        x = x.to(device=cfg.DEVICE, dtype=torch.float32)
        y = model(x)
        outputs.append(y.detach().cpu().numpy())
    return np.concatenate(outputs, 0).squeeze()


def check_cls(model, loader):
    num_total = num_corr = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=cfg.DEVICE, dtype=torch.float32)
            y = y.to(device=cfg.DEVICE, dtype=torch.long)
            logits = model(x)
            pred, _ = logits.max(1)
            num_corr += (pred == y).sum().item()
            num_total += x.shape[0]
    return num_corr / num_total


def check_reg(model, loader):
    num = mean = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=cfg.DEVICE, dtype=torch.float32)
            y = y.to(device=cfg.DEVICE, dtype=torch.float32)
            if loader.collate_fn is torch.utils.data._utils.collate.default_collate:
                N = x.shape[0]
            else:
                warnings.warn('check_reg() using dim 1 as the batch dim')
                N = x.shape[1]
            preds = model(x).squeeze(1)
            se = np.sum(((preds - y) ** 2).detach().cpu().numpy())
            if num == 0:
                mean = se / N
                num = N
            else:
                mean = mean * (num/(num+N)) + se / (num+N)
                num += N
    return mean


def fit_nn(model, loader, optimizer, epochs=10, l1_rate=0,
        callback=lambda x:None, print_every=1000, val_loader=None, scheduler=None):
    for e in range(epochs):
        model = model.to(device=cfg.DEVICE)
        for i, (x, y) in enumerate(loader):
            model.train()
            x = x.to(device=cfg.DEVICE, dtype=torch.float32)
            y = y.to(device=cfg.DEVICE, dtype=torch.float32)
            logits = model(x).squeeze(1)
            loss = F.mse_loss(logits, y)
            if l1_rate != 0:  # Apply L1 regularization (LASSO regression)
                params = torch.cat([x.flatten() for x in model.parameters()])
                loss += l1_rate * torch.norm(params, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % print_every == 0:
                if val_loader is not None:
                    val_loss = model.validate(val_loader)
                else:
                    val_loss = float('nan')
                callback(epoch=e, it=i, loss=loss, val_loss=val_loss)
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) and \
                            val_loader is not None:
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
        if np.isnan(loss.item()):
            print('\n[Stopped] nan loss')
            break

