from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lib.config as cfg
from lib.loader import OneDataset, One2OneDataset, get_loader


class Autoencoder(nn.Module):
    def __init__(self, input_features, hidden_features):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_features, input_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat

    def encode(self, x):
        if isinstance(x, OneDataset):
            loader = get_loader(x, batch_size=1024, shuffle=False)
            arr = []
            for f in loader:
                f = f.to(device=cfg.DEVICE, dtype=torch.float32)
                arr.append(self.encode(f))
            feats = torch.cat(arr, 0)
            return feats.detach().cpu().numpy()
        return self.encoder(x)


class OutputLayer(nn.Module):
    def __init__(self, input_features, num_classes=1):
        super(OutputLayer, self).__init__()
        self.fc = nn.Linear(input_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class StackedAutoencoder(nn.Module):
    def __init__(self, input_features, hidden_units, classifier=False):
        super(StackedAutoencoder, self).__init__()
        self.aes = nn.ModuleList()
        self.aes.append(Autoencoder(input_features, hidden_units[0]))
        for i in range(len(hidden_units)-1):
            in_features, out_features = hidden_units[i:i+2]
            self.aes.append(Autoencoder(in_features, out_features))
        self.classifier = None
        if classifier:
            self.classifier = OutputLayer(hidden_units[-1])

    def forward(self, x):
        with torch.no_grad():
            z = x
            for ae in self.aes:
                z = ae.encode(z)
            if self.classifier is not None:
                z = self.classifier(z)
        return z

    def encode(self, x):
        if isinstance(x, np.ndarray):
            return self.encode(OneDataset(x))
        if isinstance(x, OneDataset):
            loader = get_loader(x, batch_size=1024, shuffle=False)
            arr = []
            for f in loader:
                f = f.to(device=cfg.DEVICE, dtype=torch.float32)
                arr.append(self.encode(f))
            feats = torch.cat(arr, 0)
            return feats.detach().cpu().numpy()
        return self(x)

    def fit(*args, **kwargs):
        return fit_sae(*args, **kwargs)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


def fit_ae(model, loader, optimizer, epochs, callback, print_every, val_loader, scheduler):
    for e in range(epochs):
        model = model.to(device=cfg.DEVICE)
        model.train()
        for i, x in enumerate(loader):
            x = x.to(device=cfg.DEVICE, dtype=torch.float32)
            xhat = model(x)
            loss = torch.mean(torch.pow(xhat-x, 2))
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


def fit_sae(model, data, loader_fn=lambda x: get_loader(x),
        optimizer_fn=lambda x: optim.SGD(x, lr=1e-3, momentum=0.9),
        epochs=10, callback=lambda x:None, print_every=1000, scheduler_fn=None):
    features = data
    for i, ae in enumerate(model.aes):
        dataset = OneDataset(features)
        loader = loader_fn(dataset)
        optimizer = optimizer_fn(ae.parameters())
        scheduler = None if scheduler_fn is None else scheduler_fn(optimizer)
        fit_ae(ae, loader, optimizer, epochs,
                lambda **kwargs: callback(**kwargs, stage=i), print_every, None, scheduler)
        features = ae.encode(dataset)


