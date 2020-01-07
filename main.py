from __future__ import division, print_function
import torch
import torch.optim as optim
import lib.config as cfg
cfg.DEVICE = torch.device('cuda')
from lib.utils import print_fn
import lib.loader
from lib.loader import ProcessedCsvDataset, Dataset, get_loader
import models.nn


def main():
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='FullyConnectedNetwork')
    parser.add_argument('--data', type=str, default='auto', choices=['auto', 'o2o', 'm2o'])
    parser.add_argument('--arch', type=int, nargs='*', default=[100])
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--bs', type=int, default=200)                        # batch size
    parser.add_argument('--lr', type=float, default=1e-4)                     # learning rate
    parser.add_argument('--wd', type=float, default=0)                        # weight decay
    parser.add_argument('--scheduler', action='store_false', default=True)    # lr_scheduler
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    # Load dataset
    if args.data == 'auto':
        args.data = 'o2o' if args.model == 'FullyConnectedNetwork' else 'm2o'
    data_name = 'One2OneDataset' if args.data == 'o2o' else 'Many2OneDataset'
    dst = ProcessedCsvDataset(root_dir='data', normalized=True)
    dst.make_val_from_test()
    train_dataset = lib.loader.__dict__[data_name](dst.train_feature, dst.train_label)
    test_dataset = lib.loader.__dict__[data_name](dst.test_feature, dst.test_label)
    val_dataset = lib.loader.__dict__[data_name](dst.val_feature, dst.val_label)
    train_loader = get_loader(train_dataset, batch_size=args.bs)
    test_loader = get_loader(test_dataset, batch_size=1024, shuffle=False)
    val_loader = get_loader(val_dataset, batch_size=1024, shuffle=False)

    # Build model
    model_args = []
    if args.model == 'FullyConnectedNetwork':
        model_args = [args.arch]
    elif args.model == 'ConvolutionalNeuralNetwork':
        model_args = [args.arch[:-1], args.arch[-1]]
    elif args.model == 'RecurrentNeuralNetwork':
        model_args = [args.arch[0]]
    model = models.nn.__dict__[args.model](dst.num_features, *model_args)

    # Make optimizer
    optimizer = None
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd,
                momentum=args.momentum)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    else:
        scheduler = None

    # Fit model
    model.fit(train_loader, optimizer, epochs=args.epochs,
            callback=print_fn(end='\r'), val_loader=val_loader, scheduler=scheduler)

    # Evaluate model
    print('\ntest mse:\t{:.5}'.format(model.validate(test_loader)))
    print('train mse:\t{:.5}'.format(model.validate(train_loader)))


if __name__ == '__main__':
    main()
