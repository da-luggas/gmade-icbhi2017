import argparse
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from model import GMADE, initialize_weights

if __name__ == "__main__":
    # Set seeds for reproducibility
    utils.set_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--bs", type=int, default=64, help="Batch size for dataloaders")
    parser.add_argument("--masks", type=int, default=5, help="Number of masks for order agnostic training")
    parser.add_argument("--hs", type=str, default="128,32,128", help="Architecture of hidden layers")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to train for")
    parser.add_argument("--device", default=torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')), help="Device to run training on")
    parser.add_argument("--samples", type=int, default=5, help="Number of hidden layer resamples")
    parser.add_argument("--resample_every", type=int, default=20, help="How often to resample during training")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--dataset", type=str, default="dataset.pt", help="Path to preprocessed dataset")
    args = parser.parse_args()

    args.hs = [int(i) for i in args.hs.replace(" ", "").split(",")]
    # Split and load data
    train_set, val_set, test_set = utils.split_data(args.dataset)

    train_loader = DataLoader(train_set, batch_size=args.bs)
    val_loader = DataLoader(val_set, batch_size=args.bs)
    test_loader = DataLoader(test_set, batch_size=args.bs)

    # Initialize the model and optimizer
    model = GMADE(128 * 5, args.hs, 128 * 5 * 2, num_masks=args.masks).to(args.device)
    initialize_weights(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize loss
    criterion = nn.GaussianNLLLoss()

    # Initialize the best validation loss and patience counter for early stopping
    best_val_loss = float('inf')
    waiting = 0

    # Initialize tensorboard
    writer = SummaryWriter()

    # Train the model
    for epoch in tqdm(range(args.epochs)):
        train_loss = utils.train_epoch(model, optimizer, criterion, train_loader, args)
        val_loss = utils.eval_epoch(model, optimizer, criterion, val_loader, args)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'state_dict': model.state_dict(),
                'args': args
            }, os.path.join(writer.log_dir, "model.pt"))
            waiting = 0
        else:
            waiting += 1

        if waiting > args.patience:
            break

        writer.add_scalar("Loss/Train", train_loss, global_step=epoch)
        writer.add_scalar("Loss/Val", val_loss, global_step=epoch)

    # Evaluation model generalization using test set
    state_dict = torch.load(os.path.join(writer.log_dir, "model.pt"))['state_dict']
    roc_auc, balacc, tpr, tnr = utils.test_model(model, val_loader, test_loader, state_dict, args)

    writer.add_scalar("Metrics/ROCAUC", roc_auc)
    writer.add_scalar("Metrics/BALACC", balacc)
    writer.add_scalar("Metrics/TPR", tpr)
    writer.add_scalar("Metrics/TNR", tnr)

    writer.close()