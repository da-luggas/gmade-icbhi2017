import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from model import GMADE

class RespiratorySoundDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def split_data(dataset, prevent_leakage=False, seed=42):
    data = torch.load(dataset)
    
    recording_ids = data['recording_ids']
    cycles = data['cycles']
    labels = data['labels']

    if prevent_leakage:
        unique_recording_ids = torch.unique(recording_ids)
        train_ids, test_ids = train_test_split(unique_recording_ids, test_size=0.2, random_state=seed)

        # Create masks for selecting data
        train_mask = torch.isin(recording_ids, train_ids)
        test_mask = torch.isin(recording_ids, test_ids)

        # Separate data based on recording_id
        X_train, y_train = cycles[train_mask], labels[train_mask]
        X_test, y_test = cycles[test_mask], labels[test_mask]
    else:
        # Random splitting 80/20
        X_train, X_test, y_train, y_test = train_test_split(cycles, labels, test_size=0.2, random_state=seed)

    # Further splitting of train set into validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    X_train, y_train = X_train[y_train == 0], y_train[y_train == 0]

    train_set = RespiratorySoundDataset(X_train, y_train)
    val_set = RespiratorySoundDataset(X_val, y_val)
    test_set = RespiratorySoundDataset(X_test, y_test)

    return train_set, val_set, test_set

def get_optimal_threshold(losses, labels):
    accuracies = []
    for threshold in losses:
        y_pred = losses > threshold
        tpr = np.sum((y_pred == 1) & (labels == 1)) / np.sum(labels == 1)
        tnr = np.sum((y_pred == 0) & (labels == 0)) / np.sum(labels == 0)
        accuracy = 0.5 * (tpr + tnr)
        accuracies.append(accuracy)
    optimal_threshold = losses[np.argmax(accuracies)]
    return optimal_threshold

def test_model(model, dataloader, state_dict, args):
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.GaussianNLLLoss(reduction='none')
    scores = []
    labels = []

    with torch.no_grad():
        for batch_data in dataloader:
            mels, y = batch_data
            mels = mels.to(args.device)

            # Split the mels into 5-frame snippets
            snippets = [mels[:, :, i:i+5] for i in range(mels.size(2) - 5 + 1)]

            batch_loss = torch.zeros([mels.shape[0]])
            for snippet in snippets:
                # Flatten snippets for input in linear nn
                snippet = snippet.reshape(snippet.shape[0], -1)
                # Perform order/connectivity agnostic eval by resampling the masks
                outputs = torch.zeros(snippet.shape[0], snippet.shape[1] * 2, device=args.device)
                for s in range(args.samples):
                    # Update model at each step
                    model.update_masks()
                    outputs += model(snippet)

                outputs /= args.samples

                # Reshape outputs to mu and logvar
                outputs = outputs.view(-1, 128, 5, 2)
                mu, logvar = outputs[..., 0], outputs[..., 1]
                print(mu)
                # Reshape snippets back to original shape
                snippet = snippet.view(-1, 128, 5)

                loss = criterion(mu, snippet, logvar.exp()).mean((1, 2))
                batch_loss += loss.cpu()

            batch_loss /= len(snippets)

            scores.extend(batch_loss)
            labels.extend(y)

    scores = np.array(scores)
    labels = np.array(labels)

    threshold = get_optimal_threshold(scores, labels)
    predictions = (scores > threshold)

    roc_auc = roc_auc_score(labels, scores)
    tpr = np.sum((predictions == 1) & (labels == 1)) / np.sum(labels == 1)
    tnr = np.sum((predictions == 0) & (labels == 0)) / np.sum(labels == 0)
    balanced_accuracy = 0.5 * (tpr + tnr)

    print("ROC-AUC Score: ", roc_auc.round(2))
    print("BALACC: ", balanced_accuracy.round(2))
    print("TPR: ", tpr.round(2))
    print("TNR: ", tnr.round(2))

if __name__ == "__main__":
    # Set seeds for reproducibility
    set_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--bs", type=int, default=64, help="Batch size for dataloaders")
    parser.add_argument("--hs", type=list, default=[128, 32, 128], help="Architecture of hidden layers")
    parser.add_argument("--device", default=torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')), help="Device to run training on")
    parser.add_argument("--samples", type=int, default=5, help="Number of hidden layer resamples")
    parser.add_argument("--resample_every", type=int, default=20, help="How often to resample during training")
    parser.add_argument("--dataset", type=str, default="dataset.pt", help="Path to preprocessed dataset")
    parser.add_argument("--model", type=str, default="runs/Nov10_17-51-48_Lukass-Air.lan/model.pt", help="Path to saved model from training")
    args = parser.parse_args()

    # Split and load data
    _, _, test_set = split_data(args.dataset)

    test_loader = DataLoader(test_set, batch_size=args.bs)

    # Initialize the model and optimizer
    model = GMADE(128 * 5, args.hs, 128 * 5 * 2).to(args.device)

    # Evaluation model generalization using test set
    state_dict = torch.load(args.model)
    test_model(model, test_loader, state_dict, args)