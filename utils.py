import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class RespiratorySoundDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def set_seeds(seed=999):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def split_data(dataset, prevent_leakage=False, seed=999):
    data = torch.load(dataset)
    
    recording_ids = data['recording_ids']
    cycles = data['cycles']
    labels = data['labels']

    if prevent_leakage:
        unique_recording_ids = torch.unique(recording_ids)
        train_ids, test_ids = train_test_split(unique_recording_ids, test_size=0.2, random_state=seed, stratify=labels)

        # Create masks for selecting data
        train_mask = torch.isin(recording_ids, train_ids)
        test_mask = torch.isin(recording_ids, test_ids)

        # Separate data based on recording_id
        X_train, y_train = cycles[train_mask], labels[train_mask]
        X_test, y_test = cycles[test_mask], labels[test_mask]
    else:
        # Random splitting 80/20
        X_train, X_test, y_train, y_test = train_test_split(cycles, labels, test_size=0.2, random_state=seed, stratify=labels)

    # Further splitting of train set into validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train)

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

def train_epoch(model, optimizer, criterion, dataloader, args):
    train_loss = 0
    model.train()

    for batch_idx, batch_data in enumerate(dataloader):
        mels, _ = batch_data
        mels = mels.to(args.device)

        # Split the mels into 5-frame snippets
        snippets = [mels[:, :, i:i+5] for i in range(0, mels.size(2) - 5 + 1, 5)]
        # snippets = [mels[:, :, i:i+5] for i in range(mels.size(2) - 5 + 1)]
        # Concatenate snippets along batch dimension
        snippets = torch.cat(snippets, 0)
        # Flatten snippets for input in linear nn
        snippets = snippets.reshape(-1, 128 * 5)
        # Perform order/connectivity agnostic training by resampling the masks
        outputs = torch.zeros(snippets.shape[0], snippets.shape[1] * 2, device=args.device)
        for s in range(args.samples):
            if batch_idx % args.resample_every == 0:
                model.update_masks()
            outputs += model(snippets)

        outputs /= args.samples

        # Reshape outputs to mu and logvar
        outputs = outputs.view(-1, 128, 5, 2)
        mu, logvar = outputs[..., 0], outputs[..., 1]
        # Reshape snippets back to original shape
        snippets = snippets.view(-1, 128, 5)

        loss = criterion(mu, snippets, logvar.exp())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss / len(dataloader)
    return train_loss

def eval_epoch(model, optimizer, criterion, dataloader, args):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience // 2, min_lr=3e-5, factor=0.1)

    val_loss = 0
    model.eval()

    with torch.no_grad():
        for batch_data in dataloader:
            mels, _ = batch_data
            mels = mels.to(args.device)

            # Split the mels into 5-frame snippets
            snippets = [mels[:, :, i:i+5] for i in range(0, mels.size(2) - 5 + 1, 5)]
            # snippets = [mels[:, :, i:i+5] for i in range(mels.size(2) - 5 + 1)]
            # Concatenate snippets along batch dimension
            snippets = torch.cat(snippets, 0)
            # Flatten snippets for input in linear nn
            snippets = snippets.reshape(-1, 128 * 5)
            # Perform order/connectivity agnostic eval by resampling the masks
            outputs = torch.zeros(snippets.shape[0], snippets.shape[1] * 2, device=args.device)
            for s in range(args.samples):
                # Update model at each step
                model.update_masks()
                outputs += model(snippets)

            outputs /= args.samples

            # Reshape outputs to mu and logvar
            outputs = outputs.view(-1, 128, 5, 2)
            mu, logvar = outputs[..., 0], outputs[..., 1]
            # Reshape snippets back to original shape
            snippets = snippets.view(-1, 128, 5)

            loss = criterion(mu, snippets, logvar.exp())
            val_loss += loss.item()

    val_loss = val_loss / len(dataloader)
    scheduler.step(val_loss)
    return val_loss

def test_model(model, val_dataloader, test_dataloader, state_dict, args):
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.GaussianNLLLoss(reduction='none')
    val_scores = []
    val_labels = []

    test_scores = []
    test_labels = []

    with torch.no_grad():
        for batch_data in val_dataloader:
            mels, y = batch_data
            mels = mels.to(args.device)

            # Split the mels into 5-frame snippets
            snippets = [mels[:, :, i:i+5] for i in range(0, mels.size(2) - 5 + 1, 5)]
            # snippets = [mels[:, :, i:i+5] for i in range(mels.size(2) - 5 + 1)]

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
                # Reshape snippets back to original shape
                snippet = snippet.view(-1, 128, 5)

                loss = criterion(mu, snippet, logvar.exp()).sum((1, 2))
                batch_loss += loss.cpu()

            batch_loss /= len(snippets)

            val_scores.extend(batch_loss)
            val_labels.extend(y)

        for batch_data in test_dataloader:
            mels, y = batch_data
            mels = mels.to(args.device)

            # Split the mels into 5-frame snippets
            snippets = [mels[:, :, i:i+5] for i in range(0, mels.size(2) - 5 + 1, 5)]
            # snippets = [mels[:, :, i:i+5] for i in range(mels.size(2) - 5 + 1)]

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
                # Reshape snippets back to original shape
                snippet = snippet.view(-1, 128, 5)

                loss = criterion(mu, snippet, logvar.exp()).sum((1, 2))
                batch_loss += loss.cpu()

            batch_loss /= len(snippets)

            test_scores.extend(batch_loss)
            test_labels.extend(y)

    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels)
    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)

    threshold = get_optimal_threshold(val_scores, val_labels)
    predictions = (test_scores > threshold)

    roc_auc = roc_auc_score(test_labels, test_scores)
    tpr = np.sum((predictions == 1) & (test_labels == 1)) / np.sum(test_labels == 1)
    tnr = np.sum((predictions == 0) & (test_labels == 0)) / np.sum(test_labels == 0)
    balanced_accuracy = 0.5 * (tpr + tnr)

    print("ROC-AUC Score: ", roc_auc.round(2))
    print("BALACC: ", balanced_accuracy.round(2))
    print("TPR: ", tpr.round(2))
    print("TNR: ", tnr.round(2))

    return roc_auc, balanced_accuracy, tpr, tnr