import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


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

def load_data(dataset, batch_size):
    saved_data = torch.load(dataset)

    X_train, X_val, X_test, y_train, y_val, y_test = saved_data['X_train'], saved_data['X_val'], saved_data['X_test'], saved_data['y_train'], saved_data['y_val'], saved_data['y_test']

    train_set = RespiratorySoundDataset(X_train, y_train)
    val_set = RespiratorySoundDataset(X_val, y_val)
    test_set = RespiratorySoundDataset(X_test, y_test)
    
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader

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

def train_epoch(model, optimizer, dataloader, args):
    criterion = nn.GaussianNLLLoss()

    train_loss = 0
    model.train()

    for batch_idx, batch_data in enumerate(dataloader):
        mels, _ = batch_data
        mels = mels.to(args.device)

        # Split the mels into 5-frame snippets
        # snippets = [mels[:, :, i:i+5] for i in range(0, mels.size(2) - 5 + 1, 5)]
        snippets = [mels[:, :, i:i+5] for i in range(mels.size(2) - 5 + 1)]
        # Concatenate snippets along batch dimension
        snippets = torch.cat(snippets, 0)
        # Flatten snippets for input in linear nn
        snippets = snippets.reshape(-1, 13 * 5)
        # Perform order/connectivity agnostic training by resampling the masks
        outputs = torch.zeros(snippets.shape[0], snippets.shape[1] * 2, device=args.device)
        for s in range(args.samples):
            if batch_idx % args.resample_every == 0:
                model.update_masks()
            outputs += model(snippets)

        outputs /= args.samples

        # Reshape outputs to mu and logvar
        outputs = outputs.view(-1, 13, 5, 2)
        mu, logvar = outputs[..., 0], outputs[..., 1]
        # Reshape snippets back to original shape
        snippets = snippets.view(-1, 13, 5)

        loss = criterion(mu, snippets, logvar.exp())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss / len(dataloader)
    return train_loss

def eval_epoch(model, scheduler, dataloader, args):
    criterion = nn.GaussianNLLLoss()
    val_loss = 0
    model.eval()

    with torch.no_grad():
        for batch_data in dataloader:
            mels, _ = batch_data
            mels = mels.to(args.device)

            # Split the mels into 5-frame snippets
            # snippets = [mels[:, :, i:i+5] for i in range(0, mels.size(2) - 5 + 1, 5)]
            snippets = [mels[:, :, i:i+5] for i in range(mels.size(2) - 5 + 1)]
            # Concatenate snippets along batch dimension
            snippets = torch.cat(snippets, 0)
            # Flatten snippets for input in linear nn
            snippets = snippets.reshape(-1, 13 * 5)
            # Perform order/connectivity agnostic eval by resampling the masks
            outputs = torch.zeros(snippets.shape[0], snippets.shape[1] * 2, device=args.device)
            for s in range(args.samples):
                # Update model at each step
                model.update_masks()
                outputs += model(snippets)

            outputs /= args.samples

            # Reshape outputs to mu and logvar
            outputs = outputs.view(-1, 13, 5, 2)
            mu, logvar = outputs[..., 0], outputs[..., 1]
            # Reshape snippets back to original shape
            snippets = snippets.view(-1, 13, 5)

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
            y = y > 0

            # Split the mels into 5-frame snippets
            # snippets = [mels[:, :, i:i+5] for i in range(0, mels.size(2) - 5 + 1, 5)]
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
                outputs = outputs.view(-1, 13, 5, 2)
                mu, logvar = outputs[..., 0], outputs[..., 1]
                # Reshape snippets back to original shape
                snippet = snippet.view(-1, 13, 5)

                loss = criterion(mu, snippet, logvar.exp()).sum((1, 2))
                batch_loss += loss.cpu()

            batch_loss /= len(snippets)

            val_scores.extend(batch_loss)
            val_labels.extend(y)

        for batch_data in test_dataloader:
            mels, y = batch_data
            mels = mels.to(args.device)
            y = y > 0

            # Split the mels into 5-frame snippets
            # snippets = [mels[:, :, i:i+5] for i in range(0, mels.size(2) - 5 + 1, 5)]
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
                outputs = outputs.view(-1, 13, 5, 2)
                mu, logvar = outputs[..., 0], outputs[..., 1]
                # Reshape snippets back to original shape
                snippet = snippet.view(-1, 13, 5)

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
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    print("tn, fp, fn, tp")
    print(tn, fp, fn, tp)

    balanced_accuracy = 0.5 * (tpr + tnr)

    return roc_auc, balanced_accuracy, tpr, tnr

def test_ensemble(models, val_dataloader, test_dataloader, args):
    for model in models:
        model.eval()

    criterion = nn.GaussianNLLLoss(reduction='none')
    val_scores = []
    val_labels = []

    test_scores = []
    test_labels = []

    with torch.no_grad():
        for dataloader in [val_dataloader, test_dataloader]:
            scores = []
            labels = []

            for batch_data in dataloader:
                mels, y = batch_data
                mels = mels.to(args.device)

                # Split the mels into 5-frame snippets
                snippets = [mels[:, :, i:i + 5] for i in range(mels.size(2) - 5 + 1)]

                batch_loss = torch.zeros([mels.shape[0]])
                for snippet in snippets:
                    snippet = snippet.reshape(snippet.shape[0], -1)
                    outputs = torch.zeros(snippet.shape[0], snippet.shape[1] * 2, device=args.device)

                    for model in models:
                        model_output = torch.zeros_like(outputs)
                        for s in range(args.samples):
                            model.update_masks()
                            model_output += model(snippet)

                        model_output /= args.samples
                        outputs += model_output

                    outputs /= len(models)

                    # Reshape outputs to mu and logvar
                    outputs = outputs.view(-1, 13, 5, 2)
                    mu, logvar = outputs[..., 0], outputs[..., 1]
                    # Reshape snippets back to original shape
                    snippet = snippet.view(-1, 13, 5)

                    loss = criterion(mu, snippet, logvar.exp()).sum((1, 2))
                    batch_loss += loss.cpu()

                batch_loss /= len(snippets)

                scores.extend(batch_loss.tolist())
                labels.extend(y.tolist())

            if dataloader == val_dataloader:
                val_scores = np.array(scores)
                val_labels = np.array(labels)
            else:
                test_scores = np.array(scores)
                test_labels = np.array(labels)

    # Calculate the optimal threshold
    threshold = get_optimal_threshold(val_scores, val_labels)
    predictions = (test_scores > threshold)

    roc_auc = roc_auc_score(test_labels, test_scores)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    print("tn, fp, fn, tp")
    print(tn, fp, fn, tp)

    balanced_accuracy = 0.5 * (tpr + tnr)

    return roc_auc, balanced_accuracy, tpr, tnr