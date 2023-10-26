import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter

import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse
import os

from model import GMADE
from dataset import RespiratorySoundDataset

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

###################
## CONFIGURATION ##
###################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--hidden_sizes', type=str, default="100")
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--data_path', type=str, default="/home/lukas/thesis/anogan2d/dataset")
parser.add_argument('--resample_every', type=int, default=20)
args = parser.parse_args()

device = torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

hidden_sizes = [int(size) for size in args.hidden_sizes.split(',')]

##############
## TRAINING ##
##############

# Initialize tensorboard logging
writer = SummaryWriter()

# Load your dataset:
dataset = RespiratorySoundDataset(args.data_path)

# Splitting dataset into train and test set
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Further splitting into training and validation set out of training set
validation_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - validation_size

actual_train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

# Filtering out samples with label 1 from training dataset
indices = [i for i in range(len(actual_train_dataset)) if actual_train_dataset[i][1] == 0]
filtered_train_dataset = Subset(actual_train_dataset, indices)

# Create DataLoaders
train_loader = DataLoader(filtered_train_dataset, batch_size=args.batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize the model, optimizer and scheduler
model = GMADE(input_size=128 * 5, hidden_sizes=hidden_sizes, output_size=128 * 5 * 2).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=args.patience // 2, threshold=0.01, verbose=True)

# Initialize our loss function
nll_loss = nn.GaussianNLLLoss(full=True)

# Initialize best validation loss and patience counter for early stopping
best_val_loss = float('inf')
wait = 0

# Train the model
for epoch in tqdm(range(args.epochs)):
    train_loss = 0
    model.train()

    for batch_idx, batch_data in enumerate(train_loader):
        # Handle model mask update every args.resample_every steps
        step = epoch * len(train_loader) + batch_idx + 1
        if step % args.resample_every == 0:
            model.update_masks()

        mel_spectrogram, _ = batch_data
        mel_spectrogram = mel_spectrogram.to(device)
        mel_spectrogram = mel_spectrogram.squeeze(1)
        
        # Split the mel spectrogram into 5-frame snippets
        snippets = [mel_spectrogram[:, :, i:i+5] for i in range(mel_spectrogram.size(2) - 5 + 1)]
        
        # Concatenate snippets along batch dimension
        snippets = torch.cat(snippets, 0)
        
        # Flatten snippets
        snippets = snippets.reshape(-1, 128*5)
        
        optimizer.zero_grad()
        
        # Forward pass for all snippets
        outputs = model(snippets)
        
        # Reshape outputs to mu and logvar
        outputs = outputs.view(-1, 128, 5, 2)
        mu, logvar = outputs[..., 0], outputs[..., 1]
        
        # Reshape snippets back to original shape
        snippets = snippets.view(-1, 128, 5)
        
        loss = nll_loss(mu, snippets, logvar.exp())
                
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)

    # Validate current model
    val_loss = 0
    model.eval()

    with torch.no_grad():
        for batch_data in validation_loader:
            # Update model mask at every step
            model.update_masks()

            mel_spectrogram, _ = batch_data
            mel_spectrogram = mel_spectrogram.to(device)
            mel_spectrogram = mel_spectrogram.squeeze(1)

            # Split the mel spectrogram into 5-frame snippets
            snippets = [mel_spectrogram[:, :, i:i+5] for i in range(mel_spectrogram.size(2) - 5 + 1)]
            
            # Concatenate snippets along batch dimension
            snippets = torch.cat(snippets, 0)
            
            # Flatten snippets
            snippets = snippets.reshape(-1, 128*5)
            
            # Forward pass for all snippets
            outputs = model(snippets)
            
            # Reshape outputs to mu and logvar
            outputs = outputs.view(-1, 128, 5, 2)
            mu, logvar = outputs[..., 0], outputs[..., 1]
            
            # Reshape snippets back to original shape
            snippets = snippets.view(-1, 128, 5)
            
            loss = nll_loss(mu, snippets, logvar.exp())
            
            val_loss += loss.item()
            
        val_loss = val_loss / len(validation_loader)
    
    # Check for early stopping (with delta of 0.01)
    if abs(best_val_loss - val_loss) > 0.01:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_model.pt'))
        wait = 0
    else:
        wait += 1

    if wait > args.patience:
        print("Early stopping.")
        break

    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Val", val_loss, epoch)

    scheduler.step(val_loss)

################
## EVALUATION ##
################

model.load_state_dict(torch.load(os.path.join(writer.log_dir, 'best_model.pt')))
model.eval()

# Define loss for evaluation (no reduction)
nll_loss = nn.GaussianNLLLoss(reduction="none", full=True)
test_scores = []
test_labels = []

with torch.no_grad():
    for batch_data in test_loader:
        # Update model mask at every step
        model.update_masks()

        mel_spectrogram, labels = batch_data
        mel_spectrogram = mel_spectrogram.to(device)
        # Remove channel dimension
        mel_spectrogram = mel_spectrogram.squeeze(1)

        # Split the mel spectrogram into 5-frame snippets
        snippets = [mel_spectrogram[:, :, i:i+5] for i in range(mel_spectrogram.size(2) - 5 + 1)]
        
        batch_test_loss = torch.zeros([mel_spectrogram.shape[0]])
        for snippet in snippets:
            # Flatten snippet
            snippet = snippet.reshape(snippet.shape[0], -1)
            output = model(snippet)
            # Reshape outputs to mu and logvar
            output = output.view(snippet.shape[0], 128, 5, 2)
            mu, logvar = output[..., 0], output[..., 1]
            # Reshape snippet back to original shape
            snippet = snippet.view(snippet.shape[0], 128, 5)
            loss = nll_loss(mu, snippet, logvar.exp()).mean((1, 2))
            batch_test_loss += loss.cpu()

        test_scores.extend(batch_test_loss)
        test_labels.extend(labels)

# Compute AUC score
roc_auc = roc_auc_score(test_labels, test_scores)
writer.add_hparams(vars(args), {"AUC": roc_auc})

writer.close()