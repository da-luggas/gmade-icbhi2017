import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset

from tqdm import tqdm

from model import GMADE
from dataset import RespiratorySoundDataset

# Negative Log Likelihood (NLL) loss for the Gaussian distribution:
def nll_gaussian(y, mu, sigma):
    return (torch.log(sigma) + (y - mu)**2 / (2 * sigma**2)).sum()

###################
## CONFIGURATION ##
###################

epochs = 10
lr = 3e-4
batch_size = 128
hidden_size = 100

device = torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
device = torch.device('cpu')
##############
## TRAINING ##
##############

# Load your dataset:
dataset = RespiratorySoundDataset("/Users/lukas/Documents/PARA/1 🚀 Projects/Bachelor Thesis/ICBHI_final_database/")

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
train_loader = DataLoader(filtered_train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model and optimizer:
model = GMADE(hidden_size=hidden_size, device=device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
for epoch in tqdm(range(epochs)):
    total_loss = 0
    model.train()
    for batch_data in train_loader:
        mel_spectrogram, _ = batch_data
        mel_spectrogram = mel_spectrogram.to(device)
        
        # Split the mel spectrogram into 5-frame snippets
        snippets = [mel_spectrogram[:, :, i:i+5] for i in range(mel_spectrogram.size(2) - 5 + 1)]
        
        optimizer.zero_grad()
        
        batch_loss = 0
        for snippet in snippets:
            # Flatten snippet
            snippet = snippet.view(snippet.shape[0], 128 * 5)
            mu, sigma = model(snippet)
            loss = nll_gaussian(snippet, mu, sigma)
            batch_loss += loss
        
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")
