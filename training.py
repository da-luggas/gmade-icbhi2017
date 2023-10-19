import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset

import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from model import GMADE
from dataset import RespiratorySoundDataset

# Set seeds for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)

# Negative Log Likelihood (NLL) loss for the Gaussian distribution:
def nll_gaussian(y, mu, logvar, eval=False):
    # Calculate the variance from logvar
    var = torch.exp(logvar)

    # Calculate the Gaussian NLL
    if eval is True:
        nll = (0.5 * (logvar + ((y - mu) ** 2) / var + torch.log(torch.tensor(2.0 * 3.141592653589793)))).sum(1)
    else:
        nll = (0.5 * (logvar + ((y - mu) ** 2) / var + torch.log(torch.tensor(2.0 * 3.141592653589793)))).sum()

    return nll

###################
## CONFIGURATION ##
###################

epochs = 1
lr = 3e-4
batch_size = 128
hidden_size = 100
patience = 10

device = torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
device = torch.device('cpu') # REMOVE: debugging only

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

# Initialize best validation loss and patience counter for early stopping
best_val_loss = float('inf')
wait = 0

# Train the model
for epoch in tqdm(range(epochs)):
    total_loss = 0
    model.train()

    for batch_data in train_loader:
        mel_spectrogram, _ = batch_data
        mel_spectrogram = mel_spectrogram.to(device)
        # Remove channel dimension
        mel_spectrogram = mel_spectrogram.squeeze(1)
        
        # Split the mel spectrogram into 5-frame snippets
        snippets = [mel_spectrogram[:, :, i:i+5] for i in range(mel_spectrogram.size(2) - 5 + 1)]
        
        optimizer.zero_grad()
        
        batch_loss = 0
        for snippet in snippets:
            # Flatten snippet
            snippet = snippet.reshape(snippet.shape[0], -1)
            mu, logvar = model(snippet)
            loss = nll_gaussian(snippet, mu, logvar)
            batch_loss += loss
        
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

    # Validate current model
    val_loss = 0
    model.eval()

    with torch.no_grad():
        for batch_data in validation_loader:
            mel_spectrogram, _ = batch_data
            mel_spectrogram = mel_spectrogram.to(device)
            # Remove channel dimension
            mel_spectrogram = mel_spectrogram.squeeze(1)

            
            # Split the mel spectrogram into 5-frame snippets
            snippets = [mel_spectrogram[:, :, i:i+5] for i in range(mel_spectrogram.size(2) - 5 + 1)]
            
            batch_val_loss = 0
            for snippet in snippets:
                snippet = snippet.reshape(snippet.shape[0], -1)
                mu, logvar = model(snippet)
                loss = nll_gaussian(snippet, mu, logvar)
                batch_val_loss += loss
                
            val_loss += batch_val_loss.item()
        
    val_loss /= len(validation_loader)
    
    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        wait = 0
    else:
        wait += 1

    if wait > patience:
        print("Early stopping!")
        break

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}, Val Loss: {val_loss}")

################
## EVALUATION ##
################

model.load_state_dict(torch.load('best_model.pt'))
model.eval()

test_scores = []
test_labels = []

with torch.no_grad():
    for batch_data in test_loader:
        mel_spectrogram, labels = batch_data
        mel_spectrogram = mel_spectrogram.to(device)
        # Remove channel dimension
        mel_spectrogram = mel_spectrogram.squeeze(1)

        # Split the mel spectrogram into 5-frame snippets
        snippets = [mel_spectrogram[:, :, i:i+5] for i in range(mel_spectrogram.size(2) - 5 + 1)]
        
        batch_test_loss = torch.zeros([mel_spectrogram.shape[0]])
        for snippet in snippets:
            snippet = snippet.reshape(snippet.shape[0], -1)
            mu, logvar = model(snippet)
            loss = nll_gaussian(snippet, mu, logvar, eval=True)
            
            batch_test_loss += loss

        test_scores.extend(batch_test_loss)
        test_labels.extend(labels)

# Compute AUC score
auc_score = roc_auc_score(test_labels, test_scores)
print(f"AUC Score: {auc_score:.4f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(test_labels, test_scores)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.show()