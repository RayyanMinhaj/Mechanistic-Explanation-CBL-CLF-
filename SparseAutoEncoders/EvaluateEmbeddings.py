import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import re

# Function to extract class name from filename
def get_class_from_filename(filename):
    match = re.search(r'embedding_\d+_(.*)\.pt', filename)
    if match:
        return match.group(1)
    return None





# Custom Dataset
class EmbeddingDataset(Dataset):
    def __init__(self, embedding_files, class_to_idx):
        self.embedding_files = embedding_files
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):
        embedding_path = self.embedding_files[idx]
        embedding = torch.load(embedding_path, weights_only=True).float()
        class_name = get_class_from_filename(os.path.basename(embedding_path))
        label = self.class_to_idx[class_name]
        return embedding, label

# Prepare data
embedding_files = glob.glob('embedding_artifacts/*.pt')
class_names = sorted(list(set(get_class_from_filename(os.path.basename(f)) for f in embedding_files)))
class_to_idx = {name: i for i, name in enumerate(class_names)}
num_classes = len(class_names)

# Model
class Classifier(nn.Module):
    def __init__(self, hidden_weights, num_classes):
        super(Classifier, self).__init__()
        # The "multiplication" is a linear layer.
        # The weights are the transpose of the hidden layer weights from the SAE
        self.linear = nn.Linear(hidden_weights.shape[0], num_classes)
        # We can't directly use the weights if the dimensions don't match num_classes.
        # So we'll use the features from the multiplication as input to a new classifier.
        self.hidden_weights = nn.Parameter(hidden_weights.t(), requires_grad=False)

    def forward(self, x):
        # Project embeddings into the hidden space
        hidden_features = torch.matmul(x, self.hidden_weights)
        output = self.linear(hidden_features)
        return output

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# K-fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_index, val_index) in enumerate(kf.split(embedding_files)):
    print(f"--- Fold {fold+1} ---")

    # Load the hidden layer weights for the current fold
    try:
        hidden_layer_weights_path = f'SAE_artifacts/hidden_layer_weights_fold_{fold+1}.pt'
        hidden_layer_weights = torch.load(hidden_layer_weights_path, weights_only=True).float()
        print(f"Loaded hidden layer weights from {hidden_layer_weights_path}")
    except FileNotFoundError:
        print(f"Could not find '{hidden_layer_weights_path}'. Using random weights.")
        # Fallback to random weights if the file for the fold doesn't exist.
        hidden_size = 2048 # Example hidden size
        embedding_size = 768
        hidden_layer_weights = torch.randn(hidden_size, embedding_size)

    train_files = [embedding_files[i] for i in train_index]
    val_files = [embedding_files[i] for i in val_index]

    train_dataset = EmbeddingDataset(train_files, class_to_idx)
    val_dataset = EmbeddingDataset(val_files, class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Classifier(hidden_layer_weights, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(10): # 10 epochs for demonstration
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings.squeeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings.squeeze(1))
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    fold_accuracies.append(accuracy)
    print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")

print(f"\nAverage K-fold Accuracy: {np.mean(fold_accuracies):.4f}")
