import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import os
import glob
import numpy as np

# --- Configuration ---
EMBEDDING_DIR = 'cropped_embedding_artifacts'
OUTPUT_DIR = 'SAE_cropped_artifacts'
INPUT_DIM = 512   # Image-only CLIP feature dimension (ViT-B/32)
HIDDEN_DIM = 256  # Hidden representation dimension
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 16
SPARSITY_TARGET = 0.05  # Desired level of sparsity
SPARSITY_WEIGHT = 1e-3   # How much to penalize non-sparsity
K_FOLDS = 3



class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            # No activation function here for reconstruction
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded



class EmbeddingDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        embedding = torch.load(self.file_paths[idx], weights_only=True)
        return embedding.squeeze(0) # Remove batch dimension from saved file



def sparsity_loss(activations, target, weight):
    avg_activation = torch.mean(activations, dim=0)
    kl_div = torch.sum(target * torch.log(target / avg_activation) +
                       (1 - target) * torch.log((1 - target) / (1 - avg_activation)))
    return weight * kl_div






def train_and_evaluate():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embedding_files = glob.glob(os.path.join(EMBEDDING_DIR, '*.pt'))
    if not embedding_files:
        print("No embedding files found. Please run 01b_crop_and_embed_images.py first.")
        return
        
    embedding_files = np.array(embedding_files)

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_index, val_index) in enumerate(kf.split(embedding_files)):
        print(f"--- Fold {fold+1}/{K_FOLDS} ---")

        train_files = embedding_files[train_index]
        val_files = embedding_files[val_index]

        train_dataset = EmbeddingDataset(train_files)
        val_dataset = EmbeddingDataset(val_files)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        model = SparseAutoencoder(INPUT_DIM, HIDDEN_DIM).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            total_train_loss = 0
            for data in train_loader:
                data = data.to(device).float()  # Move data to device and cast to float
                optimizer.zero_grad()
                reconstructed, encoded = model(data)
                
                recon_loss = criterion(reconstructed, data)
                sparse_loss = sparsity_loss(encoded, SPARSITY_TARGET, SPARSITY_WEIGHT)
                loss = recon_loss + sparse_loss

                loss.backward()
                optimizer.step()
                total_train_loss += recon_loss.item() # We only care about reconstruction error for reporting
            
            avg_train_loss = total_train_loss / len(train_loader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Train Reconstruction Loss: {avg_train_loss:.6f}")

        # Evaluation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device).float()  # Move data to device and cast to float
                reconstructed, _ = model(data)
                loss = criterion(reconstructed, data)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Fold {fold+1} Validation Reconstruction Error (MSE): {avg_val_loss:.6f}")
        fold_results.append(avg_val_loss)

        # Save model and hidden layer for this fold
        model_path = os.path.join(OUTPUT_DIR, f'sae_model_fold_{fold+1}.pt')
        hidden_layer_path = os.path.join(OUTPUT_DIR, f'hidden_layer_weights_fold_{fold+1}.pt')
        
        torch.save(model.state_dict(), model_path)
        torch.save(model.encoder[0].weight.data, hidden_layer_path)
        print(f"Saved model to {model_path}")
        print(f"Saved hidden layer weights to {hidden_layer_path}")


    print("\n--- K-Fold Cross-Validation Summary ---")
    for i, loss in enumerate(fold_results):
        print(f"Fold {i+1}: {loss:.6f}")
    print(f"Average Validation Reconstruction Error (MSE): {np.mean(fold_results):.6f} (+/- {np.std(fold_results):.6f})")


if __name__ == "__main__":
    train_and_evaluate()
