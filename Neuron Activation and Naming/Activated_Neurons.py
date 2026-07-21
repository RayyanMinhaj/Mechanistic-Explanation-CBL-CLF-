import torch
import torch.nn as nn
import os
import glob
import json
from collections import defaultdict

# --- Configuration ---
SAE_MODEL_PATH = 'SAE_artifacts/sae_model_fold_3.pt'
EMBEDDING_DIR = 'embedding_artifacts'
OUTPUT_JSON_FILE = 'activated_neurons_to_embeddings.json'

# --- 1. Define the Sparse Autoencoder Model (must match the training script) ---
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# --- 2. Main Processing Logic ---
def map_neurons_to_embeddings():
    """
    Processes all embeddings, finds the most activated neuron for each in the SAE,
    and saves a JSON file mapping each neuron to the list of embeddings that
    activated it most.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the trained SAE model
    input_dim = 1024  # 512 (text) + 512 (image)
    hidden_dim = 256 # Should match the trained model
    sae_model = SparseAutoencoder(input_dim, hidden_dim).to(device)

    if not os.path.exists(SAE_MODEL_PATH):
        print(f"Error: SAE model not found at {SAE_MODEL_PATH}")
        return

    try:
        sae_model.load_state_dict(torch.load(SAE_MODEL_PATH, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Error loading model state dictionary: {e}")
        return
        
    sae_model.eval()

    # Get list of all embedding files
    embedding_files = sorted(glob.glob(os.path.join(EMBEDDING_DIR, '*.pt')))
    if not embedding_files:
        print(f"Error: No embedding files found in {EMBEDDING_DIR}")
        return

    # Dictionary to hold the mapping from neuron index to embedding filenames
    # defaultdict simplifies appending to lists
    neuron_to_embeddings_map = defaultdict(list)

    print(f"Processing {len(embedding_files)} embeddings...")

    for embed_path in embedding_files:
        try:
            # Load the embedding and ensure it's a float tensor on the correct device
            embedding = torch.load(embed_path, weights_only=True).to(device).float()

            # Get hidden layer activations from the encoder
            with torch.no_grad():
                hidden_activations = sae_model.encoder(embedding)

            # Find the single most activated neuron (top k=1)
            # torch.argmax is perfect for this
            most_activated_neuron_idx = torch.argmax(hidden_activations.squeeze()).item()

            # Get the filename of the embedding
            embedding_filename = os.path.basename(embed_path)

            # Map the neuron to the embedding file
            neuron_to_embeddings_map[most_activated_neuron_idx].append(embedding_filename)

        except Exception as e:
            print(f"  Could not process {embed_path}. Error: {e}. Skipping.")
            continue

    # Convert defaultdict to a regular dict for cleaner JSON output
    neuron_to_embeddings_map_sorted = dict(sorted(neuron_to_embeddings_map.items()))

    # Save the resulting map to a JSON file
    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(neuron_to_embeddings_map_sorted, f, indent=4)

    print(f"\nProcessing complete. Output written to {OUTPUT_JSON_FILE}")

if __name__ == '__main__':
    map_neurons_to_embeddings()
