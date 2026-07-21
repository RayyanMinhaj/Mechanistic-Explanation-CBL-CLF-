"""
========================================================================
CONCEPT BOTTLENECK MODEL (CBM) IMPLEMENTATION
========================================================================
Here we implement a three-stage CBM for multi-label concept prediction:

Stage 1: Train Concept Bottleneck Layer (CBL)
    - Maps 1024-dim embeddings → K concept logits
    - Uses masked BCE loss (only on annotated pairs)
    - Weighted by concept class imbalance

Stage 2: Extract Concept Features
    - Get concept logits from trained CBL for all samples
    - These will be input to the classifier

Stage 3: Train Sparse Linear Classifier
    - Maps K concept logits → C class predictions
    - Uses elastic-net penalty for sparsity
    - Z-normalizes concept logits before classification

Uses 3-fold cross-validation to evaluate performance.
========================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================================================
# CONFIGURATION
# ========================================================================
DATASET_ANNOTATION_FILE = 'dataset_annotation_output.json'
MERGED_CONCEPTS_FILE = 'merged_concepts_output.json'
EMBEDDING_DIR = 'embedding_artifacts'
OUTPUT_DIR = 'CBM_artifacts'




# Hyperparameters
K_FOLDS = 3
EPOCHS_CBL = 30  # Training epochs for Concept Bottleneck Layer
EPOCHS_CLF = 20  # Training epochs for Classifier
BATCH_SIZE = 16
LEARNING_RATE_CBL = 1e-3
LEARNING_RATE_CLF = 1e-3
LAMBDA_CLF = 1e-2  # Elastic-net regularization weight
ALPHA_ELASTIC = 0.99  # Elastic-net parameter (0.99 emphasizes L1 sparsity)






# ========================================================================
# COMPONENT 1: CONCEPT BOTTLENECK LAYER (CBL)
# ========================================================================
class ConceptBottleneckLayer(nn.Module):
    """
    Maps input features to concept predictions (multi-label).
    
    Input: 1024-dimensional text+image embedding
    Output: K-dimensional concept logits (one per concept)
    
    This is the FROZEN backbone in the paper becomes the input here,
    and this layer predicts which K concepts are present.
    """
    
    def __init__(self, input_dim, num_concepts):
        super(ConceptBottleneckLayer, self).__init__()
        # Simple 2-layer MLP
        self.fc1 = nn.Linear(input_dim, 512)  # Project to hidden space
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_concepts)  # Project to concept space
    
    
    
    
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] - embeddings
        Returns:
            concept_logits: [batch_size, num_concepts] - before sigmoid
        """
        x = self.fc1(x)
        x = self.relu(x)
        concept_logits = self.fc2(x)
        return concept_logits










# ========================================================================
# COMPONENT 2: SPARSE LINEAR CLASSIFIER
# ========================================================================
class SparseLinearClassifier(nn.Module):
    """
    Maps concept predictions to final class predictions.
    
    Input: K-dimensional concept logits (z-normalized)
    Output: C-dimensional class predictions
    
    Uses elastic-net penalty to encourage sparsity in the weight matrix,
    ensuring only important concepts contribute to the final prediction.
    """
    def __init__(self, num_concepts, num_classes):
        super(SparseLinearClassifier, self).__init__()
        self.fc = nn.Linear(num_concepts, num_classes)
    





    def forward(self, concept_logits):
        """
        Args:
            concept_logits: [batch_size, num_concepts] - z-normalized
        Returns:
            class_logits: [batch_size, num_classes]
        """
        class_logits = self.fc(concept_logits)
        return class_logits








# ========================================================================
# MASKED BCE LOSS
# ========================================================================
class MaskedBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss that handles incomplete annotations.
    
    - Only computes loss on annotated (embedding, concept) pairs
    - Ignores entries marked as -1 (not annotated)
    - Weights each concept by class imbalance ratio
    
    This is critical because not every image-concept pair is labeled.
    """
    
    
    def __init__(self, concept_weights):
        super(MaskedBCELoss, self).__init__()
        self.register_buffer('concept_weights', concept_weights)
    





    def forward(self, concept_logits, concept_labels, mask):
        """
        Args:
            concept_logits: [batch_size, num_concepts] - raw logits
            concept_labels: [batch_size, num_concepts] - {0, 1} for annotated
            mask: [batch_size, num_concepts] - 1 if annotated, 0 if not
        Returns:
            loss: scalar - masked BCE averaged over annotated pairs
        """
        # Binary cross-entropy with logits (applies sigmoid internally)
        bce = nn.functional.binary_cross_entropy_with_logits(
            concept_logits, concept_labels, reduction='none'
        )
        


        # Apply mask (ignore non-annotated pairs)
        # and weight by concept class imbalance
        weighted_loss = bce * mask * self.concept_weights.unsqueeze(0)
        


        # Average only over annotated pairs
        num_annotated = mask.sum()
        if num_annotated > 0:
            return weighted_loss.sum() / num_annotated
        else:
            return torch.tensor(0.0, device=bce.device)








# ========================================================================
# DATA LOADING AND PREPROCESSING
# ========================================================================
def load_and_prepare_data():
    """
    Load embeddings and create concept label matrix.
    
    Returns:
        embeddings_array: [N, 1024] - text+image embeddings
        concept_labels_array: [N, K] - ternary concept labels {-1, 0, 1}
        class_labels_array: [N] - class indices
        concept_names: list of K concept names
        class_labels: list of C class names
        concept_to_idx: dict mapping concept name to column index
        class_to_idx: dict mapping class name to index
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    
    
    
    # Load annotations and merged concepts
    print("\nLoading dataset annotation...")
    with open(DATASET_ANNOTATION_FILE, 'r') as f:
        annotations = json.load(f)
    
    print("Loading merged concepts...")
    with open(MERGED_CONCEPTS_FILE, 'r') as f:
        merged_concepts = json.load(f)
    
    # Create mappings
    concept_names = list(merged_concepts.keys())
    num_concepts = len(concept_names)
    concept_to_idx = {c: i for i, c in enumerate(concept_names)}
    
    # Extract class labels from embedding filenames
    # Example: embedding_0_Justice-Arrest-Jail.pt → "Justice-Arrest-Jail"
    class_labels_set = set()
    for embed_name in annotations.keys():
        parts = embed_name.split('_')
        if len(parts) >= 3:
            event_type = '_'.join(parts[2:]).replace('.pt', '')
            class_labels_set.add(event_type)
    
    class_labels = sorted(list(class_labels_set))
    num_classes = len(class_labels)
    class_to_idx = {c: i for i, c in enumerate(class_labels)}
    
    print(f"\nDataset Summary:")
    print(f"  Number of concepts (K): {num_concepts}")
    print(f"  Number of event classes (C): {num_classes}")
    print(f"  Number of samples (N): {len(annotations)}")
    
    # Load embeddings and create concept matrix
    print(f"\nLoading {len(annotations)} embeddings...")
    embeddings_list = []
    concept_labels_list = []
    class_labels_list = []
    
    for embed_idx, (embed_name, data) in enumerate(annotations.items()):
        if embed_idx % 100 == 0 and embed_idx > 0:
            print(f"  Processed {embed_idx}/{len(annotations)} embeddings...")
        
        try:
            # ==========================================
            # LOAD EMBEDDING
            # ==========================================
            embed_path = os.path.join(EMBEDDING_DIR, embed_name)
            embedding = torch.load(embed_path, weights_only=True).squeeze().cpu()
            embeddings_list.append(embedding.numpy())
            
            # ==========================================
            # EXTRACT CLASS LABEL
            # ==========================================
            # Get event type from embedding filename
            parts = embed_name.split('_')
            event_type = '_'.join(parts[2:]).replace('.pt', '')
            class_labels_list.append(class_to_idx[event_type])
            
            # ==========================================
            # CREATE CONCEPT LABEL VECTOR
            # ==========================================
            # Initialize with -1 (not annotated)
            concept_vector = np.full(num_concepts, -1, dtype=np.float32)
            
            # Get the concept name for this embedding
            concept_name = data['concept_name']
            follows_concept = data['follows_concept']
            
            # Check if this concept is in our merged concepts list
            if concept_name in concept_to_idx:
                concept_idx = concept_to_idx[concept_name]
                # Set to 1 (present) or 0 (absent)
                if follows_concept == 1:
                    concept_vector[concept_idx] = 1.0
                elif follows_concept == 0:
                    concept_vector[concept_idx] = 0.0
                # All other concepts remain -1 (not annotated for this sample)
            
            concept_labels_list.append(concept_vector)
        
        except Exception as e:
            print(f"    Error loading {embed_name}: {e}. Skipping.")
            continue
    
    # Convert to numpy arrays
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    concept_labels_array = np.array(concept_labels_list, dtype=np.float32)
    class_labels_array = np.array(class_labels_list, dtype=np.int64)
    
    print(f"\nData loaded successfully!")
    print(f"  Embeddings shape: {embeddings_array.shape}")
    print(f"  Concept labels shape: {concept_labels_array.shape}")
    print(f"  Class labels shape: {class_labels_array.shape}")
    
    return (embeddings_array, concept_labels_array, class_labels_array,
            concept_names, class_labels, concept_to_idx, class_to_idx)









# ========================================================================
# CALCULATE CONCEPT WEIGHTS
# ========================================================================
def calculate_concept_weights(concept_labels_array):
    """
    Calculate weight for each concept based on class imbalance.
    
    Rare concepts (fewer positive examples) get higher weight in the loss,
    helping the model pay more attention to uncommon but important concepts.
    
    Args:
        concept_labels_array: [N, K] - concept labels {-1, 0, 1}
    
    Returns:
        weights: [K] - weight for each concept
    """
    num_concepts = concept_labels_array.shape[1]
    weights = []
    
    for k in range(num_concepts):
        # Find all annotated examples for this concept (not -1)
        mask = concept_labels_array[:, k] >= 0
        annotated = concept_labels_array[mask, k]
        
        if len(annotated) > 0:
            # Proportion of positive examples
            pos_ratio = annotated.mean()
            # Weight is balanced between positive and negative classes
            weight = pos_ratio * (1 - pos_ratio)
            if weight == 0:
                weight = 1.0
        else:
            weight = 1.0
        
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum() * len(weights)
    return torch.from_numpy(weights)






# ========================================================================
# SAVE CONFUSION MATRIX AS IMAGE
# ========================================================================
def save_confusion_matrix_image(cm, class_labels, fold_num, output_dir):
    """
    Generate and save a confusion matrix heatmap as an image.
    
    Args:
        cm: confusion matrix [num_classes, num_classes]
        class_labels: list of class names
        fold_num: fold number for naming
        output_dir: directory to save the image
    """
    # Create confusion matrix directory if it doesn't exist
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    if not os.path.exists(cm_dir):
        os.makedirs(cm_dir)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - Fold {fold_num} (Classifier Layer)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save image
    image_path = os.path.join(cm_dir, f'confusion_matrix_fold_{fold_num}.png')
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Confusion matrix saved to: {image_path}")


# ========================================================================
# TRAIN CONCEPT BOTTLENECK LAYER (STAGE 1)
# ========================================================================
def train_cbl(model_cbl, train_loader, val_loader, device, num_epochs, 
              learning_rate, concept_weights):
    """
    Train the Concept Bottleneck Layer (CBL).
    
    The CBL learns to predict which K concepts are present in each embedding.
    Uses masked BCE loss to handle partial annotations (some concept-embedding
    pairs are not labeled).
    
    Args:
        model_cbl: ConceptBottleneckLayer model
        train_loader: DataLoader for training embeddings and concepts
        val_loader: DataLoader for validation
        device: torch device (cuda or cpu)
        num_epochs: number of training epochs
        learning_rate: optimizer learning rate
        concept_weights: [K] - weight for each concept
    
    Returns:
        model_cbl: trained model
    """
    optimizer = optim.Adam(model_cbl.parameters(), lr=learning_rate)
    criterion = MaskedBCELoss(concept_weights.to(device))
    
    print(f"\nTraining CBL for {num_epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # ==========================================
        # TRAINING PHASE
        # ==========================================
        model_cbl.train()
        train_loss = 0.0
        
        for embeddings, concept_labels in train_loader:
            embeddings = embeddings.to(device)
            concept_labels = concept_labels.to(device)
            
            # Create mask: 1 where annotated, 0 where -1 (not annotated)
            mask = (concept_labels >= 0).float()
            
            # Forward pass: embeddings → concept logits
            concept_logits = model_cbl(embeddings)
            
            # Compute masked BCE loss
            loss = criterion(concept_logits, concept_labels, mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # ==========================================
        # VALIDATION PHASE
        # ==========================================
        model_cbl.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for embeddings, concept_labels in val_loader:
                embeddings = embeddings.to(device)
                concept_labels = concept_labels.to(device)
                
                mask = (concept_labels >= 0).float()
                concept_logits = model_cbl(embeddings)
                loss = criterion(concept_logits, concept_labels, mask)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return model_cbl











# ========================================================================
# EXTRACT CONCEPT LOGITS (STAGE 2)
# ========================================================================
def get_concept_logits(model_cbl, data_loader, device):
    """
    Extract concept logits for all samples using the trained CBL.
    
    These logits will be input to the sparse linear classifier.
    
    Args:
        model_cbl: trained ConceptBottleneckLayer
        data_loader: DataLoader with embeddings
        device: torch device
    
    Returns:
        all_logits: [N, K] - concept logits for all samples
    """
    model_cbl.eval()
    all_logits = []
    
    with torch.no_grad():
        for embeddings, _ in data_loader:
            embeddings = embeddings.to(device)
            concept_logits = model_cbl(embeddings)
            all_logits.append(concept_logits.cpu().numpy())
    
    return np.concatenate(all_logits, axis=0)










# ========================================================================
# TRAIN CLASSIFIER (STAGE 3)
# ========================================================================
def train_classifier(model_clf, train_logits, train_labels, val_logits, val_labels,
                     device, num_epochs, learning_rate, lambda_clf, alpha):
    """
    Train the sparse linear classifier on concept logits (CBL frozen).
    
    Maps K concept logits → C class predictions using a sparse linear layer.
    Uses elastic-net penalty to encourage sparsity (only important concepts
    contribute to the final prediction).
    
    Args:
        model_clf: SparseLinearClassifier model
        train_logits: [N_train, K] - concept logits (from trained CBL)
        train_labels: [N_train] - class indices
        val_logits: [N_val, K] - concept logits for validation
        val_labels: [N_val] - class indices for validation
        device: torch device
        num_epochs: training epochs
        learning_rate: optimizer learning rate
        lambda_clf: elastic-net regularization weight
        alpha: elastic-net parameter (0.99 = emphasize L1)
    
    Returns:
        model_clf: trained classifier
        scaler: StandardScaler used for z-normalization
    """
    # ==========================================
    # Z-NORMALIZE CONCEPT LOGITS
    # ==========================================
    # Important: standardize concept logits before feeding to classifier
    scaler = StandardScaler()
    train_logits_normalized = scaler.fit_transform(train_logits)
    val_logits_normalized = scaler.transform(val_logits)
    
    # Convert to tensors
    train_logits_t = torch.from_numpy(train_logits_normalized).float().to(device)
    train_labels_t = torch.from_numpy(train_labels).long().to(device)
    val_logits_t = torch.from_numpy(val_logits_normalized).float().to(device)
    val_labels_t = torch.from_numpy(val_labels).long().to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(train_logits_t, train_labels_t)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_logits_t, val_labels_t), batch_size=16)
    
    optimizer = optim.Adam(model_clf.parameters(), lr=learning_rate)
    ce_loss = nn.CrossEntropyLoss()
    
    print(f"\nTraining Classifier for {num_epochs} epochs...")
    best_val_f1 = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # ==========================================
        # TRAINING PHASE
        # ==========================================
        model_clf.train()
        train_loss = 0.0
        
        for logits, labels in train_loader:
            # Forward pass: concept logits → class predictions
            class_logits = model_clf(logits)
            
            # Cross-entropy loss
            ce_loss_val = ce_loss(class_logits, labels)
            
            # Elastic-net penalty for sparsity
            w = model_clf.fc.weight
            l1_penalty = torch.norm(w, p=1)  # L1 norm
            l2_penalty = torch.norm(w, p=2)  # L2 norm
            elastic_penalty = lambda_clf * ((1 - alpha) * 0.5 * l2_penalty**2 + alpha * l1_penalty)
            
            # Total loss
            loss = ce_loss_val + elastic_penalty
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # ==========================================
        # VALIDATION PHASE
        # ==========================================
        model_clf.eval()
        val_loss = 0.0
        val_preds = []
        
        with torch.no_grad():
            for logits, labels in val_loader:
                class_logits = model_clf(logits)
                ce_loss_val = ce_loss(class_logits, labels)
                val_loss += ce_loss_val.item()
                
                # Get predictions
                preds = torch.argmax(class_logits, dim=1)
                val_preds.append(preds.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_preds = np.concatenate(val_preds)
        _, _, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro', zero_division=0)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model_clf.state_dict().copy()
    
    # Restore best model state
    if best_model_state is not None:
        model_clf.load_state_dict(best_model_state)
    
    return model_clf, scaler







# ========================================================================
# K-FOLD CROSS-VALIDATION TRAINING
# ========================================================================
def train_with_kfold(embeddings_array, concept_labels_array, class_labels_array,
                     concept_names, class_labels):
    """
    Train CBM using K-fold cross-validation.
    
    For each fold:
    1. Train CBL to predict concepts from embeddings
    2. Extract concept logits
    3. Train classifier to map concepts to class labels
    4. Evaluate accuracy on validation set
    
    Args:
        embeddings_array: [N, 1024] - text+image embeddings
        concept_labels_array: [N, K] - concept labels {-1, 0, 1}
        class_labels_array: [N] - class indices
        concept_names: list of K concept names
        class_labels: list of C class names
    
    Returns:
        fold_results: list of dicts with precision, recall, f1 per fold
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    num_concepts = len(concept_names)
    num_classes = len(class_labels)
    input_dim = embeddings_array.shape[1]  # 1024
    
    # Initialize K-fold splitter
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(embeddings_array)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold+1}/{K_FOLDS}")
        print(f"{'='*70}")
        
        # ==========================================
        # SPLIT DATA FOR THIS FOLD
        # ==========================================
        train_embeddings = embeddings_array[train_idx]
        train_concepts = concept_labels_array[train_idx]
        train_classes = class_labels_array[train_idx]
        
        val_embeddings = embeddings_array[val_idx]
        val_concepts = concept_labels_array[val_idx]
        val_classes = class_labels_array[val_idx]
        
        print(f"Train samples: {len(train_embeddings)}, Val samples: {len(val_embeddings)}")
        
        # Convert to tensors
        train_emb_t = torch.from_numpy(train_embeddings).float()
        train_conc_t = torch.from_numpy(train_concepts).float()
        val_emb_t = torch.from_numpy(val_embeddings).float()
        val_conc_t = torch.from_numpy(val_concepts).float()
        
        # Create datasets and loaders
        train_dataset = TensorDataset(train_emb_t, train_conc_t)
        val_dataset = TensorDataset(val_emb_t, val_conc_t)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # ==========================================
        # STAGE 1: TRAIN CONCEPT BOTTLENECK LAYER
        # ==========================================
        print("\n[STAGE 1/3] Training Concept Bottleneck Layer...")
        print("-" * 70)
        
        model_cbl = ConceptBottleneckLayer(input_dim, num_concepts).to(device)
        concept_weights = calculate_concept_weights(train_concepts)
        
        model_cbl = train_cbl(model_cbl, train_loader, val_loader, device,
                              EPOCHS_CBL, LEARNING_RATE_CBL, concept_weights)
        
        # ==========================================
        # STAGE 2: EXTRACT CONCEPT LOGITS
        # ==========================================
        print("\n[STAGE 2/3] Extracting concept logits...")
        print("-" * 70)
        
        train_concept_logits = get_concept_logits(model_cbl, train_loader, device)
        val_concept_logits = get_concept_logits(model_cbl, val_loader, device)
        
        print(f"Train concept logits shape: {train_concept_logits.shape}")
        print(f"Val concept logits shape: {val_concept_logits.shape}")
        
        # ==========================================
        # STAGE 3: TRAIN SPARSE LINEAR CLASSIFIER
        # ==========================================
        print("\n[STAGE 3/3] Training Sparse Linear Classifier...")
        print("-" * 70)
        
        model_clf = SparseLinearClassifier(num_concepts, num_classes).to(device)
        model_clf, scaler = train_classifier(model_clf, train_concept_logits, train_classes,
                                             val_concept_logits, val_classes, device,
                                             EPOCHS_CLF, LEARNING_RATE_CLF, LAMBDA_CLF, ALPHA_ELASTIC)
        
        # ==========================================
        # FOLD EVALUATION
        # ==========================================
        print("\n[EVALUATION]")
        print("-" * 70)
        
        model_cbl.eval()
        model_clf.eval()
        
        val_preds = []
        with torch.no_grad():
            # Get concept logits for validation set
            val_emb_t_device = val_emb_t.to(device)
            concept_logits = model_cbl(val_emb_t_device)
            
            # Z-normalize
            concept_logits_norm = torch.from_numpy(
                scaler.transform(concept_logits.cpu().numpy())
            ).float().to(device)
            
            # Get class predictions
            class_logits = model_clf(concept_logits_norm)
            val_preds = torch.argmax(class_logits, dim=1).cpu().numpy()
        
        # Calculate precision, recall, F1
        fold_precision, fold_recall, fold_f1, _ = precision_recall_fscore_support(
            val_classes, val_preds, average='macro', zero_division=0
        )
        fold_results.append({
            'precision': fold_precision,
            'recall': fold_recall,
            'f1': fold_f1
        })
        
        print(f"\nFold {fold+1} Validation Metrics:")
        print(f"  Precision (macro): {fold_precision:.4f}")
        print(f"  Recall (macro):    {fold_recall:.4f}")
        print(f"  F1 (macro):        {fold_f1:.4f}")
        
        # ==========================================
        # CONFUSION MATRIX FOR CLASSIFIER LAYER
        # ==========================================
        print(f"\n[Generating Confusion Matrix for Classifier Layer]")
        print("-" * 70)
        cm = confusion_matrix(val_classes, val_preds, labels=np.arange(len(class_labels)))
        
        # Save confusion matrix as image
        save_confusion_matrix_image(cm, class_labels, fold+1, OUTPUT_DIR)
        
        # Per-class precision, recall, F1
        print(f"\n[Per-Class Precision, Recall, F1]")
        p, r, f, s = precision_recall_fscore_support(
            val_classes, val_preds, labels=np.arange(len(class_labels)), zero_division=0
        )
        print(f"  {'Class':>25}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}  {'Support':>7}")
        print(f"  {'-'*25}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*7}")
        for i, class_name in enumerate(class_labels):
            print(f"  {class_name[:25]:>25}  {p[i]:>9.4f}  {r[i]:>9.4f}  {f[i]:>9.4f}  {s[i]:>7d}")
        
        # ==========================================
        # SAVE FOLD MODELS
        # ==========================================
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        torch.save(model_cbl.state_dict(), os.path.join(OUTPUT_DIR, f'cbl_fold_{fold+1}.pt'))
        torch.save(model_clf.state_dict(), os.path.join(OUTPUT_DIR, f'clf_fold_{fold+1}.pt'))
        print(f"\nSaved models to {OUTPUT_DIR}/")
    
    # ==========================================
    # OVERALL K-FOLD RESULTS
    # ==========================================
    print(f"\n{'='*70}")
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    
    for i, res in enumerate(fold_results):
        print(f"Fold {i+1}:  Precision={res['precision']:.4f}  Recall={res['recall']:.4f}  F1={res['f1']:.4f}")
    
    mean_p = np.mean([r['precision'] for r in fold_results])
    std_p  = np.std([r['precision'] for r in fold_results])
    mean_r = np.mean([r['recall'] for r in fold_results])
    std_r  = np.std([r['recall'] for r in fold_results])
    mean_f = np.mean([r['f1'] for r in fold_results])
    std_f  = np.std([r['f1'] for r in fold_results])
    
    print(f"\nMacro-Averaged Metrics ({K_FOLDS}-Fold CV):")
    print(f"  Precision: {mean_p:.4f} ± {std_p:.4f}")
    print(f"  Recall:    {mean_r:.4f} ± {std_r:.4f}")
    print(f"  F1:        {mean_f:.4f} ± {std_f:.4f}")
    print(f"{'='*70}")
    
    return fold_results









# ========================================================================
# MAINNNNNNN 
# ========================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("CONCEPT BOTTLENECK MODEL (CBM) - TRAINING")
    print("="*70)
    
    # Load data
    (embeddings_array, concept_labels_array, class_labels_array,
     concept_names, class_labels, concept_to_idx, class_to_idx) = load_and_prepare_data()
    
    # Train with K-fold cross-validation
    results = train_with_kfold(embeddings_array, concept_labels_array, class_labels_array,
                               concept_names, class_labels)
    
    print("\n✓ Training completed successfully!")
