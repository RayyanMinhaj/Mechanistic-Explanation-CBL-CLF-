import re
import torch
import clip
import numpy as np

# --- Configuration ---
INPUT_FILE = 'concept_naming_by_neuron_output_roles.txt'
OUTPUT_FILE = 'merged_concepts_output_roles.txt'
SIMILARITY_THRESHOLD = 0.95  # Cosine similarity threshold for merging
MODEL_NAME = 'ViT-B/32' # Using CLIP model now

def parse_concept_names(input_file):
    """
    Parses the input file to extract a mapping from neuron ID to its concept name.
    
    Returns:
        A dictionary {neuron_id (str): concept_name (str)}
    """
    neuron_concepts = {}
    try:
        with open(input_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return None

    # Regex to find a neuron block and capture the ID and the concept name
    # It handles optional quotes around the concept name.
    pattern = re.compile(r"Neuron ID: (\d+)\n(?:.|\n)*?MLLM Concept Name: [\"\']?(.*?)[\"\'\n$]", re.MULTILINE)
    
    matches = pattern.findall(content)
    
    for neuron_id, concept_name in matches:
        neuron_concepts[neuron_id] = concept_name.strip()
        
    if not neuron_concepts:
        print("Warning: No concept names were parsed. The input file might be empty or in an unexpected format.")

    return neuron_concepts

def merge_concepts():
    """
    Loads concept names, embeds them, merges highly similar ones,
    and writes the result to a new file.
    """
    print("1. Parsing concept names...")
    neuron_concepts = parse_concept_names(INPUT_FILE)
    if not neuron_concepts:
        print("Halting execution due to parsing failure.")
        return

    initial_concept_count = len(neuron_concepts)
    print(f"   Found {initial_concept_count} initial concepts.")

    # Separate neurons and concepts for processing
    neuron_ids = list(neuron_concepts.keys())
    concepts = list(neuron_concepts.values())

    print(f"\n2. Loading embedding model '{MODEL_NAME}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, preprocess = clip.load(MODEL_NAME, device=device)
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        print("Please ensure 'torch' and 'clip' are installed (`pip install torch openai-clip`).")
        return
        
    print("3. Encoding concepts into embeddings...")
    tokenized_concepts = clip.tokenize(concepts).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(tokenized_concepts)

    print("4. Calculating similarity and finding groups to merge...")
    # Compute cosine similarity between all pairs
    embeddings_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
    cosine_scores = torch.mm(embeddings_norm, embeddings_norm.t()).cpu().numpy()

    # Group concepts that are highly similar
    # We use a set to keep track of neurons that have already been merged
    merged_indices = set()
    concept_groups = []

    for i in range(len(concepts)):
        if i in merged_indices:
            continue
        
        # Find all other concepts that are similar to the current one
        similar_indices = np.where(cosine_scores[i] > SIMILARITY_THRESHOLD)[0]
        
        current_group = []
        for idx in similar_indices:
            if idx not in merged_indices:
                current_group.append(idx)
                merged_indices.add(idx)
        
        if current_group:
            concept_groups.append(current_group)

    final_concept_count = len(concept_groups)
    print(f"   Reduced to {final_concept_count} final concepts after merging.")

    print(f"\n5. Writing merged concepts to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"Original Concept Count: {initial_concept_count}\n")
        f.write(f"Merged Concept Count: {final_concept_count}\n")
        f.write(f"Similarity Threshold: {SIMILARITY_THRESHOLD}\n")
        f.write("="*50 + "\n\n")

        for group in concept_groups:
            # Get the original concepts and neuron IDs for this group
            grouped_concepts = [concepts[i] for i in group]
            grouped_neuron_ids = [neuron_ids[i] for i in group]
            
            # Create the new merged concept name
            # We use a set to avoid duplicating names if they were identical
            merged_name = " && ".join(sorted(list(set(grouped_concepts))))
            
            f.write(f"Merged Concept: {merged_name}\n")
            f.write(f"Original Neurons: {', '.join(sorted(grouped_neuron_ids, key=int))}\n")
            f.write("-" * 40 + "\n")

    print("\nProcessing complete.")

if __name__ == '__main__':
    merge_concepts()
