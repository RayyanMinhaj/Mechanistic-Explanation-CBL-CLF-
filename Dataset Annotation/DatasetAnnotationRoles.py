import torch
import json
import os
import base64
import requests
import random
import re
from collections import defaultdict

# --- Configuration ---
ACTIVATED_NEURONS_JSON = 'activated_neurons_to_embeddings_roles.json'
MERGED_CONCEPTS_FILE = 'merged_concepts_output_roles.txt'
CROPPED_IMAGE_DIR = 'cropped_images'
EMBEDDING_DIR = 'cropped_embedding_artifacts'
OUTPUT_JSON_FILE = 'dataset_annotation_output_roles.json'

# Limit the number of embeddings to process for a quicker run, set to None to process all
NUM_EMBEDDINGS_TO_PROCESS = None

OPENAI_API_KEY = ""



# --- 1. Helper functions ---

def encode_image(image_path):
  """Encodes an image file to a base64 string."""
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')



def parse_merged_concepts(filepath):
    """
    Parses the merged concepts text file into {concept_name: [neuron_ids]}.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    concept_to_neurons = {}
    pattern = re.compile(
        r"Merged Concept:\s*(.*?)\s*\nOriginal Neurons:\s*(.*?)(?:\n-+|\Z)",
        re.MULTILINE | re.DOTALL
    )
    for match in pattern.finditer(content):
        concept_name = match.group(1).strip()
        neurons_str = match.group(2).strip()
        neuron_ids = [n.strip() for n in neurons_str.split(',') if n.strip()]
        concept_to_neurons[concept_name] = neuron_ids

    return concept_to_neurons



def build_reverse_mappings():
    """Builds reverse mappings for quick lookups."""
    with open(ACTIVATED_NEURONS_JSON, 'r') as f:
        neuron_to_embeddings = json.load(f)
    concept_to_neurons = parse_merged_concepts(MERGED_CONCEPTS_FILE)

    embedding_to_neuron = {}
    for neuron, embeddings in neuron_to_embeddings.items():
        for embedding_file in embeddings:
            embedding_to_neuron[embedding_file] = neuron

    neuron_to_concept = {}
    for concept, neurons in concept_to_neurons.items():
        for neuron in neurons:
            neuron_to_concept[str(neuron)] = concept

    return embedding_to_neuron, neuron_to_concept





# --- 2. Main Annotation Logic ---

def annotate_dataset():
    """
    Annotates each cropped-role embedding with its concept and an MLLM-based validation.
    No text or corref needed — only the cropped image is evaluated against the concept.
    """
    print("1. Loading reference data and building mappings...")
    try:
        embedding_to_neuron, neuron_to_concept = build_reverse_mappings()
    except FileNotFoundError as e:
        print(f"Error: Could not open a required file: {e}. Please ensure all data files are present.")
        return

    all_embedding_files = list(embedding_to_neuron.keys())
    if NUM_EMBEDDINGS_TO_PROCESS is not None:
        print(f"Sampling {NUM_EMBEDDINGS_TO_PROCESS} embeddings to process.")
        files_to_process = random.sample(all_embedding_files, min(NUM_EMBEDDINGS_TO_PROCESS, len(all_embedding_files)))
    else:
        files_to_process = all_embedding_files

    final_annotations = {}
    print(f"\n2. Starting annotation for {len(files_to_process)} embeddings...")

    for i, embed_filename in enumerate(files_to_process):
        print(f"\n--- Processing item {i+1}/{len(files_to_process)}: {embed_filename} ---")

        # Find neuron and concept
        neuron_id = embedding_to_neuron.get(embed_filename)
        if not neuron_id:
            print(f"  Warning: Neuron not found for {embed_filename}. Skipping.")
            continue

        concept_name = neuron_to_concept.get(neuron_id)
        if not concept_name:
            print(f"  Warning: Concept not found for neuron {neuron_id}. Skipping.")
            continue

        print(f"  - Activated Neuron: {neuron_id}")
        print(f"  - Mapped Concept: '{concept_name}'")

        # Derive the cropped image path directly from the embedding filename
        image_filename = embed_filename.replace('.pt', '.jpg')
        image_path = os.path.join(CROPPED_IMAGE_DIR, image_filename)

        if not os.path.exists(image_path):
            print(f"  Warning: Cropped image not found at {image_path}. Skipping.")
            continue

        # --- 3. MLLM Validation (image-only) ---
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        base64_image = encode_image(image_path)

        prompt_text = f"""
        You are a data annotator. Your task is to determine if a given image matches a specific concept.

        Concept: "{concept_name}"

        Does the image accurately represent the concept?
        Respond with only '1' for YES or '0' for NO. Do not provide any other text or explanation.
        """

        content_payload = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]

        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": content_payload}],
            "max_tokens": 5
        }

        mllm_decision = -1
        try:
            print("  Sending to MLLM for validation...")
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            output = response.json()['choices'][0]['message']['content'].strip()

            if output in ['0', '1']:
                mllm_decision = int(output)
                print(f"  MLLM Decision: {mllm_decision}")
            else:
                print(f"  Warning: MLLM returned an unexpected value: '{output}'")

        except requests.exceptions.RequestException as e:
            print(f"  Error calling OpenAI API: {e}")
        except (KeyError, IndexError) as e:
            print(f"  Error parsing OpenAI response: {e}")

        # --- 4. Store Annotation ---
        final_annotations[embed_filename] = {
            "concept_name": concept_name,
            "image_path": image_path,
            "follows_concept": mllm_decision
        }

    # --- 5. Save Final Output ---
    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(final_annotations, f, indent=4)

    print(f"\n\nAnnotation complete. Output written to {OUTPUT_JSON_FILE}")






if __name__ == '__main__':
    annotate_dataset()
