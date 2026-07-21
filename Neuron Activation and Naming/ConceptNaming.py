import torch
import torch.nn as nn
import clip
from PIL import Image
import json
import os
import glob
import base64
import requests
import random

# --- Configuration ---
ACTIVATED_NEURONS_JSON = 'activated_neurons_to_embeddings.json'
OUTPUT_FILE = 'concept_naming_by_neuron_output.txt'
CORREF_FILE = 'm2e2(2)/corref.txt'
TEXT_DATA_FILE = 'm2e2(2)/text_multi.json'
IMAGE_DIR = 'm2e2(2)/image/image/'
NUM_NEURONS_TO_PROCESS = 110  # Control how many neurons to process
MAX_IMAGES_PER_NEURON = 10 # Control max images/texts to send to MLLM to avoid huge prompts

OPENAI_API_KEY = ""



# --- 1. Helper functions to retrieve original data ---
def get_sentence_by_id(sentence_id, data):
    """Finds a sentence in the loaded text data by its ID."""
    for item in data:
        if item['sentence_id'] == sentence_id:
            return item['sentence']
    return None

def get_image_path_by_sentence_id(sentence_id, corref_data):
    """Finds an image filename from the corref data using the sentence ID."""
    for line in corref_data:
        parts = line.strip().split('\t')
        if len(parts) == 2 and parts[0] == sentence_id:
            return os.path.join(IMAGE_DIR, parts[1])
    return None

def encode_image(image_path):
  """Encodes an image file to a base64 string."""
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')









# --- 2. Main Processing Logic ---
def concept_naming_by_neuron():
    """
    Loads the neuron-to-embedding map, retrieves all associated texts and images
    for each neuron, and sends them to an MLLM to generate a concept name.
    """
    # Load reference data first to fail early if files are missing
    try:
        with open(ACTIVATED_NEURONS_JSON, 'r') as f:
            neuron_map = json.load(f)
        with open(TEXT_DATA_FILE, 'r') as f:
            text_data = json.load(f)
        with open(CORREF_FILE, 'r') as f:
            corref_data = f.readlines()
    except FileNotFoundError as e:
        print(f"Error: Could not open a required file: {e}. Please ensure all data files are present.")
        return

    # Open the output file
    with open(OUTPUT_FILE, 'w') as out_f:
        print(f"Processing {min(NUM_NEURONS_TO_PROCESS, len(neuron_map))} neurons...")

        # Get a list of neuron keys to process
        neurons_to_process = list(neuron_map.keys())[:NUM_NEURONS_TO_PROCESS]

        for neuron_idx_str in neurons_to_process:
            print(f"\n--- Processing Neuron {neuron_idx_str} ---")
            out_f.write(f"Neuron ID: {neuron_idx_str}\n")

            embedding_files = neuron_map[neuron_idx_str]
            
            # To avoid overly long prompts, we can sample from the embeddings if there are too many
            if len(embedding_files) > MAX_IMAGES_PER_NEURON:
                print(f"  Neuron has {len(embedding_files)} embeddings. Sampling {MAX_IMAGES_PER_NEURON} of them.")
                embedding_files = random.sample(embedding_files, MAX_IMAGES_PER_NEURON)

            texts = []
            image_paths = []
            base64_images = []

            for embed_filename in embedding_files:
                try:
                    # Extract embedding index from filename like 'embedding_123_EventType.pt'
                    embedding_index = int(embed_filename.split('_')[1])

                    # The embedding index corresponds to the line number in corref.txt
                    if embedding_index < len(corref_data):
                        corref_line = corref_data[embedding_index]
                        corref_parts = corref_line.strip().split('\t')
                        if len(corref_parts) >= 2:
                            sentence_id, image_filename = corref_parts[0], corref_parts[1]
                        else:
                            print(f"    Could not parse corref line {embedding_index}: {corref_line}. Skipping.")
                            continue
                    else:
                        print(f"    Embedding index {embedding_index} is out of bounds for corref file. Skipping.")
                        continue

                    # Get original text and image path
                    text = get_sentence_by_id(sentence_id, text_data)
                    image_path = os.path.join(IMAGE_DIR, image_filename)

                    if text and image_path and os.path.exists(image_path):
                        texts.append(text)
                        image_paths.append(image_path)
                        base64_images.append(encode_image(image_path))
                        out_f.write(f"  - Text: {text}\n")
                        out_f.write(f"  - Image: {image_path}\n")
                    else:
                        print(f"    Could not find text or image for sentence_id {sentence_id}. Skipping.")

                except (ValueError, IndexError) as e:
                    print(f"    Error parsing filename '{embed_filename}': {e}. Skipping.")
                    continue

            if not texts or not base64_images:
                print("  No valid text/image pairs found for this neuron. Skipping MLLM call.")
                out_f.write("MLLM Output: No valid data to process.\n")
                out_f.write("=" * 50 + "\n")
                continue

            # --- 3. MLLM Interaction ---
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }

            # The prompt asks the model to find a common concept among all provided examples
            prompt_text = """
            Analyze all the text-image pairs provided below. 
            What is the single, unifying concept or theme that connects all of them?
            Describe this concept in a short, descriptive phrase (e.g., "Protest gatherings," "Vehicle accidents," "Official meetings") which should be the only output.
            """

            # Construct the content payload with the main prompt and all images/texts
            content_payload = [{"type": "text", "text": prompt_text}]
            for i in range(len(texts)):
                content_payload.append({"type": "text", "text": f"Example {i+1} Text: {texts[i]}"})
                content_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_images[i]}"}
                })

            payload = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": content_payload}],
                "max_tokens": 30
            }

            mllm_output = "Error: MLLM call failed."
            try:
                print(f"  Sending {len(texts)} text/image pairs to MLLM...")
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                mllm_output = response.json()['choices'][0]['message']['content']
                print(f"  MLLM Response: {mllm_output.strip()}")

            except requests.exceptions.RequestException as e:
                print(f"    Error calling OpenAI API: {e}")
                if response:
                    print("Status:", response.status_code)
                    print("Body:", response.text)
                mllm_output = f"Error: API request failed. {e}"
            except (KeyError, IndexError) as e:
                print(f"    Error parsing OpenAI response: {e}")
                mllm_output = f"Error: Could not parse MLLM response. {response.text}"

            # --- 4. Write to output file ---
            out_f.write(f"MLLM Concept Name: {mllm_output.strip()}\n")
            out_f.write("=" * 50 + "\n")

    print(f"\nProcessing complete. Output written to {OUTPUT_FILE}")







if __name__ == '__main__':
    concept_naming_by_neuron()
