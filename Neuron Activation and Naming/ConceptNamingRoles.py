import json
import os
import base64
import requests
import random

# --- Configuration ---
ACTIVATED_NEURONS_JSON = 'activated_neurons_to_embeddings_roles.json'
OUTPUT_FILE = 'concept_naming_by_neuron_output_roles.txt'
CROPPED_IMAGE_DIR = 'cropped_images'
NUM_NEURONS_TO_PROCESS = 500  # Control how many neurons to process
MAX_IMAGES_PER_NEURON = 50    # Control max images to send to MLLM

# Reuse API key from original script configuration
OPENAI_API_KEY = ""


# --- 1. Helper function ---
def encode_image(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# --- 2. Main Processing Logic ---
def concept_naming_by_neuron():
    """
    Loads the neuron-to-embedding map, retrieves all associated images (crops)
    for each neuron, and sends them to an MLLM to generate a concept name.
    """
    try:
        with open(ACTIVATED_NEURONS_JSON, 'r') as f:
            neuron_map = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not open {ACTIVATED_NEURONS_JSON}: {e}")
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
            
            # Sample from the embeddings if there are too many to avoid overly long prompts
            if len(embedding_files) > MAX_IMAGES_PER_NEURON:
                print(f"  Neuron has {len(embedding_files)} embeddings. Sampling {MAX_IMAGES_PER_NEURON} of them.")
                embedding_files = random.sample(embedding_files, MAX_IMAGES_PER_NEURON)

            image_paths = []
            base64_images = []

            for embed_filename in embedding_files:
                try:
                    # Map embedding filename (.pt) to cropped image filename (.jpg)
                    image_filename = embed_filename.rsplit('.', 1)[0] + '.jpg'
                    image_path = os.path.join(CROPPED_IMAGE_DIR, image_filename)

                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        base64_images.append(encode_image(image_path))
                        out_f.write(f"  - Image: {image_path}\n")
                    else:
                        print(f"    Could not find image at {image_path}. Skipping.")

                except Exception as e:
                    print(f"    Error parsing filename '{embed_filename}': {e}. Skipping.")
                    continue

            if not base64_images:
                print("  No valid cropped images found for this neuron. Skipping MLLM call.")
                out_f.write("MLLM Output: No valid data to process.\n")
                out_f.write("=" * 50 + "\n")
                continue

            # --- 3. MLLM Interaction ---
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }

            prompt_text = """
            Analyze all the cropped images provided below. 
            What is the single, unifying, detailed concept that connects all of the images? be very specific and precise in what you see is occurring.
            Describe this concept in a short, descriptive phrase (e.g., "body of a person", "rubble of a building", "fire consuming a building", etc.) which should be the only output and not more than 3-4 words.
            """

            # Construct the content payload with the main prompt and all images
            content_payload = [{"type": "text", "text": prompt_text}]
            for i in range(len(base64_images)):
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
                print(f"  Sending {len(base64_images)} images to MLLM...")
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                mllm_output = response.json()['choices'][0]['message']['content']
                print(f"  MLLM Response: {mllm_output.strip()}")

            except requests.exceptions.RequestException as e:
                print(f"    Error calling OpenAI API: {e}")
                if 'response' in locals() and response:
                    print("Status:", response.status_code)
                    print("Body:", response.text)
                mllm_output = f"Error: API request failed. {e}"
            except (KeyError, IndexError) as e:
                print(f"    Error parsing OpenAI response: {e}")
                mllm_output = f"Error: Could not parse MLLM response."

            # --- 4. Write to output file ---
            out_f.write(f"MLLM Concept Name: {mllm_output.strip()}\n")
            out_f.write("=" * 50 + "\n")

    print(f"\nProcessing complete. Output written to {OUTPUT_FILE}")


if __name__ == '__main__':
    concept_naming_by_neuron()
