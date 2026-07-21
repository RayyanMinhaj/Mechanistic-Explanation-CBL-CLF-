
import torch
import clip
from PIL import Image
import json
import os


CORREF_FILE = 'm2e2(2)/corref.txt'
TEXT_DATA_FILE = 'm2e2(2)/text_multi.json'
IMAGE_DIR = 'm2e2(2)/image/image/'
OUTPUT_DIR = 'embedding_artifacts'





# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



# Helper function to get sentence
def get_sentence_by_id(sentence_id, data):
    for item in data:
        if item['sentence_id'] == sentence_id:
            return item['sentence']
    return None



# Concatenation logic (we can try with diff methods as well)
def concatenate_embeddings(text_embedding, image_embedding):
    return torch.cat((text_embedding, image_embedding), dim=-1)




# Main processing loop
def create_embeddings():

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load text data
    with open(TEXT_DATA_FILE, 'r') as f:
        text_data = json.load(f)

    # Read corref file
    with open(CORREF_FILE, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t') # Using tab as a separator
            if len(parts) < 3:
                continue
            
            sentence_id, image_id, event_type = parts[0], parts[1], parts[2]

            # Get sentence
            sentence = get_sentence_by_id(sentence_id, text_data)
            if not sentence:
                print(f"Warning: Sentence for ID {sentence_id} not found. Skipping.")
                continue

            
            
            
            # Get image path
            image_path = os.path.join(IMAGE_DIR, image_id)
            if not os.path.exists(image_path):
                # Attempt to fix common extensions if just the ID is given
                found_image = False
                for ext in ['.jpg', '.jpeg', '.png']:
                    if os.path.exists(image_path + ext):
                        image_path = image_path + ext
                        found_image = True
                        break
                if not found_image:
                    print(f"Warning: Image file {image_id} not found at {image_path}. Skipping.")
                    continue
            
            
            
            
            
            try:
                # Preprocess image and text
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                text = clip.tokenize([sentence]).to(device)

                with torch.no_grad():
                    # Create embeddings
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(text)

                    # Normalize features (good practice for CLIP)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                # Concatenate embeddings
                combined_embedding = concatenate_embeddings(text_features, image_features)

                # Save the embedding
                output_filename = os.path.join(OUTPUT_DIR, f'embedding_{i}_{event_type.replace(":", "-")}.pt')
                torch.save(combined_embedding, output_filename)
                print(f"Saved embedding to {output_filename}")

            except Exception as e:
                print(f"Error processing line {i} ({sentence_id}, {image_id}): {e}")


if __name__ == "__main__":
    create_embeddings()
