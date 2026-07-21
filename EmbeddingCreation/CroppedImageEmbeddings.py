
import json
import os
import re

import clip
import torch
from PIL import Image


# ========================================================================
# CONFIGURATION
# ========================================================================
IMG_TRAIN_FILE = 'm2e2(2)/img_train.json'
IMAGE_DIR = 'm2e2(2)/image/image/'

CROPPED_IMAGE_DIR = 'cropped_images'
CROPPED_EMBEDDING_DIR = 'cropped_embedding_artifacts'

CLIP_MODEL_NAME = 'ViT-B/32'


# ========================================================================
# HELPER FUNCTIONS
# ========================================================================
def sanitize_name(value):
    """Make a string safe for filenames by replacing punctuation with hyphens."""
    value = value.replace(':', '-')
    value = value.replace('/', '-')
    value = value.replace(' ', '_')
    value = re.sub(r'[^A-Za-z0-9_.-]+', '-', value)
    return value.strip('-_')


def resolve_image_path(image_id):
    """Resolve the original image path for an image_id."""
    image_path = os.path.join(IMAGE_DIR, image_id)
    if os.path.exists(image_path):
        return image_path

    # Fall back to common image extensions if the image_id lacks one.
    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP']:
        candidate = image_path + ext
        if os.path.exists(candidate):
            return candidate

    return None


def crop_with_box(image, box):
    """Crop a PIL image using a role bbox list of the form [flag, x1, y1, x2, y2]."""
    if len(box) < 5:
        return None

    flag = str(box[0])
    if flag != '1':
        # The first field is used as a visibility/validity flag in this dataset.
        # We skip non-visible boxes so we only save meaningful crops.
        return None

    x1, y1, x2, y2 = map(int, box[1:5])

    # Clamp to image bounds and normalize coordinates in case the box is reversed.
    left = max(0, min(x1, x2))
    top = max(0, min(y1, y2))
    right = min(image.width, max(x1, x2))
    bottom = min(image.height, max(y1, y2))

    if right <= left or bottom <= top:
        return None

    return image.crop((left, top, right, bottom))


# ========================================================================
# MAIN PIPELINE
# ========================================================================
def crop_role_boxes_and_embed():
    """
    Crop all valid role bounding boxes and save both the cropped image and its
    CLIP image embedding.
    """
    # Create output folders if they do not already exist.
    os.makedirs(CROPPED_IMAGE_DIR, exist_ok=True)
    os.makedirs(CROPPED_EMBEDDING_DIR, exist_ok=True)

    # Load annotation and image mapping data.
    with open(IMG_TRAIN_FILE, 'r') as f:
        img_train = json.load(f)

    # Load CLIP once and reuse it for all crops.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    model.eval()

    total_crops = 0
    skipped_samples = 0

    # Sort keys for deterministic output ordering.
    for image_id in sorted(img_train.keys()):
        sample = img_train[image_id]

        image_path = resolve_image_path(image_id)
        if not image_path:
            print(f"Warning: Could not resolve image for image_id {image_id}. Skipping sample.")
            skipped_samples += 1
            continue

        try:
            original_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not open image {image_path}: {e}. Skipping sample.")
            skipped_samples += 1
            continue

        roles = sample.get('role', {})
        if not roles:
            print(f"Warning: No roles found for {image_id}. Skipping sample.")
            skipped_samples += 1
            continue

        # Each role may contain multiple boxes, so we enumerate them carefully.
        for role_name, boxes in roles.items():
            role_safe = sanitize_name(role_name)
            for box_index, box in enumerate(boxes, start=1):
                cropped_image = crop_with_box(original_image, box)
                if cropped_image is None:
                    continue

                base_name = f"{image_id}_{role_safe}_{box_index}"
                image_filename = f"{base_name}.jpg"
                embedding_filename = f"{base_name}.pt"

                cropped_image_path = os.path.join(CROPPED_IMAGE_DIR, image_filename)
                embedding_path = os.path.join(CROPPED_EMBEDDING_DIR, embedding_filename)

                try:
                    # Save the crop first so the image artifact is available on disk.
                    cropped_image.save(cropped_image_path, format='JPEG', quality=95)

                    # CLIP preprocessing expects the crop to be treated like a normal image.
                    processed = preprocess(cropped_image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        image_features = model.encode_image(processed)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    # Save the image-only embedding.
                    torch.save(image_features.cpu(), embedding_path)
                    total_crops += 1

                    print(f"Saved crop: {cropped_image_path}")
                    print(f"Saved embedding: {embedding_path}")

                except Exception as e:
                    print(f"Warning: Failed processing {base_name}: {e}")
                    continue

    print("\nProcessing complete.")
    print(f"Total crops saved: {total_crops}")
    print(f"Skipped samples: {skipped_samples}")
    print(f"Cropped images directory: {CROPPED_IMAGE_DIR}")
    print(f"Cropped embeddings directory: {CROPPED_EMBEDDING_DIR}")


if __name__ == '__main__':
    crop_role_boxes_and_embed()
