# Quick script to apply mask to all images in a folder
import cv2
import numpy as np
import yaml
import os

# Apply mask to image
def apply_mask(mask_polygon: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Apply polygon mask to image (keeps only beach area).
    
    Args:
        image: BGR image
        
    Returns:
        Masked image
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [mask_polygon], 255)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def mask_directory(input_dir: str, output_dir: str, config_path: str = "config.yaml"):
    """
    Apply mask to all images in a directory.
    
    Args:
        input_dir: Directory containing source images
        output_dir: Directory to save masked images
        config_path: Path to config.yaml containing mask_polygon
    """
    # Load config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    mask_polygon = np.array(config['segmentation']['mask_polygon'])

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Apply mask to all images in folder
    valid_extensions = ('.jpg')
    count = 0
    
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    for image_name in os.listdir(input_dir):
        if not image_name.lower().endswith(valid_extensions):
            continue
            
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not read image {image_name}")
            continue
            
        masked_image = apply_mask(mask_polygon, image)
        
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, masked_image)
        count += 1
        
    print(f"Processed {count} images from {input_dir} to {output_dir}")

if __name__ == "__main__":
    # Default paths
    INPUT_DIR = "training_data/extra"
    OUTPUT_DIR = "training_data/extra/masked"
    
    mask_directory(INPUT_DIR, OUTPUT_DIR)
