"""
Examples of the Computer Vision module usage
"""
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from colorama import init, Fore, Style

# Import required dependencies
import pylru as lru_replacement
init()  # Initialize colorama

# Import dialect functions needed by vision module
from thainlp.dialects import get_dialect_info, get_dialect_features

# Check for required dependencies
try:
    from thainlp.vision import (
        # Import high-level functions
        classify_image,
        detect_objects,
        segment_image,
        generate_image,
        estimate_depth,
        image_to_text,
        
        # Import classes for more advanced usage
        ImageClassifier,
        ZeroShotImageClassifier,
        ObjectDetector,
        ImageSegmenter,
        Text2Image,
        DepthEstimator,
        FeatureExtractor
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nPlease make sure all dependencies are installed")
    sys.exit(1)

def show_image(image, title=None):
    """Display an image"""
    plt.figure(figsize=(10, 10))
    if title:
        plt.title(title)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def classification_example(image_path):
    """Image classification example"""
    print("\n=== Image Classification Example ===")
    
    # Basic usage with default model
    results = classify_image(image_path)
    print("Classification results:")
    for label, score in results.items():
        print(f"- {label}: {score:.4f}")
    
    # Zero-shot classification
    categories = ["animal", "landscape", "food", "vehicle", "person"]
    print("\nZero-shot classification with custom categories:")
    zero_shot_classifier = ZeroShotImageClassifier()
    results = zero_shot_classifier.classify(image_path, categories)
    for category, score in results.items():
        print(f"- {category}: {score:.4f}")
    
    # Load and display the image
    image = Image.open(image_path)
    show_image(image, "Classified Image")

def object_detection_example(image_path):
    """Object detection example"""
    print("\n=== Object Detection Example ===")
    
    # Detect objects using default model
    detections = detect_objects(image_path)
    print(f"Found {len(detections)} objects:")
    for i, detection in enumerate(detections):
        print(f"{i+1}. {detection['label']} (score: {detection['score']:.4f})")
    
    # Visualize detections
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Draw bounding boxes
    for detection in detections:
        box = detection["box"]
        label = detection["label"]
        score = detection["score"]
        
        # Draw box
        draw.rectangle(box, outline="red", width=3)
        
        # Draw label with validation status
        label_text = f"{label}: {score:.2f}"
        if 'label_valid' in detection:
            label_text += " ✓" if detection['label_valid'] else " ✗"
        if 'corrected_label' in detection:
            label_text += f"\n-> {detection['corrected_label']}"
        draw.text((box[0], box[1]), label_text, fill="red")
    
    # Display the image with detections
    show_image(image, "Object Detection Results")

def segmentation_example(image_path):
    """Image segmentation example"""
    print("\n=== Image Segmentation Example ===")
    
    # Segment image using default model
    result = segment_image(image_path)
    
    print("Segmentation classes found:")
    for class_id, info in result["classes"].items():
        print(f"- {info['name']} (ID: {class_id})")
    
    # Display segmentation map
    colored_map = result["colored_map"]
    show_image(colored_map, "Segmentation Map")

def generation_example(prompt="a beautiful landscape with mountains and a lake"):
    """Image generation example"""
    print("\n=== Image Generation Example ===")
    print(f"Generating image with prompt: '{prompt}'")
    
    try:
        # Generate image from text
        image = generate_image(prompt, num_inference_steps=30)
        
        # Display generated image
        show_image(image, f"Generated: {prompt}")
    except Exception as e:
        print(f"Error generating image: {e}")
        print("Note: This example requires the diffusers package to be installed.")

def depth_estimation_example(image_path):
    """Depth estimation example"""
    print("\n=== Depth Estimation Example ===")
    
    # Estimate depth using default model
    result = estimate_depth(image_path)
    
    # Display depth map
    print("Depth range: {:.2f} to {:.2f}".format(result["min_depth"], result["max_depth"]))
    show_image(result["colored_depth"], "Depth Map")

def image_captioning_example(image_path):
    """Image captioning example with Thai validation"""
    print("\n=== Image Captioning Example ===")
    
    # Create CLIP-based embedding extractor
    from thainlp.vision.features import EmbeddingExtractor
    extractor = EmbeddingExtractor()
        
    # Load image
    image = Image.open(image_path)
    
    # Generate Thai caption
    caption = "ภาพถ่ายที่มีคนกำลังยืนอยู่ในสวน"  # Example Thai caption
    
    # Compute similarity and validate Thai
    similarity = extractor.compute_similarity(image, caption)
    print(f"\nImage-Text Similarity: {similarity:.4f}")
    
    # Display image
    show_image(image, f"Caption: {caption}")

def feature_extraction_example(image_paths):
    """Feature extraction and similarity example"""
    print("\n=== Feature Extraction Example ===")
    
    # Extract features from multiple images
    extractor = FeatureExtractor()
    features = extractor.extract(image_paths)
    
    print(f"Extracted features from {len(features)} images")
    for i, feature in enumerate(features):
        print(f"Image {i+1}: Feature shape {feature.shape}")
    
    # Compute similarity between first two images
    if len(features) >= 2:
        similarity = np.dot(features[0], features[1]) / (
            np.linalg.norm(features[0]) * np.linalg.norm(features[1])
        )
        print(f"Similarity between first two images: {similarity:.4f}")

def main():
    """Run all vision examples"""
    print("Running computer vision examples")
    
    # Use a sample image for demonstrations
    # In a real scenario, replace with your own image path
    sample_image = r"C:\Users\Admin\OneDrive\รูปภาพ\Screenshots\Screenshot 2025-04-10 200831.png"
    
    # Check if the sample image exists
    if not os.path.exists(sample_image):
        print(f"Sample image not found at {sample_image}")
        print("Please update the sample_image path or use your own images")
        return
    
    # Run examples
    classification_example(sample_image)
    object_detection_example(sample_image)
    segmentation_example(sample_image)
    generation_example()
    depth_estimation_example(sample_image)
    image_captioning_example(sample_image)
    
    # Feature extraction example with multiple images
    feature_extraction_example([sample_image, sample_image])  # Using same image twice as example

if __name__ == "__main__":
    main()
