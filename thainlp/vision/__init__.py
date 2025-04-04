"""
Computer Vision functionality for ChujaiThaiNLP.

This module provides state-of-the-art models and tools for various
computer vision tasks in both 2D and 3D domains.
"""

from .base import VisionBase, VisionConfig
from .classification import ImageClassifier, VideoClassifier, ZeroShotImageClassifier
from .detection import ObjectDetector, ZeroShotObjectDetector, KeypointDetector
from .segmentation import ImageSegmenter, InstanceSegmenter, PanopticSegmenter, MaskGenerator
from .generation import (
    Text2Image, Image2Image, Image2Video, 
    Text2Video, UnconditionalGenerator
)
from .reconstruction import DepthEstimator, Image2Text, Text23DModel, Image23DReconstructor
from .features import FeatureExtractor, EmbeddingExtractor

# Initialize default models
_image_classifier = None
_object_detector = None
_image_segmenter = None
_depth_estimator = None
_feature_extractor = None

__all__ = [
    # Base classes
    'VisionBase', 
    'VisionConfig',
    
    # Classification
    'ImageClassifier',
    'VideoClassifier',
    'ZeroShotImageClassifier',
    'classify_image',
    'classify_video',
    'zero_shot_classify_image',
    
    # Detection
    'ObjectDetector',
    'ZeroShotObjectDetector',
    'KeypointDetector',
    'detect_objects',
    'detect_keypoints',
    
    # Segmentation
    'ImageSegmenter',
    'InstanceSegmenter',
    'PanopticSegmenter',
    'MaskGenerator',
    'segment_image',
    'generate_mask',
    
    # Generation
    'Text2Image',
    'Image2Image',
    'Image2Video',
    'Text2Video',
    'UnconditionalGenerator',
    'generate_image',
    'translate_image',
    'generate_video',
    
    # Reconstruction
    'DepthEstimator',
    'Image2Text',
    'Text23DModel',
    'Image23DReconstructor',
    'estimate_depth',
    'image_to_text',
    'reconstruct_3d',
    
    # Features
    'FeatureExtractor',
    'EmbeddingExtractor',
    'extract_features',
]

# Function definitions at module level
def classify_image(image, model=None, **kwargs):
    """Classify an image using the default or specified model"""
    global _image_classifier
    if _image_classifier is None:
        _image_classifier = ImageClassifier()
    return _image_classifier.classify(image, **kwargs)

def classify_video(video, **kwargs):
    """Classify a video using the default video classifier"""
    classifier = VideoClassifier()
    return classifier.classify(video, **kwargs)

def zero_shot_classify_image(image, categories, **kwargs):
    """Classify an image with the given categories without training"""
    classifier = ZeroShotImageClassifier()
    return classifier.classify(image, categories, **kwargs)

def detect_objects(image, **kwargs):
    """Detect objects in an image using the default detector"""
    global _object_detector
    if _object_detector is None:
        _object_detector = ObjectDetector()
    return _object_detector.detect(image, **kwargs)

def detect_keypoints(image, **kwargs):
    """Detect keypoints in an image"""
    detector = KeypointDetector()
    return detector.detect(image, **kwargs)

def segment_image(image, **kwargs):
    """Segment an image using the default segmenter"""
    global _image_segmenter
    if _image_segmenter is None:
        _image_segmenter = ImageSegmenter()
    return _image_segmenter.segment(image, **kwargs)

def generate_mask(image, **kwargs):
    """Generate a mask for an image"""
    generator = MaskGenerator()
    return generator.generate(image, **kwargs)

def generate_image(prompt, **kwargs):
    """Generate an image from text"""
    generator = Text2Image()
    return generator.generate(prompt, **kwargs)

def translate_image(image, **kwargs):
    """Translate an image to another domain"""
    translator = Image2Image()
    return translator.translate(image, **kwargs)

def generate_video(prompt=None, image=None, **kwargs):
    """Generate a video from text or an image"""
    if prompt is not None:
        generator = Text2Video()
        return generator.generate(prompt, **kwargs)
    elif image is not None:
        generator = Image2Video()
        return generator.generate(image, **kwargs)
    else:
        raise ValueError("Either prompt or image must be provided")

def estimate_depth(image, **kwargs):
    """Estimate depth from an image"""
    global _depth_estimator
    if _depth_estimator is None:
        _depth_estimator = DepthEstimator()
    return _depth_estimator.estimate(image, **kwargs)

def image_to_text(image, **kwargs):
    """Convert an image to text description"""
    converter = Image2Text()
    return converter.convert(image, **kwargs)

def reconstruct_3d(image, **kwargs):
    """Reconstruct a 3D model from an image"""
    reconstructor = Image23DReconstructor()
    return reconstructor.reconstruct(image, **kwargs)

def extract_features(image, **kwargs):
    """Extract features from an image"""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = FeatureExtractor()
    return _feature_extractor.extract(image, **kwargs)

# Add all functions to globals
globals().update({
    'classify_image': classify_image,
    'classify_video': classify_video,
    'zero_shot_classify_image': zero_shot_classify_image,
    'detect_objects': detect_objects,
    'detect_keypoints': detect_keypoints,
    'segment_image': segment_image,
    'generate_mask': generate_mask,
    'generate_image': generate_image,
    'translate_image': translate_image,
    'generate_video': generate_video,
    'estimate_depth': estimate_depth,
    'image_to_text': image_to_text,
    'reconstruct_3d': reconstruct_3d,
    'extract_features': extract_features,
})
