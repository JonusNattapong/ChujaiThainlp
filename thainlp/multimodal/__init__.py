"""
Multimodal AI functionality for ChujaiThaiNLP.

This module provides advanced multimodal capabilities for seamless
cross-modal understanding and generation, including:
- Audio-Text-to-Text: Transcription, translation
- Image-Text-to-Text: OCR, captioning, analysis
- Visual Question Answering (VQA)
- Document Question Answering (DQA)
- Video-Text-to-Text: Summarization, transcription
- Visual Document Retrieval
- Any-to-Any modality conversion
"""

from .base import MultimodalBase, MultimodalConfig
from .audio_text import AudioTextProcessor, AudioTranscriber, AudioTranslator
from .image_text import ImageTextProcessor, OCRProcessor, ImageCaptioner, ImageAnalyzer
from .vqa import VisualQA
from .document_qa import DocumentQA, DocumentProcessor
from .video_text import VideoTextProcessor, VideoTranscriber, VideoSummarizer
from .document_retrieval import VisualDocumentRetriever, DocumentIndexer
from .converters import ModalityConverter, Text2TextConverter, Text2ImageConverter
from .pipeline import MultimodalPipeline

# Initialize default processors
_audio_processor = None
_image_processor = None
_vqa_processor = None
_document_processor = None
_video_processor = None
_modality_converter = None

__all__ = [
    # Base classes
    'MultimodalBase',
    'MultimodalConfig',
    
    # Audio-Text
    'AudioTextProcessor',
    'AudioTranscriber',
    'AudioTranslator',
    'transcribe_audio',
    'translate_audio',
    
    # Image-Text
    'ImageTextProcessor',
    'OCRProcessor',
    'ImageCaptioner',
    'ImageAnalyzer',
    'extract_text_from_image',
    'caption_image',
    'analyze_image',
    
    # Visual QA
    'VisualQA',
    'answer_visual_question',
    
    # Document QA
    'DocumentQA',
    'DocumentProcessor',
    'answer_document_question',
    'process_document',
    
    # Video-Text
    'VideoTextProcessor',
    'VideoTranscriber',
    'VideoSummarizer',
    'transcribe_video',
    'summarize_video',
    
    # Document Retrieval
    'VisualDocumentRetriever',
    'DocumentIndexer',
    'retrieve_similar_documents',
    'index_documents',
    
    # Modality Conversion
    'ModalityConverter',
    'Text2TextConverter',
    'Text2ImageConverter',
    'convert_modality',
    
    # Pipeline
    'MultimodalPipeline',
    'process_multimodal',
]

# Audio-Text convenience functions
def transcribe_audio(audio_path, **kwargs):
    """Transcribe audio to text"""
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioTextProcessor()
    return _audio_processor.transcribe(audio_path, **kwargs)

def translate_audio(audio_path, target_lang="en", **kwargs):
    """Transcribe and translate audio to text in target language"""
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioTextProcessor()
    return _audio_processor.translate(audio_path, target_lang=target_lang, **kwargs)

# Image-Text convenience functions
def extract_text_from_image(image_path, **kwargs):
    """Extract text from an image using OCR"""
    processor = OCRProcessor()
    return processor.extract_text(image_path, **kwargs)

def caption_image(image_path, **kwargs):
    """Generate a caption for an image"""
    captioner = ImageCaptioner()
    return captioner.caption(image_path, **kwargs)

def analyze_image(image_path, analysis_type="general", **kwargs):
    """Analyze image content based on analysis type"""
    analyzer = ImageAnalyzer()
    return analyzer.analyze(image_path, analysis_type=analysis_type, **kwargs)

# Visual QA convenience functions
def answer_visual_question(image_path, question, **kwargs):
    """Answer a question about an image"""
    global _vqa_processor
    if _vqa_processor is None:
        _vqa_processor = VisualQA()
    return _vqa_processor.answer(image_path, question, **kwargs)

# Document QA convenience functions
def answer_document_question(document_path, question, **kwargs):
    """Answer a question about a document"""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentQA()
    return _document_processor.answer(document_path, question, **kwargs)

def process_document(document_path, **kwargs):
    """Process a document for analysis and retrieval"""
    processor = DocumentProcessor()
    return processor.process(document_path, **kwargs)

# Video-Text convenience functions
def transcribe_video(video_path, **kwargs):
    """Transcribe speech from a video"""
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoTextProcessor()
    return _video_processor.transcribe(video_path, **kwargs)

def summarize_video(video_path, **kwargs):
    """Generate a summary of video content"""
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoTextProcessor()
    return _video_processor.summarize(video_path, **kwargs)

# Document Retrieval convenience functions
def retrieve_similar_documents(query, index_path=None, **kwargs):
    """Retrieve documents similar to the query"""
    retriever = VisualDocumentRetriever()
    return retriever.retrieve(query, index_path=index_path, **kwargs)

def index_documents(document_paths, index_path=None, **kwargs):
    """Index documents for future retrieval"""
    indexer = DocumentIndexer()
    return indexer.index(document_paths, index_path=index_path, **kwargs)

# Modality Conversion convenience functions
def convert_modality(source, source_type, target_type, **kwargs):
    """Convert content from one modality to another"""
    global _modality_converter
    if _modality_converter is None:
        _modality_converter = ModalityConverter()
    return _modality_converter.convert(source, source_type, target_type, **kwargs)

# Pipeline convenience functions
def process_multimodal(inputs, tasks, **kwargs):
    """Process multimodal inputs with a series of tasks"""
    pipeline = MultimodalPipeline()
    return pipeline.process(inputs, tasks, **kwargs)

# Add all functions to globals
globals().update({
    'transcribe_audio': transcribe_audio,
    'translate_audio': translate_audio,
    'extract_text_from_image': extract_text_from_image,
    'caption_image': caption_image,
    'analyze_image': analyze_image,
    'answer_visual_question': answer_visual_question,
    'answer_document_question': answer_document_question,
    'process_document': process_document,
    'transcribe_video': transcribe_video,
    'summarize_video': summarize_video,
    'retrieve_similar_documents': retrieve_similar_documents,
    'index_documents': index_documents,
    'convert_modality': convert_modality,
    'process_multimodal': process_multimodal,
})
