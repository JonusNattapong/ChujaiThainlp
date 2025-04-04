"""
Examples of the Multimodal module usage
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from thainlp.multimodal import (
    # Import high-level functions
    transcribe_audio,
    translate_audio,
    extract_text_from_image,
    caption_image,
    analyze_image,
    answer_visual_question,
    answer_document_question,
    process_document,
    transcribe_video,
    summarize_video,
    convert_modality,
    
    # Import classes for more advanced usage
    AudioTextProcessor,
    OCRProcessor,
    ImageCaptioner,
    VisualQA,
    DocumentQA,
    VideoTextProcessor,
    MultimodalPipeline
)
from thainlp.tokenization import word_tokenize # <-- Import word_tokenize

def show_image(image, title=None):
    """Display an image"""
    plt.figure(figsize=(10, 6))
    if title:
        plt.title(title)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def audio_text_example(audio_path):
    """Audio-Text processing example"""
    print("\n=== Audio-Text Processing Example ===")
    
    # Basic transcription
    transcription = transcribe_audio(audio_path)
    print(f"Transcription: {transcription}")
    
    # Transcription with timestamps
    transcription_with_timestamps = transcribe_audio(audio_path, return_timestamps=True)
    print("Transcription with timestamps:")
    for segment in transcription_with_timestamps["segments"]:
        print(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")
    
    # Translation
    translation = translate_audio(audio_path, source_lang="auto", target_lang="en")
    print(f"Translation to English: {translation}")
    
    # Advanced usage with AudioTextProcessor
    processor = AudioTextProcessor()
    transcription = processor.transcribe(audio_path, language="th")
    print(f"Thai transcription: {transcription}")

def image_text_example(image_path):
    """Image-Text processing example"""
    print("\n=== Image-Text Processing Example ===")
    
    # Load and display the image
    image = Image.open(image_path)
    show_image(image, "Input Image")
    
    # Extract text using OCR
    extracted_text = extract_text_from_image(image_path)
    print(f"Extracted text: {extracted_text}")
    
    # Generate image caption
    caption = caption_image(image_path)
    print(f"Image caption: {caption}")
    
    # Analyze image content
    analysis = analyze_image(image_path, top_k=3)
    print("Image analysis:")
    for label, score in analysis.items():
        print(f"- {label}: {score:.4f}")
    
    # Advanced usage with OCR processor and text tokenization
    print("\n--- Advanced OCR Usage & Text Integration ---")
    ocr_processor = OCRProcessor()
    # extract_text now returns a list of dictionaries
    ocr_results = ocr_processor.extract_text(image_path, return_confidence=True)
    
    # Process each result (even if only one image was input)
    for i, result in enumerate(ocr_results):
        print(f"Result for Image {i+1}:")
        extracted_text = result.get("text", "")
        confidence = result.get("confidence")
        
        print(f"  Extracted Text: '{extracted_text}'")
        if confidence is not None:
            print(f"  Confidence: {confidence:.4f}")
        else:
            print("  Confidence: Not available")
            
        # Integrate with thainlp.word_tokenize
        if extracted_text:
            tokens = word_tokenize(extracted_text)
            print(f"  Tokenized Text: {tokens}")
        else:
            print("  Tokenized Text: (No text extracted)")
        print("-" * 20)
def visual_qa_example(image_path):
    """Visual Question Answering example"""
    print("\n=== Visual Question Answering Example ===")
    
    # Load and display the image
    image = Image.open(image_path)
    show_image(image, "VQA Image")
    
    # Ask questions about the image
    questions = [
        "What can you see in this image?",
        "What colors are prominent in this image?",
        "Is there any text in this image?"
    ]
    
    for question in questions:
        answer = answer_visual_question(image_path, question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print()
    
    # Advanced usage with multiple answers
    vqa = VisualQA()
    answers = vqa.answer(
        image_path, 
        "What is in this image?", 
        num_answers=3, 
        return_scores=True
    )
    
    print("Multiple answers:")
    for answer in answers:
        print(f"- {answer['answer']} (score: {answer['score']:.4f})")

def document_qa_example(document_path):
    """Document Question Answering example"""
    print("\n=== Document Question Answering Example ===")
    
    # Process document
    doc_info = process_document(document_path)
    
    if "metadata" in doc_info:
        print(f"Document: {doc_info['metadata'].get('title', 'Untitled')}")
        print(f"Pages: {doc_info['metadata'].get('pages', 'Unknown')}")
    
    # Ask questions about the document
    questions = [
        "What is this document about?",
        "What is the main topic discussed in this document?",
        "When was this created?"
    ]
    
    for question in questions:
        answer = answer_document_question(document_path, question)
        if isinstance(answer, dict) and "answer" in answer:
            print(f"Q: {question}")
            print(f"A: {answer['answer']}")
            print(f"Confidence: {answer.get('score', 0):.4f}")
        else:
            print(f"Q: {question}")
            print(f"A: {answer}")
        print()
    
    # Advanced usage with DocumentQA
    doc_qa = DocumentQA()
    answer = doc_qa.answer(
        document_path,
        "Summarize this document briefly.",
        return_context=True
    )
    
    print("Document summary:")
    print(answer["answer"])

def video_processing_example(video_path):
    """Video processing example"""
    print("\n=== Video Processing Example ===")
    
    # Transcribe video
    transcription = transcribe_video(video_path)
    print("Video transcription:")
    if isinstance(transcription, dict) and "text" in transcription:
        print(transcription["text"])
    else:
        print(transcription)
    
    # Generate video summary
    summary = summarize_video(video_path)
    print("\nVideo summary:")
    print(summary["summary"])
    
    # Advanced usage with VideoTextProcessor
    processor = VideoTextProcessor()
    
    # Extract visual information only
    visual_only = processor.summarize(
        video_path,
        use_transcription=False,
        use_visual=True,
        num_frames=8
    )
    
    print("\nVisual-only description:")
    print(visual_only.get("visual_description", ""))

def modality_conversion_example(text, image_path, audio_path):
    """Modality conversion example"""
    print("\n=== Modality Conversion Example ===")
    
    # Text to image
    print("Converting text to image...")
    image = convert_modality(
        text,
        source_type="text",
        target_type="image",
        width=512,
        height=512,
        num_inference_steps=30
    )
    show_image(image, f"Generated from: {text}")
    
    # Text to audio
    print("Converting text to audio...")
    audio = convert_modality(
        "สวัสดีครับ นี่คือการทดสอบการแปลงข้อความเป็นเสียง",
        source_type="text",
        target_type="audio",
        voice_id=0
    )
    print(f"Generated audio shape: {audio.shape}")
    
    # Image to text
    print("Converting image to text...")
    caption = convert_modality(
        image_path,
        source_type="image",
        target_type="text",
        mode="caption"
    )
    print(f"Image caption: {caption}")
    
    # Audio to text
    print("Converting audio to text...")
    transcription = convert_modality(
        audio_path,
        source_type="audio",
        target_type="text"
    )
    print(f"Audio transcription: {transcription}")

def pipeline_example(input_path):
    """Multimodal pipeline example"""
    print("\n=== Multimodal Pipeline Example ===")
    
    # Create pipeline
    pipeline = MultimodalPipeline()
    
    # Determine input type
    input_type = pipeline.determine_input_type(input_path)
    print(f"Input type detected: {input_type}")
    
    # Define tasks based on input type
    if input_type == "image":
        # Image processing pipeline
        tasks = [
            {
                "type": "image_caption",
                "name": "caption",
                "params": {"prompt": "An image of"}
            },
            {
                "type": "image_ocr",
                "name": "ocr_text",
                "params": {"return_confidence": True}
            },
            {
                "type": "vqa",
                "name": "vqa_result",
                "params": {"question": "What can you see in this image?"}
            },
            {
                "type": "convert_modality",
                "name": "text_summary",
                "input": "caption",
                "params": {
                    "source_type": "text",
                    "target_type": "text",
                    "conversion_type": "summarize"
                }
            }
        ]
    elif input_type == "audio":
        # Audio processing pipeline
        tasks = [
            {
                "type": "audio_transcribe",
                "name": "transcription",
                "params": {"return_timestamps": True}
            },
            {
                "type": "audio_translate",
                "name": "translation",
                "params": {"target_lang": "en"}
            }
        ]
    elif input_type == "video":
        # Video processing pipeline
        tasks = [
            {
                "type": "video_transcribe",
                "name": "transcription",
                "params": {"return_timestamps": True}
            },
            {
                "type": "video_summarize",
                "name": "summary",
                "params": {"max_length": 200}
            }
        ]
    elif input_type == "document":
        # Document processing pipeline
        tasks = [
            {
                "type": "document_process",
                "name": "processed_doc",
                "params": {"extract_text": True, "extract_tables": True}
            },
            {
                "type": "document_qa",
                "name": "doc_summary",
                "params": {"question": "Summarize this document briefly."}
            }
        ]
    else:
        # Default text processing
        tasks = [
            {
                "type": "convert_modality",
                "name": "translation",
                "params": {
                    "source_type": "text",
                    "target_type": "text",
                    "conversion_type": "translate",
                    "source_lang": "th",
                    "target_lang": "en"
                }
            }
        ]
    
    # Execute pipeline
    results = pipeline.process(input_path, tasks, return_intermediate=True)
    
    # Display results
    print("\nPipeline results:")
    for task_name, result in results.items():
        if task_name == "input":
            continue
            
        print(f"\n--- {task_name} ---")
        if isinstance(result, dict):
            if "text" in result:
                print(result["text"])
            elif "answer" in result:
                print(result["answer"])
            elif "summary" in result:
                print(result["summary"])
            else:
                print(f"Result type: {type(result)}")
        else:
            print(result)

def main():
    """Run all multimodal examples"""
    print("Running multimodal examples")
    
    # Use sample files for demonstrations
    # In a real scenario, replace with your own file paths
    sample_audio = "path/to/sample_audio.wav"
    sample_image = "path/to/sample_image.jpg"
    sample_document = "path/to/sample_document.pdf"
    sample_video = "path/to/sample_video.mp4"
    
    # Check if the sample files exist
    if os.path.exists(sample_audio):
        audio_text_example(sample_audio)
    else:
        print(f"Sample audio not found at {sample_audio}")
    
    if os.path.exists(sample_image):
        image_text_example(sample_image)
        visual_qa_example(sample_image)
    else:
        print(f"Sample image not found at {sample_image}")
    
    if os.path.exists(sample_document):
        document_qa_example(sample_document)
    else:
        print(f"Sample document not found at {sample_document}")
    
    if os.path.exists(sample_video):
        video_processing_example(sample_video)
    else:
        print(f"Sample video not found at {sample_video}")
    
    # Modality conversion examples (using existing samples)
    if os.path.exists(sample_image) and os.path.exists(sample_audio):
        modality_conversion_example(
            "A beautiful sunset over mountains",
            sample_image,
            sample_audio
        )
    
    # Pipeline example (using any available sample)
    for sample_path in [sample_image, sample_audio, sample_document, sample_video]:
        if os.path.exists(sample_path):
            pipeline_example(sample_path)
            break
    else:
        print("No sample files found for pipeline example")

if __name__ == "__main__":
    main()
