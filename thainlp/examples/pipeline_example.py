"""
Example usage of the Standard Thai NLP Pipeline
"""
from thainlp.pipelines import StandardThaiPipeline

def run_standard_pipeline(text: str):
    """
    Demonstrates processing text using the standard pipeline.
    
    Args:
        text (str): The Thai text to process.
    """
    print(f"Processing text: '{text}'")
    
    # Initialize the standard pipeline
    pipeline = StandardThaiPipeline()
    
    # Process the text
    results = pipeline(text)
    
    print("\nPipeline Results:")
    if not results:
        print("No output from pipeline.")
        return
        
    # Print results sentence by sentence
    for i, sentence in enumerate(results):
        print(f"\n--- Sentence {i+1} ---")
        if not sentence:
            print("(Empty sentence)")
            continue
            
        # Print token details
        print(f"{'Token':<15} {'POS Tag':<10} {'NER Tag'}")
        print("-" * 40)
        for token, pos, ner in sentence:
            print(f"{token:<15} {pos:<10} {ner}")

def main():
    """Run pipeline examples."""
    print("ThaiNLP Standard Pipeline Example")
    print("================================")
    
    # Example text 1: Simple sentence
    text1 = "แมวกินปลาที่ริมตลิ่ง"
    run_standard_pipeline(text1)
    
    print("\n" + "="*50 + "\n")
    
    # Example text 2: Sentence with named entities and numbers
    text2 = "นายสมชาย จันทร์โอชา เดินทางไปกรุงเทพฯ เมื่อวันที่ 15 ม.ค. 2566"
    run_standard_pipeline(text2)
    
    print("\n" + "="*50 + "\n")
    
    # Example text 3: Multiple sentences
    text3 = "บริษัท ไทยเอ็นแอลพี จำกัด ก่อตั้งในปี พ.ศ. 2560 โดย ดร. วินัย มีชัย เป็นประธานคนแรก"
    run_standard_pipeline(text3)

if __name__ == "__main__":
    main()
