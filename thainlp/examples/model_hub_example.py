"""
Example usage of the ThaiNLP Model Hub
"""
import os
# Ensure transformers and torch/tensorflow are installed for HF models
# pip install transformers torch tensorflow sentencepiece
# Set cache directory if needed (optional)
# os.environ["THAINLP_CACHE_HOME"] = "/path/to/your/cache"

from thainlp.model_hub import ModelManager, list_models

def demonstrate_listing_models():
    """Show how to list available models with filtering."""
    print("=== Listing Models ===")

    # List all models
    print("\n--- All Registered Models ---")
    all_m = list_models()
    if all_m:
        for m in all_m:
            print(f"- {m['name']} (Task: {m['task']}, Source: {m['source']})")
    else:
        print("No models registered.")

    # List models for a specific task
    print("\n--- NER Models ---")
    ner_models = list_models(task="ner")
    if ner_models:
        for m in ner_models:
            print(f"- {m['name']} (Source: {m['source']})")
    else:
        print("No NER models registered.")

    # List models from a specific source (e.g., Hugging Face)
    print("\n--- Hugging Face Models ---")
    hf_models = list_models(source="huggingface")
    if hf_models:
        for m in hf_models:
            print(f"- {m['name']} (HF ID: {m['hf_id']})")
    else:
        print("No Hugging Face models currently registered.")
        print("(Note: Examples in registry.py are commented out by default)")

def demonstrate_loading_models():
    """Show how to load models using the ModelManager."""
    print("\n=== Loading Models ===")
    manager = ModelManager()

    # --- Load Built-in Models ---
    print("\n--- Loading Built-in Models ---")
    # Load HMM POS Tagger
    print("Loading 'hmm_pos'...")
    hmm_tagger = manager.load_model("hmm_pos")
    if hmm_tagger:
        print(f"Successfully loaded: {type(hmm_tagger)}")
        # Example usage
        # tokens = ["ฉัน", "รัก", "ภาษา", "ไทย"]
        # tags = hmm_tagger.tag(tokens)
        # print(f"POS Tags for '{' '.join(tokens)}': {tags}")
    else:
        print("Failed to load 'hmm_pos'.")

    # Load Rule-based NER Tagger
    print("\nLoading 'rule_based_ner'...")
    rb_ner = manager.load_model("rule_based_ner")
    if rb_ner:
        print(f"Successfully loaded: {type(rb_ner)}")
        # Example usage
        # tokens = ["นาย", "สมชาย", "ไป", "กรุงเทพ"]
        # tags = rb_ner.tag(tokens)
        # print(f"NER Tags for '{' '.join(tokens)}': {tags}")
    else:
        print("Failed to load 'rule_based_ner'.")

    # --- Load Hugging Face Models (Example) ---
    # Note: This requires uncommenting/adding HF models in registry.py
    # and installing 'transformers' and 'torch'/'tensorflow'.
    print("\n--- Loading Hugging Face Models (Example) ---")
    hf_model_name = "wangchanberta_ner" # Example name, must be in registry
    print(f"Attempting to load '{hf_model_name}' (if registered)...")

    model_info = manager.get_model_info(hf_model_name)
    if model_info and model_info['source'] == 'huggingface':
        try:
            # Load model and tokenizer
            loaded_hf_model = manager.load_model(hf_model_name) # Returns (model, tokenizer) tuple
            if loaded_hf_model:
                model, tokenizer = loaded_hf_model
                print(f"Successfully loaded HF model: {type(model)}")
                print(f"Successfully loaded HF tokenizer: {type(tokenizer)}")
                # Example usage (depends on model type)
                # text = "นายสมชายเดินทางไปกรุงเทพมหานคร"
                # inputs = tokenizer(text, return_tensors="pt") # Assuming PyTorch
                # outputs = model(**inputs)
                # print("HF Model output structure:", outputs.keys())
            else:
                print(f"Failed to load '{hf_model_name}'. Check registry and dependencies.")
        except ImportError as e:
             print(f"Import error loading '{hf_model_name}'. Is '{e.name}' installed? Try 'pip install {e.name}' or 'pip install thainlp[hf]'")
        except Exception as e:
            print(f"An error occurred loading '{hf_model_name}': {e}")
    else:
        print(f"'{hf_model_name}' is not registered as a Hugging Face model.")
        print("(Uncomment or add it in thainlp/model_hub/registry.py to test loading)")


def main():
    """Run all Model Hub demonstrations."""
    print("ThaiNLP Model Hub Examples")
    print("==========================")
    demonstrate_listing_models()
    demonstrate_loading_models()

if __name__ == "__main__":
    main()
