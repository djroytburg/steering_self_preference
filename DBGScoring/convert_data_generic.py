import json
import os
import argparse
from utils import add_jsonl_data

def convert_dataset_to_jsonl(dataset_name, model1_name="gpt-3.5-turbo", model2_name="llama3.1-8b-instruct"):
    """
    Generic converter for JSON dataset files to JSONL format expected by the DBG pipeline
    
    Args:
        dataset_name: Name of dataset (e.g., 'xsum', 'cnn')
        model1_name: Name for first model output file
        model2_name: Name for second model output file
    """
    
    print(f"Converting {dataset_name.upper()} dataset to JSONL format...")
    
    # Define file paths based on dataset name
    if dataset_name == "xsum":
        sources_file = "xsum_train_sources.json"
        model1_file = "xsum_train_gpt35_responses_merged.json"
        model2_file = "xsum_train_llama3.1-8b-instruct_responses_merged.json"
    elif dataset_name == "cnn":
        sources_file = "cnn_train_sources.json"
        model1_file = "cnn_train_gpt35_responses_merged.json"  
        model2_file = "cnn_train_llama3.1-8b-instruct_responses_merged.json"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    print("Loading JSON files...")
    
    # Load all 3 JSON files
    with open(model1_file, 'r') as f:
        model1_responses = json.load(f)
    print(f"Loaded {len(model1_responses)} {model1_name} responses")
    
    with open(model2_file, 'r') as f:
        model2_responses = json.load(f)
    print(f"Loaded {len(model2_responses)} {model2_name} responses")
    
    with open(sources_file, 'r') as f:
        articles = json.load(f)
    print(f"Loaded {len(articles)} articles")
    
    # Find common IDs across all 3 files
    common_ids = set(model1_responses.keys()) & set(model2_responses.keys()) & set(articles.keys())
    print(f"Found {len(common_ids)} common IDs across all files")
    
    # Create output directory
    output_dir = f"model_responses_fullset/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert Model 1 data
    print(f"Converting {model1_name} data...")
    model1_path = f"{output_dir}/{model1_name}.jsonl"
    if os.path.exists(model1_path):
        os.remove(model1_path)
    
    for doc_id in sorted(common_ids):
        save_item = {
            "id": doc_id,
            "model_response": model1_responses[doc_id],
            "query": articles[doc_id],
            "golden_response": ""  # No reference summary available
        }
        add_jsonl_data(model1_path, save_item)
    
    # Convert Model 2 data
    print(f"Converting {model2_name} data...")
    model2_path = f"{output_dir}/{model2_name}.jsonl"
    if os.path.exists(model2_path):
        os.remove(model2_path)
        
    for doc_id in sorted(common_ids):
        save_item = {
            "id": doc_id,
            "model_response": model2_responses[doc_id],
            "query": articles[doc_id], 
            "golden_response": ""
        }
        add_jsonl_data(model2_path, save_item)
    
    print(f"Conversion complete! Created JSONL files with {len(common_ids)} samples each.")
    print(f"{model1_name} file: {model1_path}")
    print(f"{model2_name} file: {model2_path}")
    
    return len(common_ids)

def main():
    parser = argparse.ArgumentParser(description="Convert JSON dataset to JSONL format for summary evaluation")
    parser.add_argument("--dataset", type=str, required=True, choices=["xsum", "cnn"], 
                       help="Dataset to convert (supported: xsum, cnn)")
    parser.add_argument("--model1", type=str, default="gpt-3.5-turbo",
                       help="Name for first model (default: gpt-3.5-turbo)")
    parser.add_argument("--model2", type=str, default="llama3.1-8b-instruct", 
                       help="Name for second model (default: llama3.1-8b-instruct)")
    
    args = parser.parse_args()
    
    convert_dataset_to_jsonl(args.dataset, args.model1, args.model2)

if __name__ == "__main__":
    main()
