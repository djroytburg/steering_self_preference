import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_jsonl_data, add_jsonl_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, default="llama3.1-8b-instruct", help="Model to analyze for bias")
    parser.add_argument("--comparison_model", type=str, default="gpt-3.5-turbo", help="Model to compare against")
    parser.add_argument("--data_type", type=str, default="xsum", help="Type of evaluation task (supported: 'xsum', 'cnn')")
    parser.add_argument("--base_dir", type=str, default="model_preferences_fullset/xsum", help="Base directory for data files")
    parser.add_argument("--gold_judges", nargs="+", default=["deepseek-v3", "microsoft/phi-4", "anthropic/claude-3-5-sonnet-20241022"], help="List of gold judge model names")
    parser.add_argument("--self_aware_evaluation", action='store_true', help="Use self-aware evaluation data (looks for evaluator_name_aware directory)")
    return parser.parse_args()


def judge_normal(data, model_1, model_2):
    """Calculate win rates for normal judges using probability scores."""
    m1_w, m2_w = 0, 0
    for example in data:
        model_1_prob = example.get(f"{model_1}_prob", 0)
        model_2_prob = example.get(f"{model_2}_prob", 0)
        
        if model_1_prob > model_2_prob:
            m1_w += 1
        elif model_2_prob > model_1_prob:
            m2_w += 1
        else:
            m1_w += 0.5
            m2_w += 0.5
    
    return m1_w, m2_w


def extract_preference_examples(model_pair, target_model, comparison_model, data_type, base_dir, gold_judges=None, self_aware_evaluation=False):
    """
    Extract examples showing self-preference bias vs unbiased agreement.
    
    Args:
        model_pair: Model pair in format "model1_model2"
        target_model: The model we're analyzing for bias
        comparison_model: The model it's compared against
        data_type: Type of evaluation task
        base_dir: Base directory for data files
        gold_judges: List of gold judge model names
        self_aware_evaluation: Whether to use self-aware evaluation data
    """
    if gold_judges is None:
        gold_judges = ["deepseek-v3"]
    
    awareness_mode = "self-aware" if self_aware_evaluation else "standard"
    print(f"Extracting preference examples for {target_model} vs {comparison_model} ({awareness_mode} mode)")
    print(f"Using gold judges: {gold_judges}")
    
    model_1, model_2 = model_pair.split('_')
    
    # Handle self-aware evaluation directory naming
    evaluator_name = target_model
    if self_aware_evaluation:
        evaluator_name += "_aware"
    
    # Load target model's self-evaluations (when it judges itself vs comparison model)
    target_evaluator_filepath = os.path.join(base_dir, f'evaluator_{evaluator_name}/average_{model_1}_{model_2}.jsonl')
    target_evaluator_data = load_jsonl_data(target_evaluator_filepath)
    data_len = len(target_evaluator_data)
    print(f"Loaded {data_len} examples from target model evaluations ({evaluator_name})")
    
    # Load gold judge data
    gold_judge_data = {}
    for judge in gold_judges:
        judge_data_path = os.path.join(base_dir, f"evaluator_{judge}", f"merge_{model_pair}.jsonl")
        if os.path.exists(judge_data_path):
            gold_judge_data[judge] = load_jsonl_data(judge_data_path)
            print(f"Loaded {len(gold_judge_data[judge])} items from {judge}")
        else:
            print(f"Warning: Gold judge data not found at {judge_data_path}")
            continue
    
    if not gold_judge_data:
        print(f"Error: No gold judge data found for {model_pair}")
        return
        
    # Get data IDs and create mappings
    first_judge = list(gold_judge_data.keys())[0]
    data_ids = [item['id'] for item in gold_judge_data[first_judge]]
    assert len(data_ids) == data_len, f"Data length mismatch: {len(data_ids)} vs {data_len}"
    
    judge_id_to_data = {}
    for judge, judge_data in gold_judge_data.items():
        judge_id_to_data[judge] = {item['id']: item for item in judge_data}
    
    # Create mapping for target evaluator data
    target_id_to_data = {item['id']: item for item in target_evaluator_data}
    
    # Prepare output lists
    bias_examples = []
    agreement_examples = []
    
    # Process each example
    for sub_id in data_ids:
        if sub_id not in target_id_to_data:
            continue
            
        target_eval = target_id_to_data[sub_id]
        
        # Get target model's judgment (probability scores)
        target_model_prob = target_eval.get(f"{target_model}_prob", 0)
        comparison_model_prob = target_eval.get(f"{comparison_model}_prob", 0)
        
        # Determine target model's preference
        target_prefers_self = target_model_prob > comparison_model_prob and not target_eval[f"{target_model}_ambiguous"]
        target_prefers_other = target_model_prob < comparison_model_prob and not target_eval[f"{target_model}_ambiguous"]
        # Get gold judges' consensus
        all_preferences = []
        for judge in gold_judges:
            if judge in judge_id_to_data and sub_id in judge_id_to_data[judge]:
                judge_preferences = judge_id_to_data[judge][sub_id]["preferences"]
                all_preferences.extend(judge_preferences)
        
        # Count votes for each model
        target_votes = all_preferences.count(target_model)
        comparison_votes = all_preferences.count(comparison_model)
        
        # Determine gold judges' preference
        if target_votes > comparison_votes:
            gold_prefers_target = True
        elif comparison_votes > target_votes:
            gold_prefers_target = False
        else:
            # Tie - skip this example
            continue
            
        # Get the actual responses from gold judge data
        first_judge_data = judge_id_to_data[first_judge][sub_id]
        target_response = first_judge_data[f"{target_model}_response"]
        comparison_response = first_judge_data[f"{comparison_model}_response"]
        
        # Create example entry
        example_entry = {
            "id": sub_id,
            "target_model": target_model,
            "comparison_model": comparison_model,
            "target_model_response": target_response,
            "comparison_model_response": comparison_response,
            "target_model_judgment": {
                f"{target_model}_prob": target_model_prob,
                f"{comparison_model}_prob": comparison_model_prob,
                "prefers_self": target_prefers_self
            },
            "gold_judges_consensus": {
                "judges_used": gold_judges,
                f"{target_model}_votes": target_votes,
                f"{comparison_model}_votes": comparison_votes,
                "total_votes": len(all_preferences),
                "prefers_target": gold_prefers_target
            }
        }
        
        # Categorize the example
        if target_prefers_self and not gold_prefers_target:
            # Self-preference bias: target thinks it's better, gold judges think comparison is better
            example_entry["bias_type"] = "self_preference_bias"
            bias_examples.append(example_entry)
        elif target_prefers_other and not gold_prefers_target:
            # Agreement: target thinks comparison is better, gold judges agree
            example_entry["bias_type"] = "unbiased_agreement"
            agreement_examples.append(example_entry)
    
    # Save the results with awareness suffix
    suffix = "_aware" if self_aware_evaluation else ""
    bias_output_path = f"preference_examples/{data_type}_{target_model}{suffix}_bias_examples.jsonl"
    agreement_output_path = f"preference_examples/{data_type}_{target_model}{suffix}_agreement_examples.jsonl"
    
    # Create directory if it doesn't exist
    os.makedirs("preference_examples", exist_ok=True)
    
    # Save bias examples
    if bias_examples:
        # Remove existing file if it exists
        if os.path.exists(bias_output_path):
            os.remove(bias_output_path)
        
        for example in bias_examples:
            add_jsonl_data(save_path=bias_output_path, save_data=example)
        print(f"Saved {len(bias_examples)} self-preference bias examples to {bias_output_path}")
    
    # Save agreement examples  
    if agreement_examples:
        # Remove existing file if it exists
        if os.path.exists(agreement_output_path):
            os.remove(agreement_output_path)
            
        for example in agreement_examples:
            add_jsonl_data(save_path=agreement_output_path, save_data=example)
        print(f"Saved {len(agreement_examples)} unbiased agreement examples to {agreement_output_path}")
    
    # Print summary
    print(f"\n=== Summary for {target_model} vs {comparison_model} ===")
    print(f"Self-preference bias examples: {len(bias_examples)}")
    print(f"Unbiased agreement examples: {len(agreement_examples)}")
    print(f"Total examples processed: {len(data_ids)}")
    print("=" * 60)
    
    return bias_examples, agreement_examples


def main():
    """Main function with command line arguments."""
    args = parse_args()
    
    # Create model pair from target and comparison models
    model_pair = f"{args.target_model}_{args.comparison_model}"
    
    # Extract examples
    extract_preference_examples(
        model_pair=model_pair,
        target_model=args.target_model,
        comparison_model=args.comparison_model,
        data_type=args.data_type,
        base_dir=args.base_dir,
        gold_judges=args.gold_judges,
        self_aware_evaluation=args.self_aware_evaluation
    )


if __name__ == "__main__":
    main()
