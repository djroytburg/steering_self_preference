import os
import argparse
from tqdm import tqdm

from prompts.llm_evaluators import (
    LLM_EVALUATORS_PREFERENCE_SYSTEM_PROMPT,
    LLM_EVALUATORS_PREFERENCE_USER_PROMPT,
    LLM_EVALUATORS_PREFERENCE_SYSTEM_PROMPT_AWARE,
    LLM_EVALUATORS_PREFERENCE_USER_PROMPT_AWARE_SELF_FIRST,
    LLM_EVALUATORS_PREFERENCE_USER_PROMPT_AWARE_SELF_SECOND
)
from model_manager import (OpenAIManager, HFManager)
from utils import (load_jsonl_data, add_jsonl_data)


def parse_args():
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--data_dir", type=str, default=None, help="Base directory of test data.")
    parser.add_argument("--data_type", type=str, default="xsum", help="The type of dataset (supported: 'xsum', 'cnn')")
    parser.add_argument("--model_name_1", type=str, default="gpt-4o-mini")
    parser.add_argument("--model_name_2", type=str, default="gpt-4")
    
    # evaluator args
    parser.add_argument("--model_type", type=str, default="gpt-4o-mini", help="Evaluator model type")
    parser.add_argument("--is_instruct", action='store_true', help="You should set to True if use GPT models.")
    parser.add_argument("--few_shot_instruct", action='store_true', help="Use few shot examples in `Instruct` model. Only useful when `is_instruct = True`")
    parser.add_argument("--use_infer_generate", action='store_true', help="Use `infer_generate()` instead of `prefer_generate()`.")
    parser.add_argument("--self_aware_evaluation", action='store_true', help="Use self-aware prompts that tell the model which response is its own.")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def format_dataset_summarization_instruct_model(data, system_prompt, meta_user_prompt):
    messages, ids, responses_1, responses_2 = [], [], [], []
    for data_item in data:
        user_prompt = meta_user_prompt.format(
            article=data_item["query"],
            summary1=data_item["response1"],
            summary2=data_item["response2"]
        )
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        messages.append(msg)
        ids.append(data_item["id"])
        responses_1.append(data_item["response1"])
        responses_2.append(data_item["response2"])
    return messages, ids, responses_1, responses_2

def format_dataset_summarization_instruct_model_aware(data, system_prompt, evaluator_model_name, model_name_1, model_name_2):
    """
    Format dataset for self-aware evaluation where the model knows which response is its own.
    
    Args:
        data: List of data items
        system_prompt: System prompt to use
        evaluator_model_name: Name of the model doing the evaluation
        model_name_1: Name of model that generated response1
        model_name_2: Name of model that generated response2
    """
    messages, ids, responses_1, responses_2 = [], [], [], []
    
    for data_item in data:
        # Determine if the evaluator's response is in position 1 or 2
        if evaluator_model_name == model_name_1:
            # Evaluator's response is in position 1 (summary1)
            meta_user_prompt = LLM_EVALUATORS_PREFERENCE_USER_PROMPT_AWARE_SELF_FIRST
        elif evaluator_model_name == model_name_2:
            # Evaluator's response is in position 2 (summary2)
            meta_user_prompt = LLM_EVALUATORS_PREFERENCE_USER_PROMPT_AWARE_SELF_SECOND
        else:
            # Neither response belongs to the evaluator - fallback to normal prompt
            meta_user_prompt = LLM_EVALUATORS_PREFERENCE_USER_PROMPT
        
        user_prompt = meta_user_prompt.format(
            article=data_item["query"],
            summary1=data_item["response1"],
            summary2=data_item["response2"]
        )
        
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        messages.append(msg)
        ids.append(data_item["id"])
        responses_1.append(data_item["response1"])
        responses_2.append(data_item["response2"])
    
    return messages, ids, responses_1, responses_2


def gen_preference(compare_data: list[dict], model_type: str, is_instruct, 
                   few_shot_instruct, use_infer_generate, self_aware_evaluation, batch_size, data_type, 
                   model_name_1, model_name_2, save_path):
    if data_type == "summarization" or data_type == "xsum" or data_type == "cnn":
        print(f"Data type: {data_type}")
        if is_instruct:
            print("Instruct Model")
            if self_aware_evaluation:
                print("Using self-aware evaluation prompts")
                system_prompt = LLM_EVALUATORS_PREFERENCE_SYSTEM_PROMPT_AWARE
                messages, ids, responses_1, responses_2 = format_dataset_summarization_instruct_model_aware(
                    data=compare_data,
                    system_prompt=system_prompt,
                    evaluator_model_name=model_type,
                    model_name_1=model_name_1,
                    model_name_2=model_name_2,
                )
            else:
                print("Using standard evaluation prompts")
                system_prompt = LLM_EVALUATORS_PREFERENCE_SYSTEM_PROMPT
                meta_user_prompt = LLM_EVALUATORS_PREFERENCE_USER_PROMPT
                messages, ids, responses_1, responses_2 = format_dataset_summarization_instruct_model(
                    data=compare_data,
                    system_prompt=system_prompt,
                    meta_user_prompt=meta_user_prompt,
                )
        else:
            print("Base Model - Not implemented for summarization")
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    max_tokens = 4
    temperature = 0
    logprobs = True
    top_logprobs = 2
    assert len(messages) == len(ids)
    
    # Always use OpenAIManager (API models only)
    print(f"API model: {model_type}")
    assert is_instruct == True
    model_cls = OpenAIManager(model_type=model_type)

    num_batches = (len(messages) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Generating preferences:"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(messages))
        batch_messages = messages[start_idx:end_idx]

        # Special case: Claude/Martian API doesn't work with infer_generate_parallel
        # Always use prefer_generate for Claude, even when --use_infer_generate is specified
        if use_infer_generate and "anthropic" not in model_type.lower():
            batch_responses = model_cls.infer_generate_parallel(
                messages=batch_messages,
                max_tokens=1,
                temperature=temperature,
                stop_words=None
            )

            for inner_idx in range(len(batch_responses)):
                outer_idx = start_idx + inner_idx
                inner_response = batch_responses[inner_idx]
                
                # Check for both None and empty string
                if inner_response is not None and inner_response.strip():
                    save_item = {
                        "response1": responses_1[outer_idx],
                        "response2": responses_2[outer_idx],
                        "model_preference": inner_response.strip(),
                        "id": ids[outer_idx]
                    }
                    add_jsonl_data(save_path=save_path, save_data=save_item)
        else:
            batch_responses = model_cls.prefer_generate(
                messages=batch_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
            )

            # save_data
            for inner_idx in range(len(batch_responses)):
                outer_idx = start_idx + inner_idx
                inner_response: list = batch_responses[inner_idx]
                if inner_response is not None:
                    save_item = {
                        "response1": responses_1[outer_idx],
                        "prob1": inner_response[0],
                        "response2": responses_2[outer_idx],
                        "prob2": inner_response[1],
                        "id": ids[outer_idx]
                    }
                    add_jsonl_data(save_path=save_path, save_data=save_item)



def merge_data_summarization(data1, data2):
    list2_dict = {item["id"]: item["model_response"] for item in data2}
    merged_list = []
    for item in data1:
        item_id = item["id"]
        if item_id in list2_dict:
            merged_item = {
                "response1": item["model_response"],
                "response2": list2_dict[item_id],
                "golden_response": item["golden_response"],
                "query": item["query"],
                "id": item_id
            }
            merged_list.append(merged_item)
            
    return merged_list


if __name__ == "__main__":
    args = parse_args()
    data_path_1 = os.path.join(args.data_dir, args.model_name_1 + ".jsonl")
    data_path_2 = os.path.join(args.data_dir, args.model_name_2 + ".jsonl")
    
    # load data
    data1 = load_jsonl_data(data_path_1)
    data2 = load_jsonl_data(data_path_2)
    if args.data_type == "summarization" or args.data_type == "xsum" or args.data_type == "cnn":
        merge_data_func = merge_data_summarization
    else:
        raise NotImplementedError(f"Data type {args.data_type} not supported. Only 'xsum' and 'cnn' are supported.")

    data_merge = merge_data_func(data1, data2)
    print(f"Total {len(data_merge)} items to be compared.")
    
    evaluator_name = args.model_type.split('/')[-1]
    if evaluator_name == "gpt-4o-mini":
        evaluator_name = "gpt-4o-mini_choose"

    # Map data types to directory names
    data_type_to_dir = {
        "summarization": "xsum",
        "xsum": "xsum",
        "cnn": "cnn"
    }
    dir_name = data_type_to_dir.get(args.data_type, args.data_type)

    # Create different save paths for self-aware evaluation
    base_evaluator_name = evaluator_name
    if args.self_aware_evaluation:
        base_evaluator_name += "_aware"
    
    if args.few_shot_instruct:
        save_path = f"model_preferences_fullset/{dir_name}/evaluator_{base_evaluator_name}_fewshot/{args.model_name_1}_{args.model_name_2}.jsonl"
    else:
        save_path = f"model_preferences_fullset/{dir_name}/evaluator_{base_evaluator_name}/{args.model_name_1}_{args.model_name_2}.jsonl"
    
    if os.path.exists(save_path):
        raise ValueError("Path exists!")
    
    gen_preference(
        compare_data=data_merge,
        model_type=args.model_type,
        is_instruct=args.is_instruct,
        few_shot_instruct=args.few_shot_instruct,
        use_infer_generate=args.use_infer_generate,
        self_aware_evaluation=args.self_aware_evaluation,
        batch_size=args.batch_size,
        data_type=args.data_type,
        model_name_1=args.model_name_1,
        model_name_2=args.model_name_2,
        save_path=save_path
    )
