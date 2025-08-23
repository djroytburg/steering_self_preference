from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import torch
import torch.nn.functional as F
from math import exp
import os
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from dotenv import load_dotenv

load_dotenv()

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY")
MARTIAN_API_KEY = os.getenv("MARTIAN_API_KEY")

# Model Configuration - Only supported models
MODEL_CONFIG = {
    "deepseek-v3-0324": {
        "api": "lambda_labs",
        "model_name": "deepseek-v3"
    },
    "llama3.1-8b-instruct": {
        "api": "lambda_labs", 
        "model_name": "llama3.1-8b-instruct"
    },
    "gpt-3.5-turbo": {
        "api": "openai",
        "model_name": "gpt-3.5-turbo"
    },
    "anthropic/claude-3-5-sonnet-20241022": {
        "api": "martian",
        "model_name": "anthropic/claude-3-5-sonnet-20241022"
    },
    "microsoft/phi-4": {
        "api": "martian",
        "model_name": "microsoft/phi-4"
    },
}

# API Configuration
API_CONFIG = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": OPEN_AI_API_KEY
    },
    "lambda_labs": {
        "base_url": "https://api.lambda.ai/v1",
        "api_key": LAMBDA_API_KEY
    },
    "martian": {
        "base_url": "https://withmartian.com/api/v1",
        "api_key": MARTIAN_API_KEY
    }
}

class EndOfFunctionCriteria(StoppingCriteria):
    """
        Custom `StoppingCriteria` which checks if all generated functions in the batch are completed.
        Copied from github repo of ReAlign paper.
    """

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)


class OpenAIManager():
    def __init__(self, model_type: str) -> None:
        # Validate model is supported
        if model_type not in MODEL_CONFIG:
            supported_models = ", ".join(MODEL_CONFIG.keys())
            raise ValueError(f"Unsupported model '{model_type}'. Supported models: {supported_models}")
        
        self.original_model_type = model_type
        model_config = MODEL_CONFIG[model_type]
        self.api_type = model_config["api"]
        self.model_name = model_config["model_name"]
        self.model_type = model_config["model_name"]  # For backward compatibility
        
        # Initialize the appropriate API client
        api_config = API_CONFIG[self.api_type]
        self.openai_client = OpenAI(
            base_url=api_config["base_url"],
            api_key=api_config["api_key"]
        )
        
        print(f"Initialized {self.api_type.upper()} client for model: {model_type} -> {self.model_name}")

    @retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(6))
    def single_infer(self, msg, max_tokens=256, temperature=0.0):
        """Handles a single inference request."""
        try:
            rsp = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=msg,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = rsp.choices[0].message.content
            return content
        except Exception as e:
            print(f"API server generated an error message: {e}")
            return None
    
    def infer_generate_parallel(self, messages: list[list], max_tokens=256, temperature=0.0, stop_words=None):
        """Parallel inference generation for multiple messages."""
        num_workers = len(messages)
        outputs = [None] * len(messages)  # Preallocate list to maintain order
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(self.single_infer, msg, max_tokens, temperature): i
                for i, msg in enumerate(messages)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]  # Get the original index of the message
                try:
                    result = future.result()
                    outputs[idx] = result
                except Exception as e:
                    print(f"An error occurred: {e}")
                    outputs[idx] = None  # Handle failed tasks gracefully
        return outputs

    def infer_generate(self, messages: list[list], max_tokens=256, temperature=0.0, stop_words=None):
        """Serial inference generation for backward compatibility."""
        outputs = []
        for msg in messages:
            try:
                rsp = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=msg,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                output = rsp.choices[0].message.content
            except Exception as e:
                output = None
                print(f"API server generate an error message: {e}")
            outputs.append(output)
        return outputs

    def process_message(self, msg, max_tokens, temperature, logprobs, top_logprobs):
        """Process a single message with the API call and post-process."""
        try:
            rsp = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=msg,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs
            )
            
            # Handle logprobs if available
            if hasattr(rsp.choices[0], 'logprobs') and rsp.choices[0].logprobs:
                output = rsp.choices[0].logprobs.content[0].top_logprobs
                
                # Post process
                output_prob = [0] * top_logprobs
                for top_token in output:
                    if top_token.token == "A" or top_token.token == "1":
                        output_prob[0] = exp(top_token.logprob)
                    elif top_token.token == "B" or top_token.token == "2":
                        output_prob[1] = exp(top_token.logprob)
            else:
                # Fallback: If no logprobs, try to parse the response content
                response_text = rsp.choices[0].message.content.strip()
                
                if response_text == "1":
                    output_prob = [1.0, 0.0]
                elif response_text == "2":
                    output_prob = [0.0, 1.0]
                else:
                    # If response is not "1" or "2", assign equal probability
                    output_prob = [0.5, 0.5]
                    print(f"Warning: Unexpected response '{response_text}', assigning equal probabilities")
                    
        except Exception as e:
            output_prob = None
            print(f"API server generated an error: {e}")
        return output_prob

    def prefer_generate(self, messages: list[list], max_tokens=4, temperature=0.0, logprobs=True, top_logprobs=2):
        """Generate preference scores using parallel processing."""
        if top_logprobs != 2:
            raise NotImplementedError("Assign log probs to tokens more than 2 is not implemented.")
            
        outputs = [None] * len(messages)  # Preallocate to maintain order
        with ThreadPoolExecutor(max_workers=len(messages)) as executor:
            future_to_index = {
                executor.submit(self.process_message, msg, max_tokens, temperature, logprobs, top_logprobs): i
                for i, msg in enumerate(messages)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]  # Get the original index
                try:
                    outputs[idx] = future.result()
                except Exception as e:
                    print(f"An error occurred while processing message {idx}: {e}")
                    outputs[idx] = None  # Fallback to None for failed tasks
        return outputs

    def prefer_generate_series(self, messages: list[list], max_tokens=4, temperature=0.0, logprobs=True, top_logprobs=2):
        """Serial version of preference generation for backward compatibility."""
        if top_logprobs != 2:
            raise NotImplementedError("Assign log probs to tokens more than 2 is not implemented.")
        
        outputs = []
        for msg in messages:
            try:
                rsp = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=msg,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs
                )
                
                # Handle different API response formats
                if hasattr(rsp.choices[0], 'logprobs') and rsp.choices[0].logprobs:
                    output = rsp.choices[0].logprobs.content[0].top_logprobs
                    
                    ## post process
                    output_prob = [0] * top_logprobs
                    # If the token is not A/B or 1/2, the probability is essentially 0
                    for top_token in output:
                        if top_token.token == "A" or top_token.token == "1":
                            output_prob[0] = exp(top_token.logprob)
                        elif top_token.token == "B" or top_token.token == "2":
                            output_prob[1] = exp(top_token.logprob)
                else:
                    # Fallback: If no logprobs, try to parse the response content
                    response_text = rsp.choices[0].message.content.strip()
                    if response_text == "1":
                        output_prob = [1.0, 0.0]
                    elif response_text == "2":
                        output_prob = [0.0, 1.0]
                    else:
                        # If response is not "1" or "2", assign equal probability
                        output_prob = [0.5, 0.5]
                        print(f"Warning: Unexpected response '{response_text}', assigning equal probabilities")
                        
            except Exception as e:
                output_prob = None
                print(f"API server generate an error: {e}")

            outputs.append(output_prob)    
        return outputs


class HFManager():
    def __init__(self, model_name_or_path: str, is_instruct: bool = False, bf16: bool = True) -> None:
        print(f'Load tokenizer and model from {model_name_or_path}')
        self.model_name = model_name_or_path.split('/')[-1]

        self.is_instruct = is_instruct
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer is not None: 
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if "gemma-2" in model_name_or_path.lower():
                self.tokenizer.padding_side = "right"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()
    
    def infer_generate(self, messages, max_tokens=256, temperature=0.0, stop_words=None):
        if self.is_instruct:
            device = self.model.device
            prompts = []
            for msg in messages:
                template_msg = self.tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(template_msg)
            
            if len(prompts) > 1:
                padding = True
            else:
                padding = False
            
            inputs = self.tokenizer(prompts, return_tensors="pt",
                                    padding=padding, truncation=True, add_special_tokens=False)
                                    
            prompt_length = inputs['input_ids'].shape[1]
            gen_do_sample = True if temperature > 0 else False

            outputs = self.model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                pad_token_id=self.tokenizer.pad_token_id,
                tokenizer=self.tokenizer,
                do_sample=gen_do_sample,
                temperature=temperature if gen_do_sample else None,
                max_new_tokens=max_tokens,
                top_p=1.0,
                top_k=None,
            )
            outputs = [self.tokenizer.decode(op[prompt_length:], skip_special_tokens=True).strip() for op in outputs]
        else:
            device = self.model.device
            if len(messages) > 1:
                padding = True
            else:
                padding = False
            
            inputs = self.tokenizer(messages, return_tensors="pt",
                                    add_special_tokens=True, padding=padding)
            
            prompt_length = inputs["input_ids"].shape[1]
            stopping_criteria = StoppingCriteriaList([EndOfFunctionCriteria(start_length=prompt_length, 
            eof_strings=stop_words, 
            tokenizer=self.tokenizer)])
            
            gen_do_sample = True if temperature > 0 else False
            model_outputs = self.model.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=gen_do_sample,
                temperature=temperature if gen_do_sample else None,
                max_new_tokens=max_tokens,
                top_p=1.0,
                stopping_criteria=stopping_criteria,
            )
            model_outputs = [self.tokenizer.decode(op[prompt_length:], skip_special_tokens=True) for op in model_outputs]
            
            # post process stop_criteria
            outputs = []
            for op in model_outputs:
                # may not suitable for stop_words more than 2
                outputs.append(op.rstrip(stop_words[0]).strip())
            
        return outputs

    def prefer_generate(self, messages: list, max_tokens=4, temperature=0, logprobs=True, top_logprobs=2):
        assert len(messages) == 1
        device = self.model.device
    
        if self.is_instruct:
            prompts = []
            for msg in messages:
                template_msg = self.tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(template_msg)
            if len(prompts) > 1:
                padding = True
            else:
                padding = False
            inputs = self.tokenizer(prompts, return_tensors="pt",
                                    padding=padding, truncation=True, add_special_tokens=False)
        else:
            if len(messages) > 1:
                padding = True
            else:
                padding = False
            inputs = self.tokenizer(messages, return_tensors="pt",
                                    padding=padding, add_special_tokens=True)
            
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device)
            )
        logits = outputs.logits
        first_position_logits = logits[0, len(inputs["input_ids"][0]) - 1, :]
        probs = F.softmax(first_position_logits, dim=-1)

        output_prob = [0, 0]

        # Try both A/B and 1/2 tokens
        answer_tokens_ab = ["A", "B"]
        answer_tokens_12 = ["1", "2"]
        
        # First try A/B tokens
        for posi, tok in enumerate(answer_tokens_ab):
            token_id = self.tokenizer.convert_tokens_to_ids(tok)
            if token_id != self.tokenizer.unk_token_id:  # Valid token
                output_prob[posi] += probs[token_id].item()
        
        # Then try 1/2 tokens
        for posi, tok in enumerate(answer_tokens_12):
            token_id = self.tokenizer.convert_tokens_to_ids(tok)
            if token_id != self.tokenizer.unk_token_id:  # Valid token
                output_prob[posi] += probs[token_id].item()

        return [output_prob]
