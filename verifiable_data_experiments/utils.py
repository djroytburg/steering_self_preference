def get_answer_qwen(sample_text, include_thinking = False):
    if not include_thinking:
        if "</think>" in sample_text:
            answer_start = sample_text.index("</think>") + len("</think>\n")
        elif "<think>" not in sample_text:
            answer_start = sample_text.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
        else:
            return None
    else:
        answer_start = sample_text.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    if "<|im_end|>" not in sample_text[answer_start:]:
        return sample_text[answer_start:]
    else:
        answer_end = sample_text.index("<|im_end|>", answer_start)
        return sample_text[answer_start:answer_end]

def get_answer_deepseek(sample_text, include_thinking = False, qwen = False):
    if qwen:
        assistant_start_tok = "<｜Assistant｜>"
    else:
        assistant_start_tok = "<|im_start|>assistant\n"
    if not include_thinking:
        answer_start = 0
        if "<think>" not in sample_text and qwen:
            answer_start = sample_text.index(assistant_start_tok) + len(assistant_start_tok)
        else:
            if "</think>" not in sample_text:
                return None
            while "</think>" in sample_text[answer_start:]:
                answer_start = sample_text.index("</think>", answer_start) + len("</think>\n")
    else:
        if qwen:
            answer_start = sample_text.index(assistant_start_tok) + len(assistant_start_tok)
        else:
            answer_start = 0

    if "<｜end▁of▁sentence｜>" not in sample_text[answer_start:]:
        return sample_text[answer_start:]
    else:
        answer_end = sample_text.index("<｜end▁of▁sentence｜>", answer_start)
        return sample_text[answer_start:answer_end].replace("<think>","").replace("</think>","\n")
    
def get_answer_llama(sample_text):
    ass_start_tok = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    answer_start = sample_text.index(ass_start_tok) + len(ass_start_tok)
    if "<|eot_id|>" not in sample_text[answer_start:]:
        return sample_text[answer_start:]
    else:
        answer_end = sample_text.index("<|eot_id|>", answer_start)
        return sample_text[answer_start:answer_end]

def get_answer_gpt_oss(sample_text):
    ass_start_tok = "<|start|>assistant<|channel|>analysis<|message|>"
    answer_start = sample_text.index(ass_start_tok) + len(ass_start_tok)
    if "<|return|>" not in sample_text[answer_start:]:
        full_answer = sample_text[answer_start:]
    else:
        answer_end = sample_text.index("<|return|>", answer_start)
        full_answer = sample_text[answer_start:answer_end]
    return full_answer.replace("<|end|><|start|>assistant<|channel|>final<|message|>","\n")
    
def get_answer_magistral(sample_text):
    ass_start_tok = "[/INST]"
    answer_start = sample_text.index(ass_start_tok) + len(ass_start_tok)
    if "</s>" not in sample_text[answer_start:]:
        full_answer = sample_text[answer_start:]
    else:
        answer_end = sample_text.index("</s>", answer_start)
        full_answer = sample_text[answer_start:answer_end]
    return full_answer
    
def get_answer_gemma(sample_text):
    ass_start_tok = "<start_of_turn>model"
    answer_start = sample_text.index(ass_start_tok) + len(ass_start_tok)
    if "<end_of_turn>" not in sample_text[answer_start:]:
        full_answer = sample_text[answer_start:]
    else:
        answer_end = sample_text.index("<end_of_turn>", answer_start)
        full_answer = sample_text[answer_start:answer_end]
    return full_answer
    
def answer_function(model_name):
    model_name = model_name.lower()
    if "deepseek" in model_name:
        return lambda x: get_answer_deepseek(x, include_thinking=True, qwen='qwen' in model_name)
    elif "qwen" in model_name:
        return get_answer_qwen
    elif "llama" in model_name:
        return get_answer_llama
    elif "gpt-5" in model_name:
        return lambda x: x
    elif "gpt-oss" in model_name:
        return get_answer_gpt_oss
    elif "magistral" in model_name:
        return get_answer_magistral
    elif "gemma" in model_name:
        return get_answer_gemma
    else:
        print("Warning: Unknown model name. Defaulting to identity function.")
        return lambda x, _: x
