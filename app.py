import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-6b",revision="dev")
    model = AutoModelForCausalLM.from_pretrained("PygmalionAI/pygmalion-6b",revision="dev", torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    max_new = model_inputs.get('max_new_tokens', 10)
    temperature = model_inputs.get('temperature', 1.0)
    top_k = model_inputs.get('top_k', 50)
    top_p = model_inputs.get('top_p', 1.0)
    repetition_penalty = model_inputs.get('repetition_penalty', 1.0)
    repetition_penalty_range = model_inputs.get('repetition_penalty_range', 10)
    do_sample = model_inputs.get('do_sample', True)
    num_return_sequences = model_inputs.get('num_return_sequences', 1)

    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Tokenize inputs
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Run the model, and set `pad_token_id` to `eos_token_id`:50256 for open-end generation
    output = model.generate(input_tokens, max_new_tokens=max_new, pad_token_id=50256, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, do_sample=do_sample, num_return_sequences=num_return_sequences, repetition_penalty_range=repetition_penalty_range)

    # Decode output tokens
    output_text = tokenizer.batch_decode(output, skip_special_tokens = True)[0]

    result = {"output": output_text}

    # Return the results as a dictionary
    return result
