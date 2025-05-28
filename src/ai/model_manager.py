from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

model = None
tokenizer = None

def load_model(model_name, hf_token, device=None):
    global model, tokenizer
    if model is None or tokenizer is None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, token=hf_token).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    return model, tokenizer 
