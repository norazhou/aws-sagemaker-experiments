import argparse
import json
import logging
import os
import sys

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def model_fn(model_dir):
    """
    Load the model and tokenizer for inference
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=bnb_config, device_map={"":0}).to(device).eval()

    return {"model": model, "tokenizer": tokenizer}


def predict_fn(input_data, model_dict):
    """
    Make a prediction with the model
    """
    text = input_data.pop("inputs")

    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]
   
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    predictions = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return predictions


def input_fn(request_body, request_content_type):
    """
    Transform the input request to a dictionary
    """
    return json.loads(request_body)