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

JSON_CONTENT_TYPE = "application/json"

def model_fn(model_dir):
    print("Loading model")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Model initialized")
    state_dict = torch.load(os.path.join(model_dir, "model.pth"),map_location='cpu')
    model.load_state_dict(state_dict)
    print("Model loaded from fine tuned parameters")
    #model = torch.load(os.path.join(model_dir, "model.pth"))
    model.eval()
    return model.to("cpu"), tokenizer
    #return model.to(device)

def predict_fn(input_data, model_and_tokenizer):

    print("Got input Data: {}".format(input_data))
    model, tokenizer = model_and_tokenizer

    inputs = tokenizer(text, return_tensors="pt")
    #output = model(inputs["input_ids"])
    outputs = model.generate(**inputs, max_new_tokens=20)

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return [{"generated_text": prediction}]


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    print("INPUT1")
    if content_type == JSON_CONTENT_TYPE:
        print("INPUT2")
        input_data = json.loads(serialized_input_data)
        return input_data

    else:
        raise Exception("Requested unsupported ContentType in Accept: " + content_type)
        return
    
def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    print("PREDICTION", prediction_output)

    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept

    raise Exception("Requested unsupported ContentType in Accept: " + accept)
    