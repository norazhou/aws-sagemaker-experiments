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
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
    state_dict = torch.load(os.path.join(model_dir, "model.pth"),map_location='cpu')
    model.load_state_dict(state_dict)
    #model = torch.load(os.path.join(model_dir, "model.pth"))
    model.eval()
    return model.to("cpu")
    #return model.to(device)

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
    #torch.save(model.cpu(), path)
    
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


if __name__ == "__main__":
    #parser = argparse.ArgumentParser() 
    
    print(os.environ["SM_CHANNEL_TRAINING"])
    print(os.listdir(os.environ["SM_CHANNEL_TRAINING"]))
    
    model_id = "EleutherAI/gpt-neox-20b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
    #model = model_fn(model, os.environ["SM_CHANNEL_TRAINING"]+"/pretrained_model")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    
    
    #load data from local disk
    #data = load_dataset("Abirate/english_quotes")
    #data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    
    data = load_from_disk(os.environ["SM_CHANNEL_TRAINING"])
    print(data)
    
    # needed for gpt-neo-x tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    
    save_model(model, os.environ["SM_MODEL_DIR"])
