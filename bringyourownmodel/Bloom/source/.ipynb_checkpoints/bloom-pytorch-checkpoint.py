import argparse
import json
import logging
import os
import sys

import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from peft import LoraConfig, get_peft_model
import transformers
from datasets import load_dataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net())
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


    
class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)


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
    

def merge_columns(example):
    example["prediction"] = example["quote"] + " ->: " + str(example["tags"])
    return example


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # Data and model checkpoints directories
    model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-7b1",
        load_in_8bit=True,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
    for param in model.parameters():
      param.requires_grad = False  # freeze the model - train adapters later
      if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    
    model.lm_head = CastOutputToFloat(model.lm_head)
    
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=16, #attention heads
        lora_alpha=32, #alpha scaling
        # target_modules=["q_proj", "v_proj"], #if you know the
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    
    data = load_dataset("Abirate/english_quotes")
    
    data['train'] = data['train'].map(merge_columns)
    data['train']["prediction"][:5]
    
    data['train'][0]
    data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)
    data
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data['train'],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            max_steps=200,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir='outputs'
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    
    save_model(model, os.environ["SM_MODEL_DIR"])
