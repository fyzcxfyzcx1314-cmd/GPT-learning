import os
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

def main():
    #基本参数
    model_name = "gpt2-124M"
    train_text_path = "./dataset/train.txt"
    oupput_dir = "./gpt2_finetuned"
    per_device_batch_size = 2
    gradient_accumulation_steps = 4
    num_train_epochs = 3
    learning_rate = 5e-5
    block_size = 1024

    #加载模型与分词器
    tokenizer = GPT2Tokenizer(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    #加载本地数据