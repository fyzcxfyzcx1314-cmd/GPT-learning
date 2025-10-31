import tiktoken
import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import Finetuning as ft
from torch import nn

BASE_CONFIG = { 
 "vocab_size": 50257, 
 "context_length": 1024, 
 "drop_rate": 0.0, 
 "qkv_bias": True 
}

tokenizer = tiktoken.get_encoding("gpt2")

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length, pad_id):
        self.data = pd.read_csv(csv_file)
        self.encoded_text = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
        self.encoded_text = [
            encoded_text[:self.max_length] for encoded_text in self.encoded_text
        ]
        self.encoded_text = [
            encoded_text + [pad_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_text
        ]
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_text:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
    def __getitem__(self, index):
        encoded = self.encoded_text[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    def __len__(self):
        return len(self.data)

train_dataset = SpamDataset(
    csv_file="dataset/train.csv",
    tokenizer=tokenizer,
    max_length=None
)
val_dataset = SpamDataset(
    csv_file="dataset/test.csv",
    tokenizer=tokenizer,
    max_length=train_dataset.max_length,
)
test_dataset = SpamDataset(
    csv_file="dataset/test.csv",
    tokenizer=tokenizer
)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)
def text_to_id(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def id_to_text(id, tokenizer):
    flat = id.squeeze(0)
    return tokenizer.decode(flat.tolist())
def generate_text_simple(model, idx, max_text, context_size):
    for _ in range(max_text):
        idx = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx)
        logits = logits[:, -1, :]
        idx_text = torch.softmax(logits, dim = -1)
        idx_next = torch.argmax(idx_text, dim = -1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim = -1)
    return idx

model = ft.model
text_1 = "Every effort moves you"
token_ids = generate_text_simple(
 model=ft.model,
 idx=text_to_id(text_1, tokenizer),
 max_new_tokens=15,
 context_size=BASE_CONFIG["context_length"]
)
print(id_to_text(token_ids, tokenizer))

for param in model.parameters():
    param.requires_grad = False
model.lm_head = nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=2)
# 获取模型的所有 Transformer 块
transformer_blocks = model.transformer.h

# 解冻最后一个 Transformer 块的参数
last_block_index = len(transformer_blocks) - 1
for param in transformer_blocks[last_block_index].parameters():
    param.requires_grad = True

# 解冻最终层归一化的参数
# 最后一个 Transformer 块的归一化层
model.transformer.ln_f.requires_grad = True

# 输出确认状态
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"可训练参数: {name}")