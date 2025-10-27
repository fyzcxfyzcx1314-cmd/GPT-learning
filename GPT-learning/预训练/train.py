import torch
from torch import nn
from torch.nn import functional as F
import tiktoken
import urllib.request
from torch.utils.data import DataLoader, Dataset
from GPT import GPT

GPT_CONFIG_124M = { 
 "vocab_size": 50257, 
 "context_length": 256, 
 "emb_dim": 768, 
 "n_heads": 12, 
 "n_layers": 12, 
 "drop_rate": 0.1, 
 "qkv_bias": False 
}

url = ("https://raw.githubusercontent.com/rasbt/" 
 "LLMs-from-scratch/main/ch02/01_main-chapter-code/" 
 "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

def try_gpu():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

tokenizer = tiktoken.get_encoding("gpt2")
total_characters = len(text_data) 
total_tokens = len(tokenizer.encode(text_data)) 
print("Characters:", total_characters) 
print("Tokens:", total_tokens)

train_ratio = 0.90 
split_idx = int(train_ratio * len(text_data)) 
train_data = text_data[:split_idx] 
val_data = text_data[split_idx:]

# Dataset
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        super().__init__()
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def GPTDataLoader(txt, batch_size, max_length, stride, shuffle = True, 
                  drop_last = True, num_workers = 0, tokenizer = tokenizer):
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return dataloader

train_loader = GPTDataLoader( 
    train_data, 
    batch_size=2, 
    max_length=GPT_CONFIG_124M["context_length"], 
    stride=GPT_CONFIG_124M["context_length"], 
    drop_last=True, 
    shuffle=True, 
    num_workers=0 
) 
val_loader = GPTDataLoader( 
    val_data, 
    batch_size=2, 
    max_length=GPT_CONFIG_124M["context_length"], 
    stride=GPT_CONFIG_124M["context_length"], 
    drop_last=False, 
    shuffle=False, 
    num_workers=0 
)

def calc_loss_batch(inputs, targets, model, device):
    inputs = inputs.to(device)
    targets = targets.to(device)
    logits = model(inputs)
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    return loss 

def clac_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (inputs, targets) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(inputs, targets, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = clac_loss_loader(train_loader, model, device, 
                                      num_batches=eval_iter)
        val_loss = clac_loss_loader(val_loader, model, device, 
                                      num_batches=eval_iter)
        model.train()
        return train_loss, val_loss

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

def text_to_id(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def id_to_text(id, tokenizer):
    flat = id.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_embed.weight.shape[0]
    encoded = text_to_id(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model = model, idx = encoded,
            max_text = 50, context_size = context_size
        )
    decoded_text = id_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " ")) 
    model.train()

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_token_seen = [], [], []
    token_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        model.to(device)
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(inputs, targets, model, device)
            loss.backward()
            optimizer.step()
            token_seen += inputs.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(token_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): " 
                      f"Train loss {train_loss:.3f}, " 
                      f"Val loss {val_loss:.3f}" 
                      )
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_token_seen

torch.manual_seed(123) 
model = GPT(GPT_CONFIG_124M) 
optimizer = torch.optim.AdamW( model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10 
device = try_gpu()
train_losses, val_losses, tokens_seen = train_model( 
 model, train_loader, val_loader, optimizer, device, 
 num_epochs=num_epochs, eval_freq=5, eval_iter=5, 
 start_context="Every effort moves you", tokenizer=tokenizer 
)