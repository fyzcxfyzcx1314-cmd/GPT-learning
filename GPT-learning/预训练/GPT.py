import torch
from torch import nn
import tiktoken

GPT_CONFIG_124M = { 
 "vocab_size": 50257, # 词汇表大小
 "context_length": 1024, # 上下文长度
 "emb_dim": 768, # 嵌入维度
 "n_heads": 12, # 注意力头的数量
 "n_layers": 12, # 层数
 "drop_rate": 0.1, # dropout 率
 "qkv_bias": False # 查询-键-值偏置
}

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, 
                 num_heads, dropout, qkv_bias = False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wv = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x) # (batch_size, num_tokens, d_out)
        q = q.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        k = k.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        v = v.view(batch_size, num_tokens, self.num_heads, self.head_dim) #(batch_size, num_tokens, num_heads, head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2) #(batch_size, num_heads, num_tokens, head_dim)

        #计算每个头的注意力分数
        attn_scores = q @ k.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weigths = torch.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)
        attn_weigths = self.dropout(attn_weigths)

        context_vec = (attn_weigths @ v).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class TransformBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ffn = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
        self.LayerNorm1 = nn.LayerNorm(normalized_shape=cfg["emb_dim"], eps = 1e-5)
        self.LayerNorm2 = nn.LayerNorm(normalized_shape=cfg["emb_dim"], eps = 1e-5)
        self.dropout = nn.Dropout(cfg["drop_rate"])
    def forward(self, x):
        ResNet = x
        x = self.dropout(self.att(self.LayerNorm1(x)))
        x = x + ResNet

        ResNet = x
        x = self.dropout(self.ffn(self.LayerNorm2(x)))
        x = x + ResNet

        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embed = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_embed = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.block = nn.Sequential(
            *[TransformBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.norm = nn.LayerNorm(normalized_shape=cfg["emb_dim"], eps=1e-5)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias = False
        )
    def forward(self, idx):
        batch_size, seq_len = idx.shape
        token_embed = self.token_embed(idx)
        pos_embed = self.pos_embed(
            torch.arange(seq_len, device=idx.device)
        )
        x = token_embed + pos_embed
        x = self.dropout(x)
        x = self.block(x)
        self.norm(x)
        logits = self.out_head(x)
        return logits

tokenizer = tiktoken.get_encoding("gpt2") 
batch = [] 
txt1 = "Every effort moves you" 
txt2 = "Every day holds a" 
batch.append(torch.tensor(tokenizer.encode(txt1))) 
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0) 
print(batch)

torch.manual_seed(123) 
model = GPT(GPT_CONFIG_124M) 
out = model(batch) 
print("Input batch:\n", batch) 
print("\nOutput shape:", out.shape) 
print(out)