import torch
from torch import nn

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