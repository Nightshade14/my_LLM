import torch
from torch import nn
import math


class SelfAttention(nn.Module):
    def __init__(self, num_heads=4, embedding_dim=64, block_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // self.num_heads
        self.num_tensor_groups = 3
        self.block_size = block_size

        self.qkv_proj = nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.embedding_dim * self.num_tensor_groups,
        )
        self.o_proj = nn.Linear(
            in_features=self.embedding_dim, out_features=self.embedding_dim
        )

        self.register_buffer(
            name="mask",
            tensor=torch.tril(
                input=torch.ones(self.block_size, self.block_size, dtype=torch.bool)
            )
        )
        self.mask: torch.Tensor

    def forward(self, x):
        B, T, C = x.size()
        mask= self.mask[:T, :T].unsqueeze(0).unsqueeze(0)
        x_1 = self.qkv_proj(x)
        Q, K, V = x_1.chunk(self.num_tensor_groups, dim=-1)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = Q @ K.transpose(-1, -2) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        soft_scores = torch.softmax(attn_scores, dim=-1)
        qkv_scores = soft_scores @ V

        qkv_scores_t = qkv_scores.transpose(1, 2)
        out = qkv_scores_t.contiguous().view(B, T, C)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim=64, num_heads=4, mlp_scale_factor=4, block_size=1024):
        super().__init__()

        self.block_size = block_size

        self.attention = SelfAttention(
            num_heads=num_heads, embedding_dim=emb_dim, block_size=self.block_size
        )

        self.ln1 = nn.RMSNorm(normalized_shape=emb_dim)
        self.ln2 = nn.RMSNorm(normalized_shape=emb_dim)

        self.mlp_scale_factor = 4

        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=emb_dim, out_features=emb_dim * self.mlp_scale_factor
            ),
            nn.GELU(),
            nn.Linear(in_features=emb_dim * mlp_scale_factor, out_features=emb_dim),
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DummyLLM(nn.Module):
    def __init__(
        self,
        block_size=1024,
        embedding_dim=64,
        num_heads=4,
        num_T_blocks=3,
        mlp_scale_factor=4,
        vocab_size=1024,
    ):
        super().__init__()
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_T_blocks = num_T_blocks
        self.mlp_scale_factor = mlp_scale_factor
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=self.block_size, embedding_dim=self.embedding_dim
        )
        self.network = nn.Sequential(
            *[
                TransformerBlock(
                    emb_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    mlp_scale_factor=self.mlp_scale_factor,
                    block_size=self.block_size,
                )
                for _ in range(self.num_T_blocks)
            ]
        )
        self.final_layer = nn.Linear(self.embedding_dim, self.block_size)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(end=T, device=x.device)
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(pos)
        x_in = token_emb + pos_emb
        x_temp = self.network(x_in)
        x_out = self.final_layer(x_temp)
        return x_out

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=10):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1024:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(input=logits, dim=1)
            new_token = torch.argmax(probs, keepdim=True, dim=-1)
            idx = torch.cat(tensors=(idx, new_token), dim=-1)
        return idx
