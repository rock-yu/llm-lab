"""
STEP 3 — The model: a tiny decoder-only Transformer in the GPT architecture family.

Data flow for one sequence of T token IDs:

  ids ──► token embedding + position embedding        (T, C)   "what" + "where"
      ──► N × Block:
             x = x + Attention(LayerNorm(x))   tokens look back at earlier tokens
             x = x + MLP(LayerNorm(x))         each token "thinks" on its own
      ──► LayerNorm ──► Linear to vocab        (T, V)   a score for every next token

"Causal" means position t may only attend to positions ≤ t. This permits
autoregressive next-token training without seeing future answers. Multiple
positions can be scored in parallel; prompt and padding targets may be ignored.

Default size (4 layers, 4 heads, 128 dims) ≈ 0.8M parameters. GPT-2 small is
124M; Llama-3-8B is about 8,000M. Both use decoder-only Transformers, but details
such as positional encoding, normalization, and MLP design also differ.

Run this file directly to print the parameter count.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 72
    block_size: int = 48      # max sequence length (context window)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    causal: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # query, key, value in one matmul
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout
        self.causal = cfg.causal

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # split channels into heads: (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # softmax(q·kᵀ/√d) · v, with a causal mask so token t can't see t+1…
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connections keep gradients healthy
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight      # weight tying: input & output share embeddings
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))            # (B, T, vocab)
        loss = None
        if targets is not None:
            # Cross-entropy = -log p(correct next token), averaged over positions.
            # Positions whose target is -100 (prompt, padding) are ignored.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=None, eos_id=None):
        """Autoregressive decoding: predict one token, append it, repeat.
        temperature=0 → greedy (always the most likely token). This removes
        sampling randomness, but hardware/software differences can still matter."""
        done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :]
            if temperature == 0.0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None:
                done |= nxt.squeeze(1) == eos_id
                if done.all():
                    break
        return idx


@torch.no_grad()
def predict(model, tok, inputs, device="cpu", max_new_tokens=11, batch_size=256):
    """Answer a list of raw inputs (e.g. '3 March 2024') with greedy decoding.

    Prompts are grouped by length so each batch needs no padding — the model was
    trained with real tokens starting at position 0, so we keep it that way."""
    model.eval()
    by_len = {}
    for i, text in enumerate(inputs):
        by_len.setdefault(len(text), []).append(i)
    outputs = [None] * len(inputs)
    for idxs in by_len.values():
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            x = torch.tensor([tok.encode(inputs[i] + "=") for i in chunk], device=device)
            y = model.generate(x, max_new_tokens, temperature=0.0, eos_id=tok.eos_id)
            for row, i in enumerate(chunk):
                text = tok.decode(y[row, x.size(1):].tolist())
                outputs[i] = text.split("\n")[0]          # stop at EOS
    return outputs


if __name__ == "__main__":
    from step2_tokenizer import CharTokenizer
    cfg = GPTConfig(vocab_size=CharTokenizer().vocab_size)
    model = TinyGPT(cfg)
    print(cfg)
    print(f"parameters: {model.num_params():,}")
    for name, p in list(model.named_parameters())[:8]:
        print(f"  {name:<28} {tuple(p.shape)}")
    print("  ...")
