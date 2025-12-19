import os
import re
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import autocast, GradScaler
from collections import Counter

# ===============================
# CONFIG GENERAL
# ===============================
DATA_DIR = "Content"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ===============================
# TOKENIZACIÓN WORD-LEVEL
# ===============================
def tokenize(text):
    text = text.lower()
    text = re.sub(r"\n+", " <eos> ", text)
    tokens = re.findall(r"\w+|[.,!?;:]", text)
    return tokens

files = [
    "01 - The Fellowship Of The Ring.txt",
    "02 - The Two Towers.txt",
    "03 - The Return Of The King.txt"
]

import json
import torch

DATA_DIR = "data"

encoded = torch.load(os.path.join(DATA_DIR, "tokens.pt"), map_location="cpu")
word2idx = json.load(open(os.path.join(DATA_DIR, "vocab.json")))
idx2word = {int(v): k for k, v in word2idx.items()}
vocab_size = len(word2idx)

# ===============================
# DATASET ITERABLE (RÁPIDO)
# ===============================
class WordIterableDataset(IterableDataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __iter__(self):
        max_idx = len(self.data) - self.seq_len - 1
        for i in range(0, max_idx, self.seq_len):
            x = self.data[i:i+self.seq_len]
            y = self.data[i+1:i+self.seq_len+1]
            yield x, y

# ===============================
# MODELO LSTM + MULTIHEAD ATTENTION (FIXED)
# ===============================
class MultiheadAttentionModule(nn.Module):
    def __init__(self, hidden_size, n_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size) # Added normalization for stability
        self.dropout = nn.Dropout(0.3) # Added dropout here

    def forward(self, x):
        # Using a proper residual connection and layer norm
        attn_output, _ = self.mha(x, x, x)
        return self.ln(x + self.dropout(attn_output))

class WordLSTM(nn.Module):
    def __init__(self, vocab, emb, hidden, layers, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(
            emb,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.4
        )
        self.attn = MultiheadAttentionModule(hidden, n_heads)
        self.fc = nn.Linear(hidden, vocab, bias=False)

        self.fc.weight = self.embedding.weight

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        ctx = self.attn(out)
        return self.fc(ctx), hidden

# ===============================
# TEXTO DE MUESTRA (With Top-P / Nucleus Sampling)
# ===============================
def generate(model, start_words, idx2word, word2idx, length=80, temp=0.75, top_p=0.85, device="cpu"):
    model.eval()
    words = start_words.lower().split() if isinstance(start_words, str) else start_words
    idxs = [word2idx.get(w, word2idx.get("<unk>", 0)) for w in words]
    
    hidden = None
    with torch.no_grad():
        # Step 1: Initialize hidden state with the prompt
        x = torch.tensor([idxs], device=device)
        _, hidden = model(x, None)

        for _ in range(length):
            # Step 2: Only pass the LAST token and the HIDDEN state
            x = torch.tensor([[idxs[-1]]], device=device)
            logits, hidden = model(x, hidden)
            
            logits = logits[0, -1, :] / temp

            # Moderate repetition penalty (Softer than before)
            recent = idxs[-25:]
            counts = Counter(recent)
            for idx, count in counts.items():
                logits[idx] -= (count * 0.6)

            # Nucleus Sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            logits[sorted_indices[sorted_indices_to_remove]] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            
            idxs.append(next_idx)
            if idx2word.get(next_idx) == "<eos>": break

    return " ".join([idx2word.get(i, "") for i in idxs]).replace(" <eos> ", "\n")

# ===============================
# ENTRENAMIENTO POR FASES
# ===============================
def train_phase(
    phase_name,
    seq_len,
    epochs,
    batch_size,
    lr,
    temperature,
    start_checkpoint=None
):
    global SEQ_LEN
    SEQ_LEN = seq_len

    print(f"\n===== {phase_name} =====")
    print(f"SEQ_LEN={seq_len} | LR={lr} | TEMP={temperature}")

    dataset = WordIterableDataset(encoded, seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=0
    )

    model = WordLSTM(
        vocab_size,
        emb=512,
        hidden=512,
        layers=2
    ).to(DEVICE)

    if start_checkpoint:
        model.load_state_dict(torch.load(start_checkpoint)["model"])
        print("Checkpoint cargado")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Label Smoothing prevents the model from becoming "too sure" and looping
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        steps = 0 # Track steps manually

        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        # Sample every epoch to monitor "Sense"
        print("SAMPLE:", generate(model, "Frodo", idx2word, word2idx, device=DEVICE))

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    # FASE 1 – coherencia local
    ckpt1 = train_phase(
        phase_name="phase1_local",
        seq_len=50,
        epochs=8,
        batch_size=64,
        lr=1e-4,
        temperature=0.8
    )

    # FASE 2 – coherencia larga
    train_phase(
        phase_name="phase2_long",
        seq_len=100,
        epochs=4,
        batch_size=32,
        lr=5e-5, # Lower learning rate for fine-tuning
        temperature=0.6,
        start_checkpoint=ckpt1
    )
