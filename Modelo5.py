import os
import re
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import autocast, GradScaler

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
    def __init__(self, hidden_size, n_heads=4, dropout=0.3):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size) # Added normalization for stability

    def forward(self, lstm_out):
        # Key, Query, and Value are all the LSTM output
        attn_output, _ = self.mha(lstm_out, lstm_out, lstm_out)
        
        # Instead of mean(), we add the original signal to the attention output
        # This prevents the model from "forgetting" the current word
        return self.ln(lstm_out + attn_output)

class WordLSTM(nn.Module):
    def __init__(self, vocab, emb, hidden, layers, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(
            emb,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.2
        )
        self.attn = MultiheadAttentionModule(hidden, n_heads)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        ctx = self.attn(out)
        return self.fc(ctx), hidden

# ===============================
# TEXTO DE MUESTRA (With Top-P / Nucleus Sampling)
# ===============================
def generate(
        model, 
        start_words,
        idx2word, 
        word2idx, 
        length=100, 
        temp=0.8, 
        top_p=0.9, 
        device="cpu", 
        recent_window=40
    ):
    
    model.eval()

    if isinstance(start_words, str):
        words = start_words.split()
    else:
        words = start_words

    idxs = [word2idx.get(w, word2idx.get("<unk>", 0)) for w in words]
    hidden = None
    
    eos_idx = word2idx.get("<eos>", None)

    with torch.no_grad():
        for _ in range(length):
            # Take context from current idxs (up to SEQ_LEN)
            context_idxs = idxs[-SEQ_LEN:] if len(idxs) > SEQ_LEN else idxs
            x = torch.tensor([context_idxs], device=device)
            
            logits, hidden = model(x, hidden)
            # Use only the prediction for the VERY LAST word in the sequence
            logits = logits[0, -1, :] / temp

            # NEW: Penalty for <eos> so it doesn't appear too much
            if eos_idx is not None:
                # We subtract a large value so the model chooses 
                # a "real" word instead of ending the sentence.
                logits[eos_idx] -= 5.0

            # Frequency Penalty (Repetition Control)
            from collections import Counter
            recent = idxs[-recent_window:]
            counts = Counter(recent)
            for idx, count in counts.items():
                logits[idx] -= count * 2.0  # Penalty factor 

            # Nucleus (Top-P) Sampling: Filter out the "tail" of unlikely words
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens that fall outside the top 90% (top_p) of probability
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            
            idxs.append(next_idx)
            if next_idx == eos_idx and words[-1] in [".", "!", "?"]:
                break

    #return " ".join([idx2word[i] for i in idxs]).replace(" <eos> ", "\n")
    return " ".join([idx2word[i] for i in idxs]).replace("<eos>", "\n")

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
        emb=128,
        hidden=256,
        layers=2
    ).to(DEVICE)

    if start_checkpoint:
        ckpt = torch.load(start_checkpoint, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        print("Checkpoint cargado")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        steps = 0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            
            logits, _ = model(x)

            # Logits: [Batch * Seq_Len, Vocab]
            # Targets: [Batch * Seq_Len]
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            # Removed GradScaler and Autocast blocks because they don't help on CPU
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        ppl = math.exp(avg_loss)

        print(f"Epoch {epoch} | Loss {avg_loss:.3f} | PPL {ppl:.1f}")

        # checkpoint
        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"{phase_name}_epoch{epoch}.pt"
        )
        torch.save({"model": model.state_dict()}, ckpt_path)

        # texto largo
        sample = generate(
            model,
            start_words="frodo",
            idx2word=idx2word,
            word2idx=word2idx,
            length=100,
            temp=0.5,
            top_p=0.5,
            device=DEVICE
        )
        print("\n--- SAMPLE ---\n", sample, "\n")

    return ckpt_path

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    # FASE 1 – coherencia local
    ckpt1 = train_phase(
        phase_name="phase1_local",
        seq_len=50,
        epochs=60,
        batch_size=64,
        lr=1e-4,
        temperature=0.8
    )

    # FASE 2 – coherencia larga
    train_phase(
        phase_name="phase2_long",
        seq_len=170,
        epochs=25,
        batch_size=32,
        lr=3e-4,
        temperature=0.6,
        start_checkpoint=ckpt1
    )
