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
# MODELO LSTM + ATTENTION
# ===============================
class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        scores = self.v(torch.tanh(self.attn(x)))
        weights = torch.softmax(scores, dim=1)
        return (weights * x).sum(dim=1)

class WordLSTM(nn.Module):
    def __init__(self, vocab, emb, hidden, layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(
            emb, hidden, layers,
            batch_first=True,
            dropout=0.2
        )
        self.attn = Attention(hidden)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        ctx = self.attn(out)
        return self.fc(ctx), hidden

# ===============================
# TEXTO DE MUESTRA
# ===============================
import torch

def generate(model, start_words, idx2word, word2idx, length=500, temp=0.8, top_k=50, device="cuda"):
    """
    Genera texto a partir de un modelo entrenado (palabra a palabra).
    - start_words: lista de palabras iniciales o string
    - length: número máximo de palabras a generar
    - temp: temperatura para softmax
    - top_k: número de palabras candidatas a samplear
    """
    model.eval()
    if isinstance(start_words, str):
        words = start_words.split()
    else:
        words = start_words

    # Convertir a índices
    idxs = [word2idx.get(w, word2idx["<unk>"]) for w in words]
    hidden = None

    eos_idx = word2idx.get("<eos>", None)
    eos_count = 0
    recent_window = 20  # tamaño de ventana para penalizar repetición

    for _ in range(length):
        x = torch.tensor([idxs[-SEQ_LEN:]], device=device)
        logits, hidden = model(x, hidden)  # logits: [1, vocab_size]

        logits = logits.squeeze(0) / temp

        # Penalizar repetición de palabras recientes
        recent_idxs = set(idxs[-recent_window:])
        for i in recent_idxs:
            logits[i] -= 2.0  # ajustar si repite mucho

        # Control de <eos>
        if eos_idx is not None:
            last_word = idx2word[idxs[-1]]
            if last_word not in [".", "!", "?"]:
                logits[eos_idx] -= 6.0
            else:
                logits[eos_idx] -= 2.0

        # Top-k sampling
        top_vals, top_idx = torch.topk(logits, top_k)
        probs = torch.softmax(top_vals, dim=-1)
        next_idx = top_idx[torch.multinomial(probs, 1).item()].item()

        # Contar eos para parar si aparece varias veces seguidas
        if next_idx == eos_idx:
            eos_count += 1
            if eos_count >= 2:
                break
        else:
            eos_count = 0

        idxs.append(next_idx)

    # Convertir índices a palabras
    return " ".join(idx2word[i] for i in idxs)

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
        pin_memory=True,
        num_workers=0
    )

    model = WordLSTM(
        vocab_size,
        emb=256,
        hidden=512,
        layers=3
    ).to(DEVICE)

    if start_checkpoint:
        ckpt = torch.load(start_checkpoint, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        print("Checkpoint cargado")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        steps = 0

        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                logits, _ = model(x)
                loss = criterion(logits, y[:, -1])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

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
            start_words="<BOOK_1>",
            idx2word=idx2word,
            word2idx=word2idx,
            length=400,
            temp=0.8,
            top_k=50,
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
        seq_len=30,
        epochs=30,
        batch_size=64,
        lr=1e-4,
        temperature=0.8
    )

    # FASE 2 – coherencia larga
    train_phase(
        phase_name="phase2_long",
        seq_len=120,
        epochs=15,
        batch_size=32,
        lr=3e-4,
        temperature=0.6,
        start_checkpoint=ckpt1
    )
