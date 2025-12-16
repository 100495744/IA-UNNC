import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import autocast, GradScaler
from multiprocessing import freeze_support

# ===============================
# CONFIG
# ===============================
SEQ_LEN = 50         # Secuencia de palabras
BATCH_SIZE = 512
EMB_SIZE = 512
HIDDEN_SIZE = 1024
NUM_LAYERS = 4
DROPOUT = 0.2
EPOCHS = 20          # Entrenar más para mayor coherencia
TEMPERATURE = 0.6
CHECKPOINT_DIR = "checkpoints/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
print("Device:", device)

# ===============================
# CARGA Y TOKENIZACIÓN
# ===============================
files = [
    "Content/01 - The Fellowship Of The Ring.txt",
    "Content/02 - The Two Towers.txt",
    "Content/03 - The Return Of The King.txt"
]

texts = []
def read_text(path):
    try:
        return open(path, encoding="utf-8").read()
    except UnicodeDecodeError:
        return open(path, encoding="latin-1").read()

for i, path in enumerate(files):
    t = read_text(path)
    t = t.replace("\r\n", "\n").strip()
    texts.append(f"\n<BOOK_{i+1}>\n" + t)

full_text = "\n".join(texts)

def tokenize_words(text):
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

tokens = tokenize_words(full_text)
vocab = sorted(set(tokens))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)
print("Número de tokens (palabras):", vocab_size)

# ===============================
# ITERABLE DATASET WORD-LEVEL
# ===============================
class WordIterableDataset(IterableDataset):
    def __init__(self, tokens, seq_len):
        self.data = torch.tensor([word2idx[t] for t in tokens], dtype=torch.long)
        self.seq_len = seq_len

    def __iter__(self):
        for i in range(len(self.data) - self.seq_len):
            x = self.data[i:i+self.seq_len]
            y = self.data[i+1:i+self.seq_len+1]
            yield x, y

# ===============================
# MODELO
# ===============================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        scores = self.v(torch.tanh(self.attn(lstm_out)))
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return context

class WordLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(
            emb_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        context = self.attention(out)
        logits = self.fc(context)
        return logits, hidden

# ===============================
# GENERACIÓN DE TEXTO CON PENALTY
# ===============================
def generate_words(model, start_str, length=200, temp=TEMPERATURE, seq_len=SEQ_LEN):
    model.eval()
    start_tokens = tokenize_words(start_str)
    indices = [word2idx.get(t, 0) for t in start_tokens]
    hidden = None
    for _ in range(length):
        x = torch.tensor(indices[-seq_len:], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, hidden = model(x, hidden)
            logits = logits / temp

            # Penalizar tokens recientes para evitar repetición
            for idx in set(indices[-seq_len:]):
                logits[0, idx] -= 1.0

            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            indices.append(next_idx)
    words = [idx2word[i] for i in indices]
    return ' '.join(words)

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    freeze_support()

    dataset = WordIterableDataset(tokens, SEQ_LEN)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = WordLSTM(vocab_size, EMB_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler(device="cuda")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                logits, _ = model(x)
                loss = criterion(logits, y[:, -1])

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        # Guardar checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"word_lstm_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)

        # Generar texto de ejemplo más largo
        sample_text = generate_words(model, start_str="<BOOK_1>", length=300)
        print(f"--- Ejemplo de texto tras epoch {epoch+1} ---")
        print(sample_text)
        print("------------------------------------------------------\n")
