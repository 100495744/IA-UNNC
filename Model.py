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
SEQ_LEN = 30         # Secuencia de palabras
BATCH_SIZE = 512     # Batch grande para saturar GPU
EMB_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 3
EPOCHS = 10
TEMPERATURE = 0.8
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
print("Device:", device)

# ===============================
# CARGA Y TOKENIZACIÓN POR PALABRAS
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

# Tokenización por palabras y puntuación
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
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(
            emb_size, hidden_size, num_layers=num_layers, batch_first=True
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
# FUNCIÓN DE GENERACIÓN DE TEXTO
# ===============================
def generate_words(model, start_str, length=50, temp=TEMPERATURE):
    model.eval()
    start_tokens = tokenize_words(start_str)
    indices = [word2idx.get(t, 0) for t in start_tokens]
    hidden = None
    for _ in range(length):
        x = torch.tensor(indices[-SEQ_LEN:], dtype=torch.long).unsqueeze(0).to(device)
        logits, hidden = model(x, hidden)
        probs = torch.softmax(logits / temp, dim=-1)
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
        num_workers=0,  # Windows-safe
        pin_memory=True
    )

    model = WordLSTM(vocab_size, EMB_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler(device="cuda")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0  # Contador de batches

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

        # Generar texto de ejemplo
        sample_text = generate_words(model, start_str="<BOOK_1>", length=50)
        print(f"--- Ejemplo de texto tras epoch {epoch+1} ---")
        print(sample_text)
        print("------------------------------------------------------\n")
