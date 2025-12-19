import os
import re
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json

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

# Cargar datos si ya existen
try:
    encoded = torch.load(os.path.join(DATA_DIR, "tokens.pt"), map_location="cpu")
    word2idx = json.load(open(os.path.join(DATA_DIR, "vocab.json")))
    idx2word = {int(v): k for k, v in word2idx.items()}
    vocab_size = len(word2idx)
    print(f"Vocab size: {vocab_size}")
    print(f"Total tokens: {len(encoded)}")
except:
    print("Preprocesando archivos...")
    files = [
        "01 - The Fellowship Of The Ring.txt",
        "02 - The Two Towers.txt",
        "03 - The Return Of The King.txt"
    ]
    
    all_text = ""
    for file in files:
        path = os.path.join(DATA_DIR, file)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                all_text += f.read() + " "
        else:
            print(f"Warning: {path} not found")
    
    # Crear vocabulario
    tokens = tokenize(all_text)
    word_counts = Counter(tokens)
    vocab = ["<unk>", "<pad>", "<eos>"] + [word for word, count in word_counts.items() if count >= 5]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    # Codificar tokens
    encoded = [word2idx.get(token, word2idx["<unk>"]) for token in tokens]
    
    # Guardar
    torch.save(torch.tensor(encoded), os.path.join(DATA_DIR, "tokens.pt"))
    with open(os.path.join(DATA_DIR, "vocab.json"), 'w') as f:
        json.dump(word2idx, f)
    
    vocab_size = len(vocab)
    print(f"Created vocab: {vocab_size} words")

# ===============================
# DATASET (corregido)
# ===============================
class WordDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        self.total_sequences = len(data) - seq_len
        
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return torch.tensor(x), torch.tensor(y)

# ===============================
# MODELO LSTM + MULTIHEAD ATTENTION
# ===============================
class MultiheadAttentionModule(nn.Module):
    def __init__(self, hidden_size, n_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=n_heads, 
            batch_first=True,
            dropout=0.1
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        return self.ln(x + self.dropout(attn_output))

class WordLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size=256, hidden_size=512, layers=2, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(
            emb_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=0.3 if layers > 1 else 0
        )
        self.attn = MultiheadAttentionModule(hidden_size, n_heads)
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Weight tying (opcional, puede ayudar)
        # self.fc.weight = self.embedding.weight
        
    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        lstm_out, hidden = self.lstm(emb, hidden)
        attn_out = self.attn(lstm_out)
        normalized = self.ln(attn_out)
        logits = self.fc(normalized)
        return logits, hidden

# ===============================
# GENERACIÓN MEJORADA
# ===============================
def generate(model, start_words, idx2word, word2idx, length=100, temp=0.8, top_p=0.9, device="cpu"):
    model.eval()
    
    # Procesar palabras iniciales
    if isinstance(start_words, str):
        words = start_words.lower().split()
    else:
        words = start_words
    
    # Convertir a índices
    idxs = []
    for w in words:
        if w in word2idx:
            idxs.append(word2idx[w])
        else:
            # Intentar con sin puntuación
            w_clean = re.sub(r'[^\w\s]', '', w)
            if w_clean in word2idx:
                idxs.append(word2idx[w_clean])
            else:
                idxs.append(word2idx.get("<unk>", 0))
    
    hidden = None
    generated = idxs.copy()
    
    with torch.no_grad():
        # Procesar prompt completo para obtener estado inicial
        if len(idxs) > 1:
            x = torch.tensor([idxs[:-1]], device=device)
            _, hidden = model(x, None)
        
        # Generar nuevas palabras
        current_token = torch.tensor([[idxs[-1]]], device=device)
        
        for _ in range(length):
            logits, hidden = model(current_token, hidden)
            logits = logits[0, -1, :] / temp
            
            # Penalización por repetición suave
            recent = generated[-10:]  # Ventana más pequeña
            counts = Counter(recent)
            for idx, count in counts.items():
                if count > 1:
                    logits[idx] -= count * 0.5
            
            # Nucleus sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            logits[sorted_indices[sorted_indices_to_remove]] = -float('Inf')
            
            # Muestreo
            probs = torch.softmax(logits, dim=-1)
            try:
                next_idx = torch.multinomial(probs, 1).item()
            except:
                next_idx = torch.argmax(probs).item()
            
            generated.append(next_idx)
            current_token = torch.tensor([[next_idx]], device=device)
            
            # Detener en EOS o si se repite mucho
            if idx2word.get(next_idx) == "<eos>":
                break
            if len(generated) > 20 and len(set(generated[-10:])) < 4:
                break
    
    # Convertir a texto
    result_words = []
    for i in generated:
        word = idx2word.get(i, "")
        if word == "<eos>":
            result_words.append("\n")
        elif word not in ["<unk>", "<pad>"]:
            result_words.append(word)
    
    # Formatear el texto
    text = " ".join(result_words)
    
    # Limpiar puntuación
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'(\w)\s+\n', r'\1\n', text)
    text = re.sub(r'\n\s+', '\n', text)
    
    # Capitalizar oraciones
    sentences = text.split('. ')
    sentences = [s.capitalize() for s in sentences]
    text = '. '.join(sentences)
    
    return text.strip()

# ===============================
# ENTRENAMIENTO
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
    print(f"\n===== {phase_name} =====")
    print(f"SEQ_LEN={seq_len} | LR={lr} | TEMP={temperature}")
    
    # Crear dataset
    dataset = WordDataset(encoded, seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Crear modelo
    model = WordLSTM(
        vocab_size=vocab_size,
        emb_size=256,
        hidden_size=512,
        layers=2
    ).to(DEVICE)
    
    # Cargar checkpoint si existe
    if start_checkpoint and os.path.exists(start_checkpoint):
        try:
            model.load_state_dict(torch.load(start_checkpoint))
            print(f"Checkpoint cargado: {start_checkpoint}")
        except:
            print("No se pudo cargar checkpoint, comenzando desde cero")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        batches = 0
        
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            
            # Reshape para pérdida
            loss = criterion(
                logits.view(-1, vocab_size),
                y.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
            if batches % 100 == 0:
                print(f"  Batch {batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batches
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
        
        # Generar muestra
        print("\n--- Muestra generada ---")
        sample = generate(model, "Frodo and Sam", idx2word, word2idx, 
                         length=80, temp=temperature, device=DEVICE)
        print(sample[:500])
        print("--- Fin muestra ---\n")
        
        # Guardar checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{phase_name}_best.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint guardado: {checkpoint_path}")
    
    return os.path.join(CHECKPOINT_DIR, f"{phase_name}_best.pt")

# ===============================
# FUNCIÓN PARA PROBAR
# ===============================
def test_generation(model_path=None, prompt="The ring"):
    if model_path and os.path.exists(model_path):
        model = WordLSTM(vocab_size, 256, 512, 2).to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        print(f"Modelo cargado desde {model_path}")
    else:
        print("Usando modelo aleatorio")
        model = WordLSTM(vocab_size, 256, 512, 2).to(DEVICE)
    
    print("\n" + "="*50)
    print("PRUEBA DE GENERACIÓN")
    print("="*50)
    
    prompts = [
        "Frodo and Sam",
        "Gandalf said",
        "In the land of Mordor",
        "The ring of power",
        "Aragorn the king"
    ]
    
    for p in prompts:
        print(f"\nPrompt: '{p}'")
        print("-" * 40)
        text = generate(model, p, idx2word, word2idx, 
                       length=150, temp=0.7, top_p=0.92, device=DEVICE)
        print(text)
        print()

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENTRENANDO MODELO DE GENERACIÓN DE PÁRRAFOS")
    print("="*60)
    
    # Probar generación antes de entrenar
    test_generation()
    
    # FASE 1 – Entrenamiento básico
    ckpt1 = train_phase(
        phase_name="phase1_local",
        seq_len=32,  # Más corto para empezar
        epochs=10,
        batch_size=64,
        lr=3e-4,
        temperature=0.8
    )
    
    # FASE 2 – Entrenamiento con contexto más largo
    ckpt2 = train_phase(
        phase_name="phase2_long",
        seq_len=64,
        epochs=8,
        batch_size=32,
        lr=1e-4,
        temperature=0.7,
        start_checkpoint=ckpt1
    )
    
    # FASE 3 – Refinamiento final
    ckpt3 = train_phase(
        phase_name="phase3_final",
        seq_len=128,
        epochs=6,
        batch_size=16,
        lr=5e-5,
        temperature=0.6,
        start_checkpoint=ckpt2
    )
    
    # Probar el modelo final
    print("\n" + "="*60)
    print("MODELO FINAL - PRUEBA COMPLETA")
    print("="*60)
    test_generation(ckpt3, "The journey of the ring")