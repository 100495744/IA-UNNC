import os
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ===============================
# 1. CONFIGURACIÓN (Optimizado para GPU)
# ===============================
DATA_DIR = "Content"           
CHECKPOINT_DIR = "checkpoints" 
OUTPUT_DIR = "stories"        
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# 2. PROCESAMIENTO DE TEXTO
# ===============================
def clean_and_tokenize(text):
    print(f"[PROCESO] Limpiando texto de Tolkien...")
    text = text.lower()
    # Separar puntuación para que el modelo aprenda a pausar las oraciones
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    tokens = text.split()
    return tokens

def build_vocab(all_tokens, min_freq=2):
    """ Crea un diccionario para convertir palabras en números y viceversa """
    print(f"[PROCESO] Construyendo vocabulario...")
    counts = Counter(all_tokens)
    vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + \
            [word for word, freq in counts.items() if freq >= min_freq]
    
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for word, i in word2idx.items()}
    return word2idx, idx2word

# ===============================
# 3. MODELO PROFUNDO (LSTM + ATENCIÓN)
# ===============================
class StoryModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # LSTM con Dropout para evitar que el modelo solo repita frases hechas
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=0.3)
        
        # Permite al modelo 'mirar atrás' y recordar palabras importantes de la frase.
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.project_context = nn.Linear(hidden_dim * 2, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)

        # Esto hace que el modelo aprenda el doble de rápido con la mitad de memoria.
        self.fc.weight = self.embedding.weight # Weight Tying (Truco de eficiencia)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.layer_norm(lstm_out)

        # Cálculo de Pesos de Atención
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)

        # Combinamos la memoria del LSTM con el contexto de la atención
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        combined = torch.cat([lstm_out[:, -1, :], context], dim=1)
        projected = torch.relu(self.project_context(combined))
        return self.fc(projected), hidden

# ===============================
# 4. DATASET CON STRIDE VARIABLE
# ===============================
class StoryDataset(Dataset):
    def __init__(self, encoded_tokens, seq_len, stride=1):
        self.data = encoded_tokens
        self.seq_len = seq_len

        # El 'stride' determina cuánto saltamos entre ejemplos. 1 = leer palabra por palabra.
        self.indices = range(0, len(encoded_tokens) - seq_len, stride)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        # X: bloque de texto de entrada. Y: la palabra que sigue inmediatamente.
        return (torch.tensor(self.data[idx : idx + self.seq_len]), 
                torch.tensor(self.data[idx + self.seq_len]))

# ===============================
# 5. GENERACIÓN NUCLEUS SAMPLING
# ===============================
def generate_autonomous(model, word2idx, idx2word, max_len=120, temp=0.75, top_p=0.9):
    model.eval()
    generated_idxs = [word2idx.get("<SOS>", 1)] # Empezamos con el token de inicio
    hidden = None
    
    with torch.no_grad(): # No calculamos gradientes para ir más rápido
        for _ in range(max_len):
            input_tensor = torch.tensor([[generated_idxs[-1]]], device=DEVICE)
            logits, hidden = model(input_tensor, hidden)
            logits = logits / temp # Temperatura: >1.0 es caótico/creativo, <1.0 es conservador/seguro.
            
            # Penalización de repetición dinámica
            for token in set(generated_idxs[-15:]):
                logits[0, token] -= 2.0

            # Filtro Nucleus (Top-P)
            # Solo elegimos entre las palabras más probables.
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            indices_to_remove = cumulative_probs > top_p
            indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
            indices_to_remove[..., 0] = 0
            
            sorted_logits[indices_to_remove] = -float('Inf')
            probs = torch.softmax(sorted_logits, dim=-1)
            next_token = sorted_indices[0, torch.multinomial(probs, 1).item()].item()
            
            generated_idxs.append(next_token)
            if idx2word.get(next_token) == "<EOS>": break

    # Limpieza de tokens especiales y formato
    words = [idx2word.get(i, "") for i in generated_idxs if i > 3]
    raw_story = " ".join(words)
    # Corregir espacios en puntuación: "hola ." -> "hola."
    clean_story = re.sub(r'\s+([.,!?;:])', r'\1', raw_story)
    return clean_story.capitalize()

# ===============================
# 6. PIPELINE DE ENTRENAMIENTO
# ===============================
def run_pipeline():
    print("="*50)
    print(f"SISTEMA INICIADO EN: {DEVICE.upper()}")
    print("="*50)

    torch.backends.cudnn.benchmark = True
    
    # Cargar textos
    all_text = ""
    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            print(f"[CARGANDO] {file}")
            with open(os.path.join(DATA_DIR, file), 'r', encoding='utf-8', errors='ignore') as f:
                all_text += f.read() + " <EOS> "

    tokens = clean_and_tokenize(all_text)
    word2idx, idx2word = build_vocab(tokens)
    encoded = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
    
    # Modelo con mayor capacidad
    #model = StoryModel(len(word2idx), embed_dim=512, hidden_dim=512).to(DEVICE)
    model = StoryModel(len(word2idx), embed_dim=512, hidden_dim=512, num_layers=3).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss() #La métrica que castiga al modelo cuando se equivoca de palabra

    scaler = torch.amp.GradScaler('cuda') if DEVICE == "cuda" else None

    # Empezamos con secuencias cortas y saltos para aprender rápido
    # Terminamos con secuencias largas y paso a paso para aprender coherencia
    phases = [
        {"name": "1. Estructura Básica", "seq": 32, "epochs": 4, "stride": 8},
        {"name": "2. Coherencia Media", "seq": 64, "epochs": 4, "stride": 4},
        {"name": "3. Pulido Final (Lento pero preciso)", "seq": 128, "epochs": 10, "stride": 2}
    ]    

    for phase in phases:
        print(f"\n>>> INICIANDO: {phase['name']}")
        dataset = StoryDataset(encoded, phase['seq'], stride=phase['stride'])
        # pin_memory acelera el paso de datos de la RAM a la GPU
        #loader = DataLoader(dataset, batch_size=512, shuffle=True, pin_memory=True)
        loader = DataLoader(
            dataset, 
            batch_size=512, # Aumentamos batch porque FP16 consume menos memoria
            shuffle=True, 
            pin_memory=True,
            num_workers=4,  # Usa 4 núcleos de CPU para cargar datos
            persistent_workers=True
        )
        
        for epoch in range(phase['epochs']):
            model.train()
            for b_idx, (x, y) in enumerate(loader):
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True) # Más eficiente que zero_grad()

                # --- CAMBIO PARA VELOCIDAD: autocast ---
                # Ejecuta la pasada hacia adelante en precisión mixta
                if DEVICE == "cuda":
                    with torch.amp.autocast('cuda'):
                        logits, _ = model(x)
                        loss = criterion(logits, y)
                    
                    # Escala la pérdida y hace backprop
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, _ = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

                if b_idx % 100 == 0:
                    print(f"  E{epoch+1} | B{b_idx}/{len(loader)} | Loss: {loss.item():.4f}", end='\r')

    # Guardar el modelo final
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "tolkien_model_v2.pt"))
    
    print("\n" + "="*30 + "\nRESULTADOS FINALES\n" + "="*30)
    for i in range(3):
        story = generate_autonomous(model, word2idx, idx2word, max_len=150, temp=0.7)
        print(f"\nHISTORIA {i+1}:\n{story}")
        with open(os.path.join(OUTPUT_DIR, f"historia_pro_{i+1}.txt"), 'w') as f:
            f.write(story)

if __name__ == "__main__":
    run_pipeline()