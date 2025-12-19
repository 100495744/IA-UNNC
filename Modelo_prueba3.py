import os
import re
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ===============================
# 1. CONFIGURATION
# ===============================
DATA_DIR = "Content"           
CHECKPOINT_DIR = "checkpoints" 
OUTPUT_DIR = "stories"        
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# 2. DATA PROCESSING
# ===============================
def clean_and_tokenize(text):
    print(f"[PROCESS] Cleaning text and extracting tokens...")
    text = text.lower()
    # Ensure punctuation are separate tokens for better grammar learning
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    tokens = text.split()
    print(f"[INFO] Total tokens extracted: {len(tokens):,}")
    return tokens

def build_vocab(all_tokens, min_freq=2):
    print(f"[PROCESS] Building vocabulary (min_freq={min_freq})...")
    counts = Counter(all_tokens)
    # Special tokens for control logic
    vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + \
            [word for word, freq in counts.items() if freq >= min_freq]
    
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for word, i in word2idx.items()}
    print(f"[INFO] Vocabulary size: {len(word2idx):,} unique words.")
    return word2idx, idx2word

# ===============================
# 3. MODEL ARCHITECTURE
# ===============================
class StoryModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, num_layers=2):
        super().__init__()
        print(f"[MODEL] Initializing LSTM+Attention (Embed:{embed_dim}, Hidden:{hidden_dim})")
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=0.3
        )
        
        # Attention scores the importance of each word in the sequence
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Projection layer to fix the dimension mismatch (1024 -> 512)
        self.project_context = nn.Linear(hidden_dim * 2, embed_dim)
        
        # Output layer (Tied weights with embedding)
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)
        self.fc.weight = self.embedding.weight
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.layer_norm(lstm_out)

        # Attention Mechanism
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        # Combine last hidden state and the context vector
        combined = torch.cat([lstm_out[:, -1, :], context], dim=1)
        projected = torch.relu(self.project_context(combined))
        
        logits = self.fc(projected)
        return logits, hidden

# ===============================
# 4. DATASET & GENERATION
# ===============================
class StoryDataset(Dataset):
    def __init__(self, encoded_tokens, seq_len):
        self.data = encoded_tokens
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, i):
        return (torch.tensor(self.data[i : i + self.seq_len]), 
                torch.tensor(self.data[i + self.seq_len]))

def generate_autonomous(model, word2idx, idx2word, max_len=200, temp=0.8, top_p=0.9):
    model.eval()
    generated_idxs = [word2idx.get("<SOS>", 1)]
    hidden = None
    
    with torch.no_grad():
        for _ in range(max_len):
            input_tensor = torch.tensor([[generated_idxs[-1]]], device=DEVICE)
            logits, hidden = model(input_tensor, hidden)
            logits = logits / temp
            
            # Simple repetition penalty
            for token in set(generated_idxs[-15:]):
                logits[0, token] -= 1.0

            # Nucleus Sampling (Top-P)
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

    # Cleanup formatting
    raw_words = [idx2word.get(i, "") for i in generated_idxs if i > 3]
    text = " ".join(raw_words)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text) # Fix punctuation spacing
    return text.capitalize()

# ===============================
# 5. THE RUN PIPELINE
# ===============================
def run_pipeline():
    print("="*50)
    print(f"STARTING SYSTEM ON: {DEVICE.upper()}")
    print("="*50)
    
    # --- Load Files ---
    if not os.path.exists(DATA_DIR):
        print(f"[FATAL ERROR] Folder '{DATA_DIR}' not found. Please create it.")
        return

    all_text = ""
    txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    print(f"[INFO] Found {len(txt_files)} text files.")

    for file in txt_files:
        print(f"[FILE] Reading {file}...")
        with open(os.path.join(DATA_DIR, file), 'r', encoding='utf-8', errors='ignore') as f:
            all_text += f.read() + " <EOS> "

    # --- Prepare Data ---
    tokens = clean_and_tokenize(all_text)
    word2idx, idx2word = build_vocab(tokens)
    encoded = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
    
    # --- Setup Model ---
    model = StoryModel(len(word2idx)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # --- Training Phases ---
    phases = [
        {"name": "Local Grammar", "seq": 35, "epochs": 6},
        {"name": "Story Flow", "seq": 90, "epochs": 4}
    ]

    

    for p_idx, phase in enumerate(phases):
        print(f"\n[PHASE {p_idx+1}] Starting: {phase['name']}")
        print(f"[INFO] Sequence Length: {phase['seq']} | Epochs: {phase['epochs']}")
        
        dataset = StoryDataset(encoded, phase['seq'])
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        num_batches = len(loader)

        for epoch in range(phase['epochs']):
            model.train()
            total_loss = 0
            
            for b_idx, (x, y) in enumerate(loader):
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.2)
                optimizer.step()
                total_loss += loss.item()
                
                # Batch status print
                if b_idx % 100 == 0:
                    print(f"  > Epoch {epoch+1} | Batch {b_idx}/{num_batches} | Current Loss: {loss.item():.4f}", end='\r')
            
            avg_loss = total_loss / num_batches
            print(f"\n[FINISH] Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
            
            # Show the user what the AI is thinking
            preview = generate_autonomous(model, word2idx, idx2word, max_len=30)
            print(f"[PREVIEW] {preview}...")

    # --- Final Generation ---
    print("\n" + "="*50)
    print("FINAL STORY GENERATION (Autonomous Mode)")
    print("="*50)
    
    for i in range(3):
        print(f"\n[STORY {i+1} START]")
        story = generate_autonomous(model, word2idx, idx2word, max_len=150)
        print(story)
        
        # Save to file
        save_path = os.path.join(OUTPUT_DIR, f"story_{i+1}.txt")
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(story)
        print(f"[SYSTEM] Story saved to {save_path}")

    print("\n[SUCCESS] Pipeline completed.")

if __name__ == "__main__":
    run_pipeline()