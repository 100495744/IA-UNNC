import os
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.amp import autocast, GradScaler # For faster training (FP16)

# ===============================
# 1. CONFIGURATION (GPU Optimized)
# ===============================
DATA_DIR = "Content"           
CHECKPOINT_DIR = "checkpoints" 
OUTPUT_DIR = "stories"         
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Technical improvement: Optimizes convolution/LSTM algorithms for your specific GPU
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

# ===============================
# 2. TEXT PROCESSING
# ===============================
def clean_and_tokenize(text):
    print(f"[PROCESS] Cleaning Tolkien text...")
    text = text.lower()
    # Separate punctuation so the model learns to pause sentences
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    tokens = text.split()
    return tokens

def build_vocab(all_tokens, min_freq=2):
    """ Creates a dictionary to convert words to numbers and vice versa """
    print(f"[PROCESS] Building vocabulary...")
    counts = Counter(all_tokens)
    vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + \
            [word for word, freq in counts.items() if freq >= min_freq]
    
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for word, i in word2idx.items()}
    return word2idx, idx2word

# ===============================
# 3. DEEP MODEL (LSTM + ATTENTION)
# ===============================
class StoryModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # LSTM with Dropout to prevent the model from just repeating fixed phrases
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=0.3)
        
        # Allows the model to 'look back' and remember important words in the sentence
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.project_context = nn.Linear(hidden_dim * 2, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight Tying: Makes the model learn twice as fast with half the memory (Efficiency trick)
        self.fc.weight = self.embedding.weight 
        
        # NEW: Normalization layers to prevent Loss from plateauing
        self.ln_lstm = nn.LayerNorm(hidden_dim)
        self.ln_context = nn.LayerNorm(embed_dim)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.ln_lstm(lstm_out) # Stabilizes LSTM output

        # Attention Weights calculation
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)

        # Combine LSTM memory with attention context
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        combined = torch.cat([lstm_out[:, -1, :], context], dim=1)
        # Apply normalization before the final layer
        projected = torch.relu(self.ln_context(self.project_context(combined)))
        return self.fc(projected), hidden

# ===============================
# 4. DATASET WITH VARIABLE STRIDE
# ===============================
class StoryDataset(Dataset):
    def __init__(self, encoded_tokens, seq_len, stride=1):
        self.data = encoded_tokens
        self.seq_len = seq_len

        # 'stride' determines how much we jump between examples. 1 = read word by word.
        self.indices = range(0, len(encoded_tokens) - seq_len, stride)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        # X: input text block. Y: the word that follows immediately.
        return (torch.tensor(self.data[idx : idx + self.seq_len], dtype=torch.long), 
                torch.tensor(self.data[idx + self.seq_len], dtype=torch.long))

# ===============================
# 5. NUCLEUS SAMPLING GENERATION
# ===============================
def generate_autonomous(model, word2idx, idx2word, max_len=120, temp=0.75, top_p=0.9):
    model.eval()
    generated_idxs = [word2idx.get("<SOS>", 1)] # Start with the Start-of-Sentence token
    hidden = None
    
    with torch.no_grad(): # Disable gradients for faster inference
        for _ in range(max_len):
            # Optimization: Autocast for consistent generation with training precision
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                input_tensor = torch.tensor([[generated_idxs[-1]]], device=DEVICE)
                logits, hidden = model(input_tensor, hidden)
            
            logits = logits / temp # Temperature: >1.0 is creative/chaotic, <1.0 is safe/conservative.
            
            # Dynamic repetition penalty
            for token in set(generated_idxs[-15:]):
                logits[0, token] -= 2.0

            # Nucleus Filtering (Top-P)
            # Only choose from the most probable words.
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

    # Clean special tokens and format output
    words = [idx2word.get(i, "") for i in generated_idxs if i > 3]
    raw_story = " ".join(words)
    # Fix spaces in punctuation: "hello ." -> "hello."
    clean_story = re.sub(r'\s+([.,!?;:])', r'\1', raw_story)
    return clean_story.capitalize()

# ===============================
# 6. TRAINING PIPELINE
# ===============================
def run_pipeline():
    print("="*50)
    print(f"SYSTEM STARTED ON: {DEVICE.upper()}")
    print("="*50)
    
    # Load texts
    all_text = ""
    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            print(f"[LOADING] {file}")
            with open(os.path.join(DATA_DIR, file), 'r', encoding='utf-8', errors='ignore') as f:
                all_text += f.read() + " <EOS> "

    tokens = clean_and_tokenize(all_text)
    word2idx, idx2word = build_vocab(tokens)
    encoded = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
    
    # Higher capacity model
    model = StoryModel(len(word2idx), embed_dim=512, hidden_dim=512, num_layers=3).to(DEVICE)
    # Lower initial LR and added weight_decay for stability
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    # Scheduler: if loss doesn't drop in 1 epoch, reduce LR by half
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    criterion = nn.CrossEntropyLoss() # The metric that penalizes the model for wrong word predictions

    # Mixed precision tool: trains much faster on GPU
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # Training phases: Start with short sequences to learn syntax, finish with long sequences for coherence
    phases = [
        {"name": "1. Basic Structure", "seq": 32, "epochs": 4, "stride": 8},
        {"name": "2. Medium Coherence", "seq": 64, "epochs": 4, "stride": 4},
        {"name": "3. Final Polish (Slow but precise)", "seq": 128, "epochs": 10, "stride": 2}
    ]    

    for phase in phases:
        print(f"\n>>> STARTING: {phase['name']}")
        dataset = StoryDataset(encoded, phase['seq'], stride=phase['stride'])
        # pin_memory speeds up data transfer from RAM to GPU
        loader = DataLoader(
            dataset, 
            batch_size=256, 
            shuffle=True, 
            pin_memory=(DEVICE == "cuda"),
            num_workers=0 # Stable for Google Colab environments
        )
        
        for epoch in range(phase['epochs']):
            model.train()
            total_loss = 0
            for b_idx, (x, y) in enumerate(loader):
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True) # More efficient than zero_grad()

                # Mixed Precision Training (FP16)
                with autocast(device_type='cuda' if DEVICE == "cuda" else 'cpu'):
                    # Prediction and error calculation
                    logits, _ = model(x)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward() # Backpropagation: model learns from mistakes
                
                # Gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                
                if b_idx % 100 == 0:
                    print(f"  E{epoch+1} | B{b_idx}/{len(loader)} | Loss: {loss.item():.4f}", end='\r')
            
            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss) # Update scheduler with average loss
            print(f"\n[OK] Epoch {epoch+1} Finished. Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            # Generate a small sample to see intelligence evolution
            print(f"PREVIEW: {generate_autonomous(model, word2idx, idx2word, max_len=30)}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "tolkien_model_v2.pt"))
    
    print("\n" + "="*30 + "\nGENERATING LONG STORIES\n" + "="*30)
    for i in range(3):
        # Increase max_len to 600 for multiple paragraphs
        story = generate_autonomous(model, word2idx, idx2word, max_len=600, temp=0.7)
        
        # Paragraph logic: insert a newline every 4 sentences
        sentences = story.split('.')
        parraphed_story = ""
        for j, sentence in enumerate(sentences):
            if sentence.strip():
                parraphed_story += sentence.strip() + ". "
                if (j + 1) % 4 == 0: # New paragraph every 4 dots
                    parraphed_story += "\n\n"
        
        print(f"\nHISTORY {i+1}:\n{parraphed_story}")
        
        # Save to output directory
        with open(os.path.join(OUTPUT_DIR, f"long_story_{i+1}.txt"), "w") as f:
            f.write(parraphed_story)

if __name__ == "__main__":
    run_pipeline()
