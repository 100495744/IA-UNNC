import os
import re
import json
import torch

# ===============================
# CONFIG
# ===============================
DATA_DIR = "Content"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

EOS_TOKEN = "<eos>"

# ===============================
# FUNCIONES
# ===============================
def tokenize(text):
    """
    Tokenización por palabras con separación de signos de puntuación
    y reemplazo de saltos de línea por <eos>.
    """
    text = text.lower()
    text = re.sub(r"\n+", f" {EOS_TOKEN} ", text)  # saltos de línea -> <eos>
    # separar signos de puntuación y palabras
    return re.findall(r"\w+|[.,!?;:]", text)

def build_vocab(tokens, special_tokens=[EOS_TOKEN, "<pad>", "<unk>"]):
    """
    Construye vocabulario word2idx e idx2word
    incluyendo tokens especiales.
    """
    vocab = sorted(set(tokens))
    # incluir tokens especiales al inicio
    for tok in special_tokens:
        if tok not in vocab:
            vocab.insert(0, tok)
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

# ===============================
# TOKENIZAR TEXTOS
# ===============================
files = [
    "01 - The Fellowship Of The Ring.txt",
    "02 - The Two Towers.txt",
    "03 - The Return Of The King.txt"
]

tokens = []
for f in files:
    path = os.path.join(DATA_DIR, f)
    try:
        txt = open(path, encoding="utf-8").read()
    except UnicodeDecodeError:
        txt = open(path, encoding="latin-1").read()
    tokens.extend(tokenize(txt))

# ===============================
# VOCABULARIO Y CODIFICACIÓN
# ===============================
word2idx, idx2word = build_vocab(tokens, special_tokens=[EOS_TOKEN, "<pad>", "<unk>"])

encoded = torch.tensor([word2idx.get(w, word2idx["<unk>"]) for w in tokens], dtype=torch.long)

# ===============================
# GUARDAR
# ===============================
torch.save(encoded, os.path.join(OUT_DIR, "tokens.pt"))
json.dump(word2idx, open(os.path.join(OUT_DIR, "vocab.json"), "w"))
json.dump(idx2word, open(os.path.join(OUT_DIR, "idx2word.json"), "w"))

# metadata del tokenizador
json.dump({
    "lowercase": True,
    "regex": r"\w+|[.,!?;:]",
    "eos_token": EOS_TOKEN
}, open(os.path.join(OUT_DIR, "tokenizer.json"), "w"))

print("Tokenización completada")
print("Vocab size:", len(word2idx))
print("Total tokens:", len(encoded))
