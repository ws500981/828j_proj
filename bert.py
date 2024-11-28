import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tiktoken

# Hyperparameters
batch_size = 32
head_size = 4
n_embd = 16
n_heads = 4
n_layers = 2
dropout = 0.2
learning_rate = 1e-4

eval_interval = 100
max_iters = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1337)

# Tokenizer and Dataset Preparation
class BinaryDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer.encode(text)
        x = torch.tensor(tokens, dtype=torch.long)
        return x, torch.tensor(label, dtype=torch.float)

# Self-Attention Head with Masking
# Self-Attention Head without Causal Masking
class Head(nn.Module):
    def __init__(self, head_size, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * (1.0 / (head_size ** 0.5))  # (B, T, T)
        if mask is not None:
            mask_combined = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, T, T)
            large_neg = -1e9
            wei = wei.masked_fill(mask_combined == 0, large_neg)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v  # (B, T, head_size)


# Multi-Head Attention without block_size
class MultiHead(nn.Module):
    def __init__(self, head_size, n_embd, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = [head(x, mask=mask) for head in self.heads]  # List of (B, T, head_size)
        x = torch.cat(out, dim=-1)  # (B, T, n_heads * head_size)
        x = self.dropout(self.proj(x))
        return x


# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Block without block_size
class Block(nn.Module):
    def __init__(self, head_size, n_embd, n_heads):
        super().__init__()
        self.sa_head = MultiHead(head_size, n_embd, n_heads)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa_head(self.ln1(x), mask=mask)
        x = x + self.ff(self.ln2(x))
        return x


# Estimate Loss Function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        loader = train_loader if split == 'train' else val_loader
        losses = []
        for xb, yb, mask in loader:
            xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
            logits = model(xb, mask).squeeze()
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# GPTBinaryClassifier with mean pooling
class GPTBinaryClassifier(nn.Module):
    def __init__(self, vocab_size, n_embd, n_heads, n_layers, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(1000, n_embd)  # Use a large enough max length
        self.blocks = nn.ModuleList(
            [Block(head_size, n_embd, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.classifier_head = nn.Linear(n_embd, 1)  # Single logit output for binary classification

    def forward(self, idx, mask):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_indices = torch.arange(T, device=idx.device)
        pos_embd = self.positional_embedding_table(pos_indices)  # (T, n_embd)
        x = tok_embd + pos_embd  # (B, T, n_embd)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln_f(x)  # (B, T, n_embd)
        # Apply mask before pooling
        x = x * mask.unsqueeze(-1)
        x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True)  # Mean pooling over valid tokens
        logits = self.classifier_head(x)  # (B, 1)
        return logits

def collate_fn(batch):
    texts, labels = zip(*batch)
    # Get the lengths of each sequence
    lengths = [len(x) for x in texts]
    max_length = max(lengths)
    # Pad sequences to the max length in the batch
    padded_texts = [torch.cat([x, torch.zeros(max_length - len(x), dtype=torch.long)]) for x in texts]
    # Stack into tensors
    x = torch.stack(padded_texts)
    y = torch.tensor(labels, dtype=torch.float)
    # Create padding mask: 1 for real tokens, 0 for padding tokens
    mask = torch.stack([torch.cat([torch.ones(len(x_i)), torch.zeros(max_length - len(x_i))]) for x_i in texts])
    return x, y, mask

# Training
if __name__ == "__main__":
    # Load the CSV file
    df = pd.read_csv("/home/wuw15/data_dir/cwproj/dataset_9062.csv")
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    labels = [float(label) for label in labels]  # Ensure labels are floats

    # Train-test split
    train_ratio = 0.8
    n_train = int(len(texts) * train_ratio)
    train_texts, val_texts = texts[:n_train], texts[n_train:]
    train_labels, val_labels = labels[:n_train], labels[n_train:]

    # Tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Prepare datasets and dataloaders
    train_dataset = BinaryDataset(train_texts, train_labels, tokenizer)
    val_dataset = BinaryDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model and optimizer
    vocab_size = tokenizer.n_vocab
    model = GPTBinaryClassifier(vocab_size, n_embd, n_heads, n_layers, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for iter in range(max_iters):
        model.train()
        for xb, yb, mask in train_loader:
            xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(xb, mask).squeeze()
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        # Evaluate periodically
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"Iter {iter}, Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")


    # Save the trained model
    torch.save(model.state_dict(), "/home/wuw15/data_dir/cwproj/gpt_binary_classifier2.pth")
