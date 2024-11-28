import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
import os
import glob

# 1. Prepare your dataset
# Replace this with your actual data loading logic
df = pd.read_csv("/home/wuw15/data_dir/cwproj/dataset.csv")

# 2. Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# 3. Load the tokenizer and tokenize the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(
    list(train_texts),
    truncation=True,
    padding=True,
    max_length=128
)
val_encodings = tokenizer(
    list(val_texts),
    truncation=True,
    padding=True,
    max_length=128
)

# 4. Create a custom Dataset class
class BinaryClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.values  # Convert to NumPy array if it's a pandas Series

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.labels)

# 5. Create the datasets
train_dataset = BinaryClassificationDataset(train_encodings, train_labels)
val_dataset = BinaryClassificationDataset(val_encodings, val_labels)

# 6. Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 7. Move the model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 8. Create the DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 9. Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)

# Total number of training steps
total_steps = len(train_loader) * 3  # epochs = 3

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Define paths for saving checkpoints
checkpoint_dir = "/home/wuw15/data_dir/cwproj/bert_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_pattern = os.path.join(checkpoint_dir, "bert_checkpoint_epoch*.pth")
def extract_epoch_num(filename):
    basename = os.path.basename(filename)
    match = re.search(r'bert_checkpoint_epoch(\d+)\.pth', basename)
    if match:
        return int(match.group(1))
    else:
        return None
checkpoint_files = glob.glob(checkpoint_pattern)

checkpoint_epochs = []
for file in checkpoint_files:
    epoch_num = extract_epoch_num(file)
    if epoch_num is not None:
        checkpoint_epochs.append((epoch_num, file))

checkpoint_epochs.sort(key=lambda x: x[0])


# Check if a checkpoint exists before training
if checkpoint_epochs:
    latest_epoch, latest_checkpoint = checkpoint_epochs[-1]
    print(f"Loading checkpoint '{latest_checkpoint}' from epoch {latest_epoch}")
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    model.to(device)  # Ensure model is on the correct device
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("No checkpoint found. Starting training from scratch.")
    start_epoch = 0


# 10. Training loop with checkpoint saving
epochs = 5
best_val_accuracy = 0.0  # For tracking the best model

for epoch in range(start_epoch, epochs):
    # Training
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch} Training Loss: {avg_train_loss}")
    
    # Validation
    model.eval()
    val_labels_list = []
    val_preds_list = []
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_val_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            val_labels_list.extend(labels.cpu().numpy())
            val_preds_list.extend(preds.cpu().numpy())
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels_list, val_preds_list)
    print(f"Epoch {epoch} Validation Loss: {avg_val_loss}")
    print(f"Epoch {epoch} Validation Accuracy: {val_accuracy}")
    
    # Save checkpoint at the end of each epoch
    checkpoint_path = os.path.join(checkpoint_dir, f"bert_checkpoint_epoch{epoch}.pth")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'val_accuracy': val_accuracy,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at '{checkpoint_path}' after epoch {epoch}")
    
    # Optionally, save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        model.save_pretrained('best_model')
        tokenizer.save_pretrained('best_model')
        print(f"New best model saved with validation accuracy {best_val_accuracy:.4f}")

# 11. Save the final trained model
model.save_pretrained('saved_model')
tokenizer.save_pretrained('saved_model')
