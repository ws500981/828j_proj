import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
import os
import glob

def prepare_dataset(df, bert_name):
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    tokenizer = DistilBertTokenizer.from_pretrained(bert_name)
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
    return train_encodings, train_labels, val_encodings, val_labels

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

class DistilBertModel(object):
    def __init__(self,train_loader, val_loader, bert_name):
        super(DistilBertModel, self).__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = DistilBertForSequenceClassification.from_pretrained(bert_name, num_labels=2)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.epochs = 3
        self.scheduler = get_linear_schedule_with_warmup(
                        self.optimizer,
                        num_warmup_steps=0,
                        num_training_steps=len(self.train_loader) * self.epochs)
        

    def train(self):
        # Define paths for saving checkpoints
        checkpoint_dir = f"./bert_checkpoints_real"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training
        best_val_accuracy = 0.0 
        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                del input_ids
                del attention_mask
                del labels
                loss = outputs.loss
                total_train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                torch.cuda.empty_cache()
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            print(f"Epoch {epoch} Training Loss: {avg_train_loss}")
            
            # Validation
            self.model.eval()
            val_labels_list = []
            val_preds_list = []
            total_val_loss = 0
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self.model(
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
            avg_val_loss = total_val_loss / len(self.val_loader)
            val_accuracy = accuracy_score(val_labels_list, val_preds_list)
            print(f"Epoch {epoch} Validation Loss: {avg_val_loss}")
            print(f"Epoch {epoch} Validation Accuracy: {val_accuracy}")

            # Save checkpoint at the end of each epoch
            checkpoint_path = os.path.join(checkpoint_dir, f"distil_bert_checkpoint_epoch{epoch}.pth")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at '{checkpoint_path}' after epoch {epoch}")
        
        self.model.save_pretrained(f'saved_model_bert_real')

if __name__ == "__main__":
    # choose the dataset name
    dataset_name = "real_lort.csv"
    df = pd.read_csv(os.path.join("./../data/fin_data/",dataset_name), dtype=object)
    df["label"] = df["label"].astype(int)
    bert_name = 'distilbert/distilbert-base-uncased'

    train_encodings, train_labels, val_encodings, val_labels = prepare_dataset(df, bert_name)
    
    train_dataset = BinaryClassificationDataset(train_encodings, train_labels)
    val_dataset = BinaryClassificationDataset(val_encodings, val_labels)

    class_weights = [1/df[df["label"]==0].shape[0], 1/df[df["label"]==1].shape[0]]
    sample_weights = [0]*len(train_dataset)
    # print(len(train_dataset))
    for idx, item in enumerate(train_dataset):
        sample_weights[idx] = class_weights[item["labels"]]

    sampler = WeightedRandomSampler(sample_weights, num_samples = len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    obj = DistilBertModel(train_loader, val_loader, bert_name)
    obj.train()