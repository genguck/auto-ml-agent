from typing import Dict, Any, List
from .base import JinjaTemplate, TemplateRegistry

TEXT_CLASSIFICATION_TEMPLATE = """
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * input_ids.size(0)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            running_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Text Classification Training")
    parser.add_argument("--data_path", type=str, default="{{ data_path }}", help="CSV dataset path")
    parser.add_argument("--model_name", type=str, default="{{ model_name }}", help="HuggingFace model name")
    parser.add_argument("--num_classes", type=int, default={{ num_classes }}, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default={{ batch_size }}, help="Batch size")
    parser.add_argument("--epochs", type=int, default={{ epochs }}, help="Number of epochs")
    parser.add_argument("--lr", type=float, default={{ learning_rate }}, help="Learning rate")
    parser.add_argument("--max_len", type=int, default={{ max_len }}, help="Max sequence length")
    parser.add_argument("--output_dir", type=str, default="{{ output_dir }}", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.data_path)
    texts = df["text"].values
    labels = df["label"].values

    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_classes
    ).to(device)

    full_dataset = TextDataset(texts, labels, tokenizer, args.max_len)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"[Epoch {epoch}/{args.epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print(f"[Epoch {epoch}] Best model saved with val_acc: {best_val_acc:.4f}")

    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.savefig(os.path.join(args.output_dir, "training_curve.png"))
    plt.close()

    results = {
        "model": args.model_name,
        "epochs": args.epochs,
        "best_val_acc": best_val_acc,
        "final_train_acc": train_accs[-1],
        "final_val_acc": val_accs[-1],
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Training complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
"""


@TemplateRegistry.register
class TextClassificationTemplate(JinjaTemplate):
    def __init__(self):
        super().__init__(TEXT_CLASSIFICATION_TEMPLATE)

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model_name": "bert-base-chinese",
            "data_path": "./data.csv",
            "num_classes": 2,
            "batch_size": 16,
            "epochs": 10,
            "learning_rate": 2e-5,
            "max_len": 128,
            "output_dir": "./output",
        }

    def validate_config(self, config: Dict[str, Any]) -> bool:
        required_fields = ["model_name", "data_path", "num_classes", "batch_size", "epochs", "learning_rate"]
        for field in required_fields:
            if field not in config:
                return False

        if not isinstance(config["num_classes"], int) or config["num_classes"] <= 0:
            return False

        if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
            return False

        if not isinstance(config["epochs"], int) or config["epochs"] <= 0:
            return False

        if not isinstance(config["learning_rate"], float) or config["learning_rate"] <= 0:
            return False

        if not isinstance(config.get("max_len", 128), int) or config.get("max_len", 128) <= 0:
            return False

        return True
