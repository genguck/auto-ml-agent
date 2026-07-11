from typing import Dict, Any, List
from .base import JinjaTemplate, TemplateRegistry

IMAGE_CLASSIFICATION_TEMPLATE = """
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


def get_model(model_name: str, num_classes: int) -> nn.Module:
    \"\"\"Get pre-trained model with modified head\"\"\"
    if model_name == "ResNet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "VGG16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "ViT":
        model = models.vit_b_16(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
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
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Image Classification Training")
    parser.add_argument("--data_dir", type=str, default="{{ data_dir }}", help="Dataset directory")
    parser.add_argument("--model", type=str, default="{{ model_name }}", help="Model name")
    parser.add_argument("--num_classes", type=int, default={{ num_classes }}, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default={{ batch_size }}, help="Batch size")
    parser.add_argument("--epochs", type=int, default={{ epochs }}, help="Number of epochs")
    parser.add_argument("--lr", type=float, default={{ learning_rate }}, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="{{ output_dir }}", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform_train)
    
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = get_model(args.model, args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"[Epoch {epoch}/{args.epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print(f"[Epoch {epoch}] Best model saved with val_acc: {best_val_acc:.4f}")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))

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
        "model": args.model,
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
class ImageClassificationTemplate(JinjaTemplate):
    def __init__(self):
        super().__init__(IMAGE_CLASSIFICATION_TEMPLATE)

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model_name": "ResNet50",
            "data_dir": "./data",
            "num_classes": 10,
            "batch_size": 32,
            "epochs": 40,
            "learning_rate": 0.001,
            "output_dir": "./output",
        }

    def validate_config(self, config: Dict[str, Any]) -> bool:
        required_fields = ["model_name", "data_dir", "num_classes", "batch_size", "epochs", "learning_rate"]
        for field in required_fields:
            if field not in config:
                return False

        valid_models = ["ResNet18", "ResNet50", "VGG16", "ViT"]
        if config["model_name"] not in valid_models:
            return False

        if not isinstance(config["num_classes"], int) or config["num_classes"] <= 0:
            return False

        if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
            return False

        if not isinstance(config["epochs"], int) or config["epochs"] <= 0:
            return False

        if not isinstance(config["learning_rate"], float) or config["learning_rate"] <= 0:
            return False

        return True
