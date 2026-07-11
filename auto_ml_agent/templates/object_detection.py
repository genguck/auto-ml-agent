from typing import Dict, Any, List
from .base import JinjaTemplate, TemplateRegistry

OBJECT_DETECTION_TEMPLATE = """
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)
        
        self.image_info = []
        self.categories = {}
        for cat in self.annotations["categories"]:
            self.categories[cat["id"]] = cat["name"]
        
        for img in self.annotations["images"]:
            image_id = img["id"]
            file_name = img["file_name"]
            height = img["height"]
            width = img["width"]
            
            boxes = []
            labels = []
            for ann in self.annotations["annotations"]:
                if ann["image_id"] == image_id:
                    bbox = ann["bbox"]
                    boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    labels.append(ann["category_id"])
            
            self.image_info.append({
                "file_name": file_name,
                "height": height,
                "width": width,
                "boxes": boxes,
                "labels": labels,
            })

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        info = self.image_info[idx]
        image_path = os.path.join(self.root_dir, info["file_name"])
        image = plt.imread(image_path)
        
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image)

        boxes = torch.tensor(info["boxes"], dtype=torch.float32)
        labels = torch.tensor(info["labels"], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


def get_model(num_classes):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

        if num_batches % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"[Epoch {epoch}] Batch {num_batches} - Avg Loss: {avg_loss:.4f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def compute_mAP(model, dataloader, device, num_classes):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                preds = {
                    "boxes": output["boxes"].cpu().numpy(),
                    "scores": output["scores"].cpu().numpy(),
                    "labels": output["labels"].cpu().numpy(),
                }
                tgt = {
                    "boxes": targets[i]["boxes"].cpu().numpy(),
                    "labels": targets[i]["labels"].cpu().numpy(),
                }
                all_predictions.append(preds)
                all_targets.append(tgt)

    iou_thresholds = np.linspace(0.5, 0.95, 10)
    ap_per_class = defaultdict(list)

    for iou_thresh in iou_thresholds:
        for cls in range(1, num_classes):
            tp = 0
            fp = 0
            fn = 0
            
            for preds, tgt in zip(all_predictions, all_targets):
                cls_preds = preds["boxes"][preds["labels"] == cls]
                cls_tgts = tgt["boxes"][tgt["labels"] == cls]
                
                matched = set()
                for pred in cls_preds:
                    best_iou = 0
                    best_idx = -1
                    for j, tgt_box in enumerate(cls_tgts):
                        if j in matched:
                            continue
                        iou = compute_iou(pred, tgt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = j
                    
                    if best_iou >= iou_thresh:
                        tp += 1
                        matched.add(best_idx)
                    else:
                        fp += 1
                
                fn += len(cls_tgts) - len(matched)
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0
            ap_per_class[cls].append(precision)

    mAP_50_95 = np.mean([np.mean(aps) for aps in ap_per_class.values()]) if ap_per_class else 0.0
    mAP_50 = np.mean([aps[0] for aps in ap_per_class.values()]) if ap_per_class else 0.0

    return mAP_50_95, mAP_50


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Object Detection Training")
    parser.add_argument("--data_dir", type=str, default="{{ data_dir }}", help="Dataset directory")
    parser.add_argument("--train_annot", type=str, default="{{ train_annot }}", help="Train annotations JSON")
    parser.add_argument("--val_annot", type=str, default="{{ val_annot }}", help="Val annotations JSON")
    parser.add_argument("--num_classes", type=int, default={{ num_classes }}, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default={{ batch_size }}, help="Batch size")
    parser.add_argument("--epochs", type=int, default={{ epochs }}, help="Number of epochs")
    parser.add_argument("--lr", type=float, default={{ learning_rate }}, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="{{ output_dir }}", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = CocoDataset(args.data_dir, args.train_annot)
    val_dataset = CocoDataset(args.data_dir, args.val_annot)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, collate_fn=collate_fn
    )

    model = get_model(args.num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_mAP = 0.0
    train_losses = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()
        train_losses.append(train_loss)

        print(f"[Epoch {epoch}/{args.epochs}] Train Loss: {train_loss:.4f}")

        mAP_50_95, mAP_50 = compute_mAP(model, val_loader, device, args.num_classes)
        print(f"[Epoch {epoch}] mAP@0.5: {mAP_50:.4f}, mAP@0.5:0.95: {mAP_50_95:.4f}")

        if mAP_50_95 > best_mAP:
            best_mAP = mAP_50_95
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print(f"[Epoch {epoch}] Best model saved with mAP: {best_mAP:.4f}")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "training_curve.png"))
    plt.close()

    results = {
        "model": "FasterRCNN_ResNet50_FPN",
        "epochs": args.epochs,
        "best_mAP_50_95": best_mAP,
        "train_losses": train_losses,
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Training complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
"""


@TemplateRegistry.register
class ObjectDetectionTemplate(JinjaTemplate):
    def __init__(self):
        super().__init__(OBJECT_DETECTION_TEMPLATE)

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "data_dir": "./data",
            "train_annot": "./annotations/train.json",
            "val_annot": "./annotations/val.json",
            "num_classes": 91,
            "batch_size": 8,
            "epochs": 20,
            "learning_rate": 0.005,
            "output_dir": "./output",
        }

    def validate_config(self, config: Dict[str, Any]) -> bool:
        required_fields = ["data_dir", "train_annot", "val_annot", "num_classes", "batch_size", "epochs", "learning_rate"]
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

        return True
