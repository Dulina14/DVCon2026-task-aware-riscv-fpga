"""
Training script for object-level reranking using bbox and preferred.
Uses cropped objects + text prompts for training.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import mobileclip
from tqdm import tqdm


class CocoTasksDatasetObjectLevel(Dataset):
    """Dataset that creates one sample per object (cropped to bbox)."""
    
    def __init__(self, jsonl_path, preprocess, tokenizer):
        self.data = []
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        
        print("Loading JSONL...")
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Expand to one sample per object
        self.samples = []
        print("Creating object-level samples...")
        for entry in self.data:
            image_path = entry['image_path']
            text = entry['task_prompt']
            
            for obj in entry['objects']:
                bbox = obj['bbox_xywh']
                preferred = obj['preferred']
                
                self.samples.append({
                    'image_path': image_path,
                    'bbox': bbox,
                    'text': text,
                    'preferred': preferred
                })
        
        print(f"Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        bbox = sample['bbox']
        text = sample['text']
        preferred = sample['preferred']
        
        try:
            # Load full image
            image = Image.open(image_path).convert('RGB')
            
            # Crop to bbox
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Handle edge cases
            w = max(w, 1)
            h = max(h, 1)
            
            # Crop: left, top, right, bottom
            image = image.crop((x, y, x + w, y + h))
            
            # Resize cropped region to standard size
            image = image.resize((224, 224), Image.LANCOS)
            
            # Preprocess
            image = self.preprocess(image)
            
        except Exception as e:
            # If crop fails, use zeros
            image = torch.zeros(3, 224, 224)
        
        # Tokenize text
        text_tokens = self.tokenizer([text])
        
        return image, text_tokens[0], preferred


def collate_fn(batch):
    """Collate function for batching."""
    images, texts, labels = zip(*batch)
    images = torch.stack(images)
    texts = torch.stack(texts)
    labels = torch.tensor(labels, dtype=torch.float32)
    return images, texts, labels


def train_model(model, train_loader, optimizer, device, epoch):
    """Training loop with binary classification loss."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, texts, labels) in enumerate(pbar):
        images = images.to(device)
        texts = texts.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Encode image and text
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Combine: image + text features
        combined = image_features * text_features
        
        # Binary classification: does image match text?
        # Use logit_scale as a learned scalar
        similarity = model.logit_scale.exp() * (image_features * text_features).sum(dim=-1)
        
        # Sigmoid for binary classification
        probs = torch.sigmoid(similarity)
        
        # Binary cross entropy loss
        loss = nn.BCELoss()(probs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Track accuracy
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            acc = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{acc:.1f}%'
            })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    device = torch.device('cpu')
    print(f'Device: {device}')
    
    # Load model
    model_name = 'mobileclip_s2'
    pretrained_path = 'D:/mobileclip/checkpoints/mobileclip_s2.pt'
    
    print(f'Loading model: {model_name}')
    model, _, preprocess = mobileclip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_path
    )
    tokenizer = mobileclip.get_tokenizer(model_name)
    model = model.to(device)
    model.train()
    
    # Dataset paths
    train_jsonl = 'D:/datasets/derived/coco_tasks_train.jsonl'
    test_jsonl = 'D:/datasets/derived/coco_tasks_test.jsonl'
    
    # Create datasets (object-level)
    print('Creating training dataset...')
    train_dataset = CocoTasksDatasetObjectLevel(train_jsonl, preprocess, tokenizer)
    
    print('Creating test dataset...')
    test_dataset = CocoTasksDatasetObjectLevel(test_jsonl, preprocess, tokenizer)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    print(f'Batches per epoch: {len(train_loader)}')
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Training
    num_epochs = 3
    save_dir = 'D:/mobileclip/checkpoints/finetuned_object'
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_model(
            model, train_loader, optimizer, device, epoch
        )
        print(f'Epoch {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
        
        # Save checkpoint
        save_path = os.path.join(save_dir, f'mobileclip_s2_obj_epoch{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
        }, save_path)
        print(f'Saved: {save_path}')
    
    print('Training complete!')


if __name__ == '__main__':
    main()