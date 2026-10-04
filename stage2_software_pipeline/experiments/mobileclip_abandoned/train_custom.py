import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import mobileclip
from tqdm import tqdm

class CocoTasksDataset(Dataset):
    def __init__(self, jsonl_path, preprocess, tokenizer):
        self.data = []
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        text = item['task_prompt']
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        text_tokens = self.tokenizer([text])
        
        return image, text_tokens[0]

def collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images)
    texts = torch.stack(texts)
    return images, texts

def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, texts) in enumerate(pbar):
        images = images.to(device)
        texts = texts.to(device)
        
        optimizer.zero_grad()
        
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits_per_image = model.logit_scale.exp() * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        batch_size = images.shape[0]
        labels = torch.arange(batch_size, device=device)
        
        loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total_loss/(batch_idx+1):.4f}'})
        
        if batch_idx >= 50:
            break
    
    return total_loss / min(51, len(train_loader))

def main():
    device = torch.device('cpu')
    print('Using device:', device)
    
    model_name = 'mobileclip_s2'
    pretrained_path = 'D:/mobileclip/checkpoints/mobileclip_s2.pt'
    
    print('Loading model...')
    model, _, preprocess = mobileclip.create_model_and_transforms(
        model_name, 
        pretrained=pretrained_path
    )
    tokenizer = mobileclip.get_tokenizer(model_name)
    model = model.to(device)
    model.train()
    
    train_jsonl = 'D:/datasets/derived/coco_tasks_train.jsonl'
    
    print('Loading dataset...')
    train_dataset = CocoTasksDataset(train_jsonl, preprocess, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Total batches (capped at 50): 51')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    num_epochs = 1
    save_dir = 'D:/mobileclip/checkpoints/finetuned'
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, epoch)
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}')
        
        save_path = os.path.join(save_dir, f'mobileclip_s2_epoch{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
        }, save_path)
        print(f'Model saved to {save_path}')
    
    print('Training complete!')

if __name__ == '__main__':
    main()
