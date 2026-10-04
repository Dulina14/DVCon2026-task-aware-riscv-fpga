import os
import json
import torch
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
            image = torch.zeros(3, 224, 224)
        
        text_tokens = self.tokenizer([text])
        
        return image, text_tokens[0], text

def collate_fn(batch):
    images, texts, raw_texts = zip(*batch)
    images = torch.stack(images)
    texts = torch.stack(texts)
    return images, texts, raw_texts

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    pbar = tqdm(test_loader, desc='Evaluating')
    with torch.no_grad():
        for images, texts, raw_texts in pbar:
            images = images.to(device)
            texts = texts.to(device)
            
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            logits_per_image = model.logit_scale.exp() * image_features @ text_features.T
            
            predictions = logits_per_image.argmax(dim=1)
            
            for i, pred in enumerate(predictions):
                if pred == i:
                    correct += 1
                total += 1
            
            pbar.set_postfix({'acc': f'{100*correct/total:.2f}%'})
    
    return 100 * correct / total

def main():
    device = torch.device('cpu')
    print('Using device:', device)
    
    model_name = 'mobileclip_s2'
    checkpoint_path = 'D:/mobileclip/checkpoints/finetuned/mobileclip_s2_epoch1.pt'
    
    print('Loading model...')
    model, _, preprocess = mobileclip.create_model_and_transforms(model_name)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model_state_dict'])
    tokenizer = mobileclip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    
    test_jsonl = 'D:/datasets/derived/coco_tasks_test.jsonl'
    
    print('Loading test dataset...')
    test_dataset = CocoTasksDataset(test_jsonl, preprocess, tokenizer)
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f'Test samples: {len(test_dataset)}')
    print(f'Running inference...')
    
    accuracy = evaluate(model, test_loader, device)
    print(f'\n===== Results =====')
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Correct: {int(accuracy * len(test_dataset) / 100)}/{len(test_dataset)}')

if __name__ == '__main__':
    main()
