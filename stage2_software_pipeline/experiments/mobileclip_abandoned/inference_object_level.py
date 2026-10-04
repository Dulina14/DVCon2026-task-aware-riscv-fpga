"""
Inference script for object-level reranking model.
Given an image, multiple bboxes, and a text prompt,
predict which object matches the text.
"""

import json
import torch
import mobileclip
from PIL import Image
from tqdm import tqdm


def load_model(checkpoint_path, model_name='mobileclip_s2'):
    """Load the trained model."""
    print(f"Loading model from {checkpoint_path}")
    model, _, preprocess = mobileclip.create_model_and_transforms(model_name)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model_state_dict'])
    tokenizer = mobileclip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def predict_object(model, image_path, bbox, text, preprocess, tokenizer, device):
    """Predict if cropped object matches text prompt."""
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    w = max(w, 1)
    h = max(h, 1)
    
    # Load and crop image
    image = Image.open(image_path).convert('RGB')
    image = image.crop((x, y, x + w, y + h))
    image = image.resize((224, 224), Image.LANCOS)
    
    # Preprocess
    image_tensor = preprocess(image).unsqueeze(0)
    text_tokens = tokenizer([text])
    
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarity = model.logit_scale.exp() * (image_features * text_features).sum(dim=-1)
        probability = torch.sigmoid(similarity).item()
    
    return probability


def rerank_objects(model, image_path, objects, task_prompt, preprocess, tokenizer, device):
    """Rank all objects by their match probability to the text."""
    results = []
    
    for obj in objects:
        bbox = obj['bbox_xywh']
        prob = predict_object(model, image_path, bbox, task_prompt, preprocess, tokenizer, device)
        results.append({
            'bbox': bbox,
            'probability': prob,
            'preferred': obj['preferred']
        })
    
    # Sort by probability (highest first)
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    return results


def main():
    device = torch.device('cpu')
    model_path = 'D:/mobileclip/checkpoints/finetuned_object/mobileclip_s2_obj_epoch1.pt'
    
    model, preprocess, tokenizer = load_model(model_path)
    
    # Test on a sample from JSONL
    test_jsonl = 'D:/datasets/derived/coco_tasks_test.jsonl'
    
    print("\n=== Testing on sample ===")
    with open(test_jsonl, 'r') as f:
        entry = json.loads(f.readline())
    
    image_path = entry['image_path']
    task_prompt = entry['task_prompt']
    objects = entry['objects']
    
    print(f"Image: {image_path}")
    print(f"Task: {task_prompt}")
    print(f"Objects: {len(objects)}")
    print()
    
    # Rerank objects
    results = rerank_objects(model, image_path, objects, task_prompt, preprocess, tokenizer, device)
    
    print("=== Reranking Results ===")
    print(f"{'Rank':<6} {'BBox':<30} {'Prob':<10} {'True':<6}")
    print("-" * 55)
    for i, r in enumerate(results):
        bbox_str = str(r['bbox'])
        print(f"{i+1:<6} {bbox_str:<30} {r['probability']:.4f}   {r['preferred']}")
    
    # Check if correct (preferred=1 at rank 1)
    if results[0]['preferred'] == 1:
        print("\n✓ Correct: Preferred object ranked #1")
    else:
        print("\n✗ Wrong: Preferred object not at rank #1")


if __name__ == '__main__':
    main()