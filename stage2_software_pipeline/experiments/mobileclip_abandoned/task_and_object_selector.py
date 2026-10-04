"""
Use MobileCLIP to identify task from user text prompt.
Then filter objects by relevant categories and rank with MobileCLIP.
"""

import torch
import mobileclip
from PIL import Image
import json

# The 14 task prompts from the paper
TASK_PROMPTS = {
    1: "step on something to reach top of a shelf",
    2: "sit comfortably",
    3: "place flowers",
    4: "get potatoes out of fire",
    5: "water plant",
    6: "get lemon out of tea",
    7: "dig hole",
    8: "open bottle of beer",
    9: "open parcel",
    10: "serve wine",
    11: "pour sugar",
    12: "smear butter",
    13: "extinguish fire",
    14: "pound carpet",
}

# Task -> Relevant COCO categories (from the paper)
TASK_CATEGORIES = {
    1: [57, 60, 61, 65, 56],  # bench, chair, etc
    2: [18, 57, 60, 56, 65],  # bed, couch, chair
    3: [60, 86, 87],  # vase
    4: [53, 55],  # pot, bowl
    5: [60, 86, 58, 77],  # vase, pot, cup
    6: [40, 50],  # lemon
    7: [43, 45],  # shovel, hoe
    8: [40, 44, 47],  # bottle, wine glass, cup
    9: [41, 42, 43],  # box, scissors, knife
    10: [44, 47, 42, 40],  # wine glass, cup, bowl
    11: [42, 43, 47],  # bowl, knife, cup
    12: [45, 50],  # knife, lemon
    13: [49, 51],  # fire extinguisher
    14: [45, 50],  # hammer
}

def load_model():
    """Load MobileCLIP-S2 model."""
    model, _, preprocess = mobileclip.create_model_and_transforms(
        'mobileclip_s2',
        pretrained='D:/mobileclip/checkpoints/mobileclip_s2.pt'
    )
    tokenizer = mobileclip.get_tokenizer('mobileclip_s2')
    return model, preprocess, tokenizer

def identify_task(model, tokenizer, user_text):
    """Identify which task the user text refers to."""
    # Encode user text
    user_tokens = tokenizer([user_text])
    
    # Encode all task prompts
    task_texts = list(TASK_PROMPTS.values())
    task_tokens = tokenizer(task_texts)
    
    with torch.no_grad():
        user_features = model.encode_text(user_tokens)
        task_features = model.encode_text(task_tokens)
        
        user_features /= user_features.norm(dim=-1, keepdim=True)
        task_features /= task_features.norm(dim=-1, keepdim=True)
        
        similarities = (100.0 * user_features @ task_features.T).softmax(dim=-1)
    
    # Get best task
    task_id = similarities.argmax(dim=1).item() + 1
    confidence = similarities[0, task_id - 1].item()
    
    return task_id, TASK_PROMPTS[task_id], confidence

def filter_objects_by_category(objects, task_id):
    """Filter objects to only relevant categories."""
    relevant_cats = TASK_CATEGORIES.get(task_id, [])
    filtered = [obj for obj in objects if obj['coco_category_id'] in relevant_cats]
    return filtered

def rank_objects(model, tokenizer, image_path, objects, task_prompt):
    """Rank objects by their suitability for the task."""
    _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s2')
    
    task_tokens = tokenizer([task_prompt])
    task_features = model.encode_text(task_tokens)
    task_features /= task_features.norm(dim=-1, keepdim=True)
    
    results = []
    
    for obj in objects:
        bbox = obj['bbox_xywh']
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        try:
            # Load and crop image
            image = Image.open(image_path).convert('RGB')
            image = image.crop((x, y, x + w, y + h))
            image = image.resize((224, 224), Image.LANCZOS)
            image = preprocess(image).unsqueeze(0)
            
            # Encode
            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * img_features @ task_features.T).item()
            
            results.append({
                'bbox': bbox,
                'coco_category_id': obj['coco_category_id'],
                'score': similarity,
                'preferred': obj.get('preferred', None)
            })
        except:
            continue
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def main():
    device = torch.device('cpu')
    
    print("Loading MobileCLIP model...")
    model, preprocess, tokenizer = load_model()
    model.eval()
    
    # Example: user asks about opening a package
    user_text = "where can I put this vace"
    
    task_id, task_prompt, confidence = identify_task(model, tokenizer, user_text)
    print(f"\nUser text: '{user_text}'")
    print(f"Identified task: {task_id} - '{task_prompt}' (confidence: {confidence:.2f})")
    
    # Load image and objects
    # In practice, you'd use object detection first
    # Then filter and rank

if __name__ == '__main__':
    main()