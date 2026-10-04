import torch
import mobileclip
from PIL import Image
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python inference.py <image_path> <text_prompt>")
        print("Example: python inference.py test.jpg 'a dog'")
        sys.exit(1)
    
    image_path = sys.argv[1]
    text_prompt = sys.argv[2]
    
    checkpoint_path = 'D:/mobileclip/checkpoints/finetuned/mobileclip_s2_epoch1.pt'
    
    print(f"Loading model from {checkpoint_path}...")
    model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s2')
    model.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model_state_dict'])
    tokenizer = mobileclip.get_tokenizer('mobileclip_s2')
    model.eval()
    
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    
    print(f"Text prompt: '{text_prompt}'")
    text = tokenizer([text_prompt])
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T).item()
        probability = torch.sigmoid(torch.tensor(similarity) * model.logit_scale.exp()).item()
    
    print(f"\n===== Result =====")
    print(f"Similarity: {similarity:.4f}")
    print(f"Probability: {probability:.2%}")

if __name__ == '__main__':
    main()
