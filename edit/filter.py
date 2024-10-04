import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class ImageTextSimilarityScorer:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def compute_similarity_scores(self, images, description):
        inputs = self.processor(
            text=[description] * len(images),
            images=images,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        similarity_scores = outputs.logits_per_image.cpu().numpy().flatten()

        return similarity_scores[::len(images)]

    def process_batch(self, image_paths, description=""):
        images = [Image.open(path).convert("RGB") for path in image_paths]
        return self.compute_similarity_scores(images, description)





















