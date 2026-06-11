import requests
import torch
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForImageClassification


image_processor = AutoImageProcessor.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",
)
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",
    device_map="auto"
)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt").to(model.device)

with torch.no_grad():
  logits = model(**inputs).logits

predicted_class_id = logits.argmax(dim=-1).item()
predicted_class_label = model.config.id2label[predicted_class_id]
print(f"The predicted class label is: {predicted_class_label}")
