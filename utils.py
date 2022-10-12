from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import torch
BICUBIC = Image.BICUBIC
import torch
import numpy as np

def load_feature(feature_path, device):
    video_features = torch.empty([0, 512], dtype=torch.float16).to(device)
    features = np.load(feature_path)
    features = torch.tensor(features).to(device)
    #features /= features.norm(dim=-1, keepdim=True)
    video_features = torch.cat((video_features, features))
    return video_features


def load_image(image,device):
    raw_image = Image.fromarray(image).convert('RGB')       
    transform = transforms.Compose([
        transforms.Resize((384,384),interpolation=BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image


def convert_image_to_rgb(image):
    return image.convert("RGB")


