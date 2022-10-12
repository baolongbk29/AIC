import clip
import torch
from lavis.models import load_model
from utils import convert_image_to_rgb
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import numpy as np
from TransNetV2.inference_pytorch.transnetv2_pytorch import TransNetV2


BICUBIC = Image.BICUBIC


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip_model(model_name ,device):
    # Load the open CLIP model
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess

def load_blip_itm(device):
    model_blip= load_model("blip_image_text_matching", "large")
    model_blip .eval()
    model_blip  = model_blip .to(device)
    return model_blip

def inference_transnet(video_frames):
    key_frame = []
    model = TransNetV2()
    state_dict = torch.load("/home/finn/AIC/TransNetV2/transnetv2-pytorch-weights.pth")
    model.load_state_dict(state_dict)
    model.eval().cuda()


    transform = Compose([
        Resize((27, 48), interpolation=BICUBIC),
        convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
    frame_preprocessed = torch.stack([transform(frame) for frame in video_frames]).to(device)
    frame_preprocessed= frame_preprocessed.permute(0, 2 ,3 ,1, )
    frame_preprocessed = torch.unsqueeze(frame_preprocessed, dim=0).to(torch.uint8)
    frame_preprocessed.size()
    
    with torch.no_grad():
        # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels
        single_frame_pred, all_frame_pred = model(frame_preprocessed.cuda())
        
        single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
        all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()

        key_frame_id = np.argsort(all_frame_pred[0], axis=0)[::-1][:5]

        for id in key_frame_id:

            key_frame.append(video_frames[id[0]])

    return key_frame


if __name__=="__main__":
    model_blip = load_blip_itm(device)