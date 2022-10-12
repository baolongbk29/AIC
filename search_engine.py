
from concurrent.futures import process
from genericpath import exists
from random import sample
from utils import load_feature, load_image
from models import load_clip_model, load_blip_itm, inference_transnet
import torch
import clip
import glob
import cv2
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image


class SEARCH_ENGINE():
    """
    Three type: 
        Text
        Video
        Multimodal
    """
    def __init__(self, model_name ="ViT-B/32", type=None):

        self.model_clip, self.preprocess_clip = load_clip_model(model_name = model_name, device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'))
        self.model_blip = load_blip_itm(torch.device("cuda" if torch.cuda.is_available() else 'cpu'))

        self.database_features = load_feature(r"/home/finn/AIC/C00_feature.npy", torch.device("cuda" if torch.cuda.is_available() else 'cpu'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def search_text_query(self, search_query, display_heatmap=True, display_results_count=10):

        with torch.no_grad():
            text_features = self.model_clip.encode_text(clip.tokenize(search_query).to(torch.device("cuda" if torch.cuda.is_available() else 'cpu')))
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute the similarity between the search query and each frame using the Cosine similarity
        similarities = (100.0 * self.database_features @ text_features.T)
        values, best_photo_idx = similarities.topk(display_results_count, dim=0)


        torch.cuda.empty_cache()

        image_path_list = glob.glob(r"/home/finn/AIC/C00_V00/*/**.jpg")
        image_path_list.sort()

        dir = "results"
        os.makedirs(dir, exist_ok=True)
        filelist = glob.glob(os.path.join(dir, "*"))
        for f in filelist:
            os.remove(f)

        for i in range(9):
            image_path = image_path_list[best_photo_idx[i][0]]
            image = cv2.imread(image_path)
            image_for_blip= load_image(image, torch.device("cuda" if torch.cuda.is_available() else 'cpu'))
            sample = {"image": image_for_blip, "text_input":search_query}

            itm_output = self.model_blip(sample,match_head='itm')
            itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
            

            name = os.path.split(image_path)[-1]
            cv2.imwrite(os.path.join(dir, name), image)
            print(f'Probability {name} : %.4f '%itm_score)

    def search_video_query(self, search_video_path, display_heatmap=True, display_results_count=9):
        
        print("==========Extract frame===========")
        # The frame images will be stored in video_frames
        video_frames = []
        # Open the video file
        capture = cv2.VideoCapture(search_video_path)
        current_frame = 0
        while capture.isOpened():
            # Read the current frame
            ret, frame = capture.read()
            #frame = cv2.resize(frame, (48, 27))
            # Convert it to a PIL image (required for CLIP) and store it
            if ret == True:
                video_frames.append(Image.fromarray(frame[:, :, ::-1]))
            else:
                break

            # Skip N frames
            current_frame += 25
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        # Print some statistics
        print(f"Frames extracted: {len(video_frames)}")

        #Key frame extract with TransNet-V2
        print("==========Extract Key frame===========")
        key_frames = inference_transnet(video_frames)

        frame_preprocessed  = torch.stack([self.preprocess_clip(frame) for frame in key_frames]).to(self.device)


        # Encode and normalize the search query using CLIP
        with torch.no_grad():
            key_frame_features = self.model_clip.encode_image(frame_preprocessed)
            key_frame_features /= key_frame_features.norm(dim=-1, keepdim=True)

        # Compute the similarity between the search query and each frame using the Cosine similarity
        similarities = (100.0 * self.database_features @ key_frame_features.T)
        values, best_photo_idx = similarities.topk(display_results_count, dim=0)
        image_path_list = glob.glob("/home/finn/AIC/C00_V00/*/**.jpg")
        image_path_list.sort()

        dir = "results"
        os.makedirs(dir, exist_ok=True)
        filelist = glob.glob(os.path.join(dir, "*"))
        for f in filelist:
            os.remove(f)

        for i in range(9):
            image_path = image_path_list[best_photo_idx[i][0]]
            image = cv2.imread(image_path)
            name = os.path.split(image_path)[-1]
            cv2.imwrite(os.path.join(dir, name), image)
                
     

if __name__=="__main__":
    serch_engine = SEARCH_ENGINE()

    print(f"============================")
    print(f"===WELCOME SEARCH ENGINE====")
    print(f"============================")
    print('Type query [text of video]:')
    type = input()
    if str(type)=="text":
        print("Enter your text query:")
        text = input()
        serch_engine.search_text_query(str(text))

    elif str(type)=="video":
        print("Enter your video query path:")
        video = input()
        serch_engine.search_video_query(str(video))


