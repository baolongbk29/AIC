import math
import numpy as np
import torch
import glob
from PIL import Image
import clip
import torch

import math
import numpy as np
import torch



image_path_list = glob.glob("/home/finn/AIC/C00_V00/*/**.jpg")
image_path_list.sort()
video_frames = []
for image_path in image_path_list:

    video_frames.append(Image.open(image_path))
# Print some statistics
print(f"Frames extracted: {len(video_frames)}")

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)


import math
import numpy as np
import torch

# You can try tuning the batch size for very large videos, but it should usually be OK
batch_size = 256
batches = math.ceil(len(video_frames) / batch_size)

# The encoded features will bs stored in video_features
video_features = torch.empty([0, 768], dtype=torch.float16).to(device)

# Process each batch
for i in range(batches):
  
  print(f"Processing batch {i+1}/{batches}")

  # Get the relevant frames
  batch_frames = video_frames[i*batch_size : (i+1)*batch_size]
  
  # Preprocess the images for the batch
  batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)
  
  # Encode with CLIP and normalize
  with torch.no_grad():
    batch_features = model.encode_image(batch_preprocessed)
    batch_features /= batch_features.norm(dim=-1, keepdim=True)

  # Append the batch to the list containing all features
  video_features = torch.cat((video_features, batch_features))

# Print some stats
print(f"Features: {video_features.shape}")

np.save("/home/finn/AIC/C00_ViT-L/14_feature.npy",np.array(video_features.cpu()))