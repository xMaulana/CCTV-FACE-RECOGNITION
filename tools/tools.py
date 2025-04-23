import time
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import copy
import pickle as pkl
import datetime as dt

def get_ms():
    return int(round(time.time() * 1000))

def sleep_ms(tm:int=0):
    time.sleep(tm/1000)

def str_to_datetime(string:str, format:str="%Y-%m-%d %H:%M:%S"):
    return dt.datetime.strptime(string, format)

def datetime_to_str(dtt:dt.datetime, format:str="%Y-%m-%d %H:%M:%S"):
    return dt.datetime.strftime(dtt, format)

def get_datetime(t_format:str="%Y-%m-%d %H:%M:%S", string:bool=True):
    if string:
        return datetime_to_str(dt.datetime.now(), format=t_format)
    return dt.datetime.now()

def round_all(val):
    if isinstance(val, torch.Tensor):
        return torch.round(val)

    return round(val)

def img2np(img:str):
    img_np = plt.imread(img)
  
    if img.endswith(".png"):
        img_np = np.uint8(img_np * 255)
  
    return img_np

def img2tensor(img:np.ndarray, size=(112, 112), mean=0.5, std=0.5, device="cpu"): 
  img = Image.fromarray(img)

  transform = T.Compose([
      T.Resize(size),  # Resize to 112x112 (ArcFace standard)
      T.ToTensor(),           # Convert to tensor
      T.Normalize(mean=[mean, mean, mean], std=[std, std, std])  # Normalize to [-1, 1]
  ])

    # Apply the transforms
  tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
  return tensor

def display(img:np.ndarray):
  plt.imshow(img)

def save_image(img : np.ndarray, filepath:str):
  # Save img using PIL
  pil_img = Image.fromarray(img)
  pil_img.save(filepath)

def normalize_point(point, box):
  x1, y1, _, _ = map(int, box)
  
  normalized_point = copy.deepcopy(point)
  for i, p in enumerate(point) : 
    normalized_point[i][0] = (p[0] - x1)
    normalized_point[i][1] = (p[1] - y1)
  
  return normalized_point

def load_facebank_json(facebank:str, device:str="cpu") -> dict:
    with open(facebank, "r") as f:
        fbank = json.load(f)

        final_fbank = {
            "features": [],
            "classes": []
        }

        for name, known_embedding in fbank.items():
            for k in known_embedding:
                final_fbank["features"].append(torch.tensor(k[0]))
                final_fbank["classes"].append(name)
        
        del fbank
        final_fbank["features"] = torch.stack(final_fbank["features"], dim=0).to(device)

        return final_fbank
    
def load_facebank(facebank:str, device:str="cpu") -> dict:
    assert facebank.endswith((".json",".pkl"))

    if facebank.endswith(".json"):
        final_fbank = load_facebank_json(facebank, device)
    elif facebank.endswith(".pkl"):
        final_fbank = pkl.load(open(facebank, "rb"))
        final_fbank["features"] = final_fbank["features"].to(device)

    return final_fbank

    
def align_face(face, face_points, reference_points, checker_only:bool=False, im_size= (112,112)):
    # H, W, _ = np.float32(face).shape
    H, W = im_size
    src_points = np.float32(face_points)

    reference_points_fixed = np.float32(reference_points)
    
    M = cv2.estimateAffinePartial2D(src_points, reference_points_fixed, method=cv2.LMEDS)[0]
    
    aligned_face = cv2.warpAffine(face, M, (W,H), flags = cv2.INTER_LINEAR)
    
    if checker_only:
        img = Image.fromarray(aligned_face)
        img.save("aligned_face.jpg")

    return aligned_face

def crop_face(img, box, save_path=None, convert_rgbobgr:bool=True):
    if convert_rgbobgr:
        img = Image.fromarray(img[:,:,::-1])
    else:
        img = Image.fromarray(img)

    x1, y1, x2, y2 = map(int, box)
    img_cropped = img.crop((x1,y1,x2,y2))
   
    if save_path is not None:
        img_cropped.save(save_path)

    img_cropped = np.array(img_cropped)

    return img_cropped

def resize_scale(img, point, dest_size:tuple=(112,112)):
    h, w, _ = img.shape
    scale_x, scale_y = dest_size[0]/w, dest_size[1]/h

    ar_point = np.array(point, dtype=np.float32) * [scale_x, scale_y]

    imnew = cv2.resize(img, dest_size, interpolation=cv2.INTER_AREA)

    return imnew, ar_point



