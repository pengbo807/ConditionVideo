import decord
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange
from controlnet_aux import OpenposeDetector
import torch
from torchvision import transforms
import os
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import imageio
import numpy as np
import cv2
from PIL import Image
palette = np.asarray([
    [0, 0, 0],
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
])


def get_canny(image):
    '''
    input: torch.Tensor
    return: Image
    '''
    image = np.array(image)

    low_threshold = 30
    high_threshold = 60

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

class ConditionVideoDataset(Dataset):
    '''
    Dataset for Condition Video
    
    Args:
        width: width of video frames
        height: height of video frames
        n_sample_frames: number of frames to sample from video
        sample_start_idx: start index of sampling frames
        sample_frame_rate: sample frame rate
        cond_type: type of conditioning image
    '''
    def __init__(
            self,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
            cond_type: str = None,
    ):
        assert cond_type is not None
        self.cond_type = cond_type

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        # set cond extractor
        if cond_type == "openpose":
            self.cond_extractor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet") # TODO:change all pose extractor
        elif cond_type == "canny":
            self.cond_extractor = get_canny
        elif cond_type == "segmentation":
            from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
            self.image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
            self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
            self.cond_extractor = self.get_segmentation
        elif cond_type == "depth":
            from transformers import pipeline
            self.depth_estimator = pipeline('depth-estimation')
            self.cond_extractor = self.get_depth
        else:
            raise NotImplementedError

        self.conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize((width,height), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]) # FIXME:        

    def get_segmentation(self, image):
        image = np.array(image)
        image = Image.fromarray(image).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)
        seg = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(512,512)])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        image = Image.fromarray(color_seg)
        return image
    
    def get_depth(self, image):
        image = np.array(image)
        image = Image.fromarray(image).convert("RGB")
        image = self.depth_estimator(image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image
    
    def get_video(self, video_dir):
        vr = decord.VideoReader(video_dir, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")
        return (video/127.5 -1.0)
    
    def get_video_and_save(self, video_dir, save_path, fps=8):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vr = decord.VideoReader(video_dir, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        imageio.mimsave(save_path, video, fps=fps)
        video = rearrange(video, "f h w c -> f c h w")
        return (video/127.5 -1.0)

    def get_cond(self, video):
        conds = []
        print("get cond")
        for i in tqdm(range(len(video))):
            img = video[i]
            cond = self.cond_extractor(img)
            cond = self.conditioning_image_transforms(cond)
            conds.append(cond)
        conds = torch.stack(conds)
        return conds

    def load_conds(self, whole_conds_path, whole_conds_video_path):
        if whole_conds_path is not None and os.path.exists(whole_conds_path):
            self.whole_conds = torch.load(whole_conds_path)
            return self.whole_conds
        else:
            assert whole_conds_video_path is not None and os.path.exists(whole_conds_video_path)
            print(f"loading from {whole_conds_video_path} and processing pose and videos")
            conds = []
            vr = decord.VideoReader(whole_conds_video_path, width=self.width, height=self.height)
            sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))
            for i in tqdm(range(0,len(sample_index) - self.n_sample_frames+1,self.n_sample_frames)):
                video = vr.get_batch(sample_index[i:i+self.n_sample_frames])
                cond = self.get_cond(video)
                conds.append(cond)
            conds = torch.stack(conds)
            based_dir = os.path.dirname(whole_conds_video_path)
            whole_conds_path = os.path.join(based_dir, f"whole_{self.cond_type}.pt")
            torch.save(conds, whole_conds_path)
            return conds





    