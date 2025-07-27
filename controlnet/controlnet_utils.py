import cv2
import numpy as np
import torch
from diffusers.utils import load_image
from PIL import Image
from transformers import pipeline


def generate_canny_image(image_pil, low_threshold=100, high_threshold=200):
    image_np = np.array(image_pil)

    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    canny = cv2.Canny(gray, low_threshold, high_threshold)

    canny_pil = Image.fromarray(canny).convert("RGB")

    return canny_pil


def generate_depth_image(image_pil):
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")

    depth = depth_estimator(image_pil)
    depth_image = depth["depth"]

    depth_np = np.array(depth_image)
    depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
    depth_np = (depth_np * 255).astype(np.uint8)

    depth_pil = Image.fromarray(depth_np).convert("RGB")

    return depth_pil


def generate_control_images_with_controlnet_aux(image_pil):
    from controlnet_aux import CannyDetector, MidasDetector, OpenposeDetector

    canny_processor = CannyDetector()
    depth_processor = MidasDetector.from_pretrained("lllyasviel/Annotators")

    canny_image = canny_processor(image_pil)
    depth_image = depth_processor(image_pil)
    openpose_image = openpose_processor(image_pil)
