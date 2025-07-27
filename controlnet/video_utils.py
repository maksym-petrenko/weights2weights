import os

import mediapipe as mp
import torch
from controlnet_utils import generate_canny_image, generate_depth_image

from facemesh import generate_controlnet_face_pose, get_canonical_3d_landmarks

mp_face_mesh = mp.solutions.face_mesh


def get_facemesh(imgs: list, dir: str | None = None):
    result = []

    if dir is not None:
        os.makedirs(dir, exist_ok=True)

    for i, image in enumerate(imgs):
        canonical_landmarks = get_canonical_3d_landmarks(image)
        output_path = f"{dir}/{i:04d}.png" if dir is not None else None

        pitch = 30
        yaw = 0
        roll = 0

        generated_image = generate_controlnet_face_pose(
            canonical_landmarks_3d=canonical_landmarks,
            pitch_deg=pitch,
            yaw_deg=yaw,
            roll_deg=roll,
            output_size=(512, 512),
            output_path=output_path,
            connections_to_draw=mp_face_mesh.FACEMESH_CONTOURS,
        )
        result.append(generated_image)

    return result


def generate_sequence(**kwargs):
    """
    Requires:
        - images (list): the initial sequence
        - steps (int): the number of sampler steps
        - start (int): the starting step (the bigger the less denoise is applied)
        - network
        - unet
        - vae
        - text_encoder
        - tokenizer
        - noise_scheduler
        - prompt
        - negative_prompt
        - guidance_scale
        - device
        - controlnet_openpose
        - controlnet_canny
        - controlnet_depth
        - gif_duration
        - output_gif_path
    """

    # load facemesh into tensors
    # load the latents
    # make clip preparation
    for i in range(steps):
        ...
        # load all canny and depth for now
        # denoise each image one step
