import json
import numpy as np
import cv2
import torch
from torchvision import transforms, models
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
import json
import os

from copy import deepcopy
from torchvision.transforms import Resize, Compose, GaussianBlur, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation

from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torchviz import make_dot

import torch.onnx

import argparse

import cv2

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from torchvision.transforms import functional as F

import io
import PIL
from torchvision.transforms import ToTensor

from matplotlib import pyplot as plt

DEBUG = False

def plot(epoch, ind, keypoints_same, ground_img, sat_img, path=None):
    # keypoints = x1_g1, y1_g1, x1_s1, y1_s1, x2_g1, y2_g1, x2_s1, y2_s1,
    ground_img = ground_img.detach().cpu()
    sat_img = sat_img.detach().cpu()
    keypoints_same = keypoints_same.detach().cpu() * NEW_IMAGE_SIZE # scale keypoints back up to original image size

    ground_img = ground_img.numpy()
    sat_img = sat_img.numpy()
    keypoints_same = keypoints_same.numpy()

    # reshape images to (H, W, C)
    ground_img = np.transpose(ground_img, (1, 2, 0))
    sat_img = np.transpose(sat_img, (1, 2, 0))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(ground_img)
    ax[1].imshow(sat_img)

    for i in range(0, len(keypoints_same), 4):
        x1_g1, y1_g1, x1_s1, y1_s1 = keypoints_same[i:i+4]
        ax[0].scatter(x1_g1, y1_g1, color="red")
        ax[1].scatter(x1_s1, y1_s1, color="red")
    if path is None:
        os.makedirs("test_images_outputs", exist_ok=True)
        os.makedirs(f"test_images_outputs/{epoch}", exist_ok=True)
        plt.savefig(f"test_images_outputs/{epoch}/test_{ind}.png")
        plt.close()
    else:
        plt.savefig(path)
        plt.close()

def scale_keypoints(keypoints, scale_factor):
    if DEBUG:
        print("(scale_keypoints)", f"input {keypoints=}, {scale_factor=}")
    
    if np.all(np.array(keypoints) < 1.0) and np.all(np.array(keypoints) >= 0.0):
        if DEBUG: print("keypoints are already in normalized to [0, 1]!")
        return keypoints

    ori = scale_factor * 5000
    a = np.array(keypoints) * scale_factor
    b =  np.array(a) / ori
    # check that the output is [0, 1]
    if not np.all(b <= 1.0):
        if DEBUG: print(f"b: {b}, b.shape: {b.shape}")
        raise Exception("b is not in [0, 1]")
    if DEBUG:
        print("(scale_keypoints)", f"output {b=}")
    return b

def custom_mse_loss(output, target):
    mask = (target != 0)
    masked_output = output * mask
    mse = torch.mean((masked_output - target) ** 2)
    return mse

# def find_closest_keypoint(pred_keypoint, )

def create_heatmap(size, center, sigma):
    """Create a heatmap with a single peak."""
    if DEBUG:
        print(f"(create_heatmap)", f"{size=}, {center=}, {sigma=}")
    x = torch.arange(0, size[1]).repeat(size[0], 1).to(center.device)
    y = torch.arange(0, size[0]).view(-1, 1).repeat(1, size[1]).to(center.device)
    x = x - center[1]
    y = y - center[0]
    heatmap = torch.exp(-0.5 * (x**2 + y**2) / sigma**2)
    return heatmap.to(center.device)


def keypoints_to_heatmaps(keypoints, heatmap_size, sigma_ground, sigma_satellite, writer=None, mode="multiple"):
    """Convert keypoints to heatmaps."""
    if DEBUG:
        print("(keypoints_to_heatmaps)", f"{keypoints.shape=}, {heatmap_size=}, {sigma_ground=}, {sigma_satellite=}")
        print("(keypoints_to_heatmaps)", f"input {keypoints=}")

    if mode == "multiple":
        batch_size, n_values = keypoints.shape
        n_keypoints = n_values // 2  # Each keypoint has x and y coordinates

        heatmap_size_tensor = torch.tensor(heatmap_size, device=keypoints.device)
        heatmaps = torch.zeros(batch_size, n_keypoints, *heatmap_size, device=keypoints.device)

        for i in range(batch_size):
            for j in range(0, n_values, 2):  # Loop over keypoints (each keypoint has two values)
                center = (keypoints[i, j:j+2] * heatmap_size_tensor).long()  # Get x and y coordinates for the keypoint
                sigma = sigma_ground if (j // 2) % 2 == 0 else sigma_satellite
                heatmaps[i, j // 2] = create_heatmap(heatmap_size, center, sigma)
                if DEBUG:
                    print(f"(keypoints_to_heatmaps)", f"keypoints[{i}, {j//2}] = {keypoints[i, j//2].shape} is_ground: {(j // 2) % 2 == 0}")

        if DEBUG and writer is not None:
            print("(keypoints_to_heatmaps)", f"output {heatmaps.shape=}")
            # log to writer as heatmaps (image)
            for i in range(batch_size):
                for j in range(n_keypoints):
                    norm = heatmaps[i, j].clone()
                    norm -= norm.min()
                    norm /= norm.max()
                    writer.add_image(f"heatmap_{i}_{j}", norm, global_step=0, dataformats='HW')
    
    if mode == "single":
        # make a single heatmap
        batch_size, n_keypoints = keypoints.shape
        n_keypoints = n_keypoints // 2  # Each keypoint has x and y coordinates
        
        heatmap_size_tensor = torch.tensor(heatmap_size, device=keypoints.device)
        heatmaps = torch.zeros(batch_size, *heatmap_size, device=keypoints.device)

        for i in range(batch_size):
            for j in range(0, n_keypoints, 2):
                center = (keypoints[i, j:j+2] * heatmap_size_tensor).long()
                heatmaps[i] += create_heatmap(heatmap_size, center, sigma_ground)
                if DEBUG:
                    print(f"(keypoints_to_heatmaps)", f"keypoints[{i}, {j}] = {keypoints[i, j].shape} is_ground: {j % 2 == 0}")
            # normalize the heatmap
            heatmaps[i] = heatmaps[i] / torch.max(heatmaps[i])

    return heatmaps  # Shape: (batch_size, n_keypoints, heatmap_size, heatmap_size)


def heatmap_top_k(heatmap, k=16):
    if DEBUG:
        print("(heatmap_top_k)", f"{heatmap.shape=}, {k=}")

    # get the original shape
    original_shape = heatmap.shape

    # flatten the heatmap
    heatmap = heatmap.flatten()

    # get the top k values
    top_k = torch.topk(heatmap, k=k)
    if DEBUG:
        print(f"(heatmap_top_k)", f"{top_k=}, {heatmap.shape=}")

    # get the indices of the top k values
    indices = top_k.indices
    if DEBUG:
        print(f"(heatmap_top_k)", f"{indices=}")

    # get the values of the top k values
    values = top_k.values

    # convert the indices to x, y coordinates based on the original shape
    x = indices % original_shape[1]
    y = indices // original_shape[1]

    # convert the x, y coordinates to a list of tuples
    coordinates = [(x[i].item(), y[i].item()) for i in range(k)]

    # return the coordinates and values
    return coordinates, values


def plot_to_image(figure):
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    
    # Convert PNG buffer to TF image
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    
    return image

def heatmaps_to_keypoints(heatmaps, num_keypoints = 16, writer=None):
    """Convert heatmaps to keypoints."""
    if DEBUG:
        print("(heatmaps_to_keypoints)", f"{heatmaps.shape=}, {num_keypoints=}")

    batch_size, _, heatmap_height, heatmap_width = heatmaps.shape
    keypoints = torch.zeros(batch_size, num_keypoints * heatmaps.shape[1], 2, device=heatmaps.device)  # Each keypoint has x and y coordinates

    for i in range(batch_size):
        for j in range(heatmaps.shape[1]):  # Loop over keypoints
            # save the heatmap as image
            if writer is not None:
                writer.add_image(f"heatmaps_to_keypoints_heatmap_{i}_{j}", heatmaps[i, j], global_step=0, dataformats='HW')

            if DEBUG:
                print(f"(heatmaps_to_keypoints)", f"{i=}, {j=}, {heatmaps[i, j].shape=}")

            # find the max value in the heatmap
            coordinates, values = heatmap_top_k(heatmaps[i, j], k=num_keypoints)
            xs = [coord[0] for coord in coordinates]
            ys = [coord[1] for coord in coordinates]

            # convert to tensors
            xs = torch.tensor(xs, device=heatmaps.device)
            ys = torch.tensor(ys, device=heatmaps.device)

            if DEBUG:
                print(f"(heatmaps_to_keypoints)", f"{xs=}, {ys=}, {heatmap_width=}, {heatmap_height=}, {values=}")

            # Store the coordinates in the keypoints tensor
            for k in range(num_keypoints):
                keypoints[i, j*num_keypoints + k, 0] = xs[k].float() / heatmap_width  # Normalize x coordinates to [0, 1]
                keypoints[i, j*num_keypoints + k, 1] = ys[k].float() / heatmap_height  # Normalize y coordinates to [0, 1]

    if DEBUG:
        print("(heatmaps_to_keypoints)", f"output {keypoints.shape=}")

    return keypoints  # Shape: (batch_size, num_keypoints, 2)

def load_labels(json_paths, nmax=16, scale_factor=1.0):
    if DEBUG:
        print("(load_labels)", f"{json_paths=}, {nmax=}, {scale_factor=}")

    ground_keypoints_all = []  # Each item: [(x1_g1, y1_g1), ...]
    satellite_keypoints_all = []  # Each item: [(x1_s1, y1_s1), ...]
    img_paths = []
    for json_path in json_paths:
        p = json_path.split("/")[-1].replace(".json", ".png")
        img_paths.append(dict(ground_image=p, naip_image=p, worldcover_image=p, elevation_image=p))
        ground_keypoints = []
        satellite_keypoints = []
        parsed = json.load(open(json_path))
        for x in parsed:
            for n, y in enumerate(x):
                if n % 2 == 0:
                    if DEBUG: print("ground", y)
                    ground_keypoints.append(tuple(y))  # Append (x_g, y_g)
                else:
                    if DEBUG: print("satellite", y)
                    satellite_keypoints.append(tuple(y))  # Append (x_s, y_s)
        if DEBUG:
            print("(load_labels)", f"{len(ground_keypoints)=}, {len(satellite_keypoints)=}")
            print("(load_labels)", f"{ground_keypoints=}, {satellite_keypoints=}")

        if len(ground_keypoints) < nmax:
            ground_keypoints += [(0, 0)] * (nmax - len(ground_keypoints))  # Fill with (0, 0)
        if len(satellite_keypoints) < nmax:
            satellite_keypoints += [(0, 0)] * (nmax - len(satellite_keypoints))  # Fill with (0, 0)
        if len(ground_keypoints) > nmax:
            ground_keypoints = ground_keypoints[:nmax]
        if len(satellite_keypoints) > nmax:
            satellite_keypoints = satellite_keypoints[:nmax]
        ground_keypoints = scale_keypoints(ground_keypoints, scale_factor)  # scale keypoints
        satellite_keypoints = scale_keypoints(satellite_keypoints, scale_factor)  # scale keypoints

        if DEBUG:
            print("(load_labels)", f"{len(ground_keypoints)=}, {len(satellite_keypoints)=}")
            print("(load_labels)", f"{ground_keypoints=}, {satellite_keypoints=}")
        
        ground_keypoints_all.append(ground_keypoints)
        satellite_keypoints_all.append(satellite_keypoints)
    
    if DEBUG:
        print("(load_labels)", f"{len(ground_keypoints_all)=}, {len(satellite_keypoints_all)=}")
    
    assert len(ground_keypoints_all) == len(satellite_keypoints_all) == len(img_paths), f"{len(ground_keypoints_all)=}, {len(satellite_keypoints_all)=}, {len(img_paths)=} should be equal"

    satellite_keypoints_all = torch.tensor(satellite_keypoints_all, dtype=torch.float32)
    ground_keypoints_all = torch.tensor(ground_keypoints_all, dtype=torch.float32)
    
    if DEBUG:
        print("(load_labels)", f"{ground_keypoints_all.shape=}, {satellite_keypoints_all.shape=}")

    return img_paths, (ground_keypoints_all, satellite_keypoints_all)


class KeypointTransform:
    def __init__(self, img_transform):
        self.img_transform = img_transform

    def __call__(self, keypoints, img):
        for transform in self.img_transform.transforms:
            if isinstance(transform, RandomHorizontalFlip):
                if torch.rand(1) < transform.p:
                    img = F.hflip(img)
                    keypoints = self.horizontal_flip_keypoints(keypoints, img.width)
            elif isinstance(transform, RandomVerticalFlip):
                if torch.rand(1) < transform.p:
                    img = F.vflip(img)
                    keypoints = self.vertical_flip_keypoints(keypoints, img.height)
            elif isinstance(transform, RandomRotation):
                angle = transform.degrees
                img = F.rotate(img, angle)
                keypoints = self.rotate_keypoints(keypoints, angle, img.width, img.height)
            else:
                img = transform(img)
        return keypoints, img

    def horizontal_flip_keypoints(self, keypoints):
        flipped_keypoints = deepcopy(keypoints)
        flipped_keypoints[0::2] = 1 - keypoints[0::2]
        return flipped_keypoints

    def vertical_flip_keypoints(self, keypoints):
        flipped_keypoints = deepcopy(keypoints)
        flipped_keypoints[1::2] = 1 - keypoints[1::2]
        return flipped_keypoints

    def rotate_keypoints(self, keypoints, angle):
        angle_rad = -angle * np.pi / 180
        rot_matrix = torch.tensor([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])

        xy_keypoints = torch.stack((keypoints[0::2], keypoints[1::2]), dim=-1)
        center = torch.tensor([0.5, 0.5], dtype=torch.float)
        xy_rotated = torch.matmul(xy_keypoints - center, rot_matrix) + center

        rotated_keypoints = deepcopy(keypoints)
        rotated_keypoints[0::2] = xy_rotated[:, 0]
        rotated_keypoints[1::2] = xy_rotated[:, 1]
        return rotated_keypoints

    def apply_all(self, ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints):
        for transform in self.img_transform.transforms:
            if isinstance(transform, RandomHorizontalFlip):
                if torch.rand(1) < transform.p:
                    ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints = self.apply_horizontal_flip(ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints)
            elif isinstance(transform, RandomVerticalFlip):
                if torch.rand(1) < transform.p:
                    ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints = self.apply_vertical_flip(ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints)
            elif isinstance(transform, RandomRotation):
                angle = transform.degrees[0]
                ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints = self.apply_rotation(ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints, angle)
            else:
                ground_img, naip_img, worldcover_img = transform(ground_img), transform(naip_img), transform(worldcover_img)

        return ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints

    def apply_horizontal_flip(self, ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints):
        ground_img, naip_img, worldcover_img = F.hflip(ground_img), F.hflip(naip_img), F.hflip(worldcover_img)
        ground_keypoints = self.horizontal_flip_keypoints(ground_keypoints, ground_img.width)
        satellite_keypoints = self.horizontal_flip_keypoints(satellite_keypoints, naip_img.width)
        return ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints

    def apply_vertical_flip(self, ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints):
        ground_img, naip_img, worldcover_img = F.vflip(ground_img), F.vflip(naip_img), F.vflip(worldcover_img)
        ground_keypoints = self.vertical_flip_keypoints(ground_keypoints, ground_img.height)
        satellite_keypoints = self.vertical_flip_keypoints(satellite_keypoints, naip_img.height)
        return ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints

    def apply_rotation(self, ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints, angle):
        ground_img, naip_img, worldcover_img = F.rotate(ground_img, angle), F.rotate(naip_img, angle), F.rotate(worldcover_img, angle)
        ground_keypoints = self.rotate_keypoints(ground_keypoints, angle, ground_img.width, ground_img.height)
        satellite_keypoints = self.rotate_keypoints(satellite_keypoints, angle, naip_img.width, naip_img.height)
        return ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints

class aKeypointDataset(Dataset):
    def __init__(self, ground_image_dir, naip_image_dir, worldcover_image_dir, labels, image_paths, transform=None, img_transform=None, inf=False, full_length=False):
        self.ground_image_dir = ground_image_dir
        self.naip_image_dir = naip_image_dir
        self.worldcover_image_dir = worldcover_image_dir
        self.ground_keypoints, self.satellite_keypoints = labels
        self.image_paths = image_paths
        self.transform = transform
        self.img_transform = img_transform
        self.inf = inf
        self.full_length = full_length

    def __len__(self):
        if self.inf:
            return len(self.ground_keypoints)
        if self.full_length:
            return len(self.ground_keypoints) * 4
        else: return 32

    def get_labels(self, idx):
        if DEBUG:
            print(f"(get_labels) idx: {idx}")
            print(f"(get_labels) {self.ground_keypoints.shape=}")
            print(f"(get_labels) {self.ground_keypoints[idx].shape=}")

        g_keypoints, s_keypoints = self.ground_keypoints[idx], self.satellite_keypoints[idx]
        # convert to flat list x,y,x,y,x,y,x,y
        ground_points = [item for sublist in g_keypoints for item in sublist]
        satellite_points = [item for sublist in s_keypoints for item in sublist]
        return ground_points, satellite_points

    def labels_to_combined(self, ground_points, satellite_points):
        # opposite of get_labels
        combined_points = []
        for i in range(len(ground_points)):
            combined_points.append(ground_points[i])
            combined_points.append(satellite_points[i])

        return combined_points

    def create_padding_mask(self, img):
        img_np = np.array(img)
        # create mask where rgb values are all 0
        mask = np.zeros((img.shape[-2], img.shape[-1]))
        # set mask where rgb values are not all 0 to 1
        mask[np.where(img[0] != 0)] = 1

        torch_mask = torch.tensor(mask, dtype=torch.float32)
        return torch_mask

    def __getitem__(self, idx):
        if idx >= len(self.ground_keypoints):
            idx %= len(self.ground_keypoints)

        ground_img1_path = os.path.join(self.ground_image_dir, self.image_paths[idx]["ground_image"])
        naip_img1_path = os.path.join(self.naip_image_dir, self.image_paths[idx]["naip_image"])
        worldcover_img1_path = os.path.join(self.worldcover_image_dir, self.image_paths[idx]["worldcover_image"])

        ground_img1 = Image.open(ground_img1_path)
        naip_img1 = Image.open(naip_img1_path)
        worldcover_img1 = Image.open(worldcover_img1_path)
        ground_keypoints, satellite_keypoints = self.get_labels(idx)

        ground_keypoints = torch.tensor(ground_keypoints, dtype=torch.float32)
        satellite_keypoints = torch.tensor(satellite_keypoints, dtype=torch.float32)
        
        # check that ground keypoints and satellite keypoints are [0, 1]
        if torch.any(ground_keypoints > 1) or torch.any(ground_keypoints < 0):
            raise ValueError("Ground keypoints are not normalized")
        if torch.any(satellite_keypoints > 1) or torch.any(satellite_keypoints < 0):
            raise ValueError("Satellite keypoints are not normalized")

        if self.transform:
            ground_img1 = self.transform(ground_img1)
            naip_img1 = self.transform(naip_img1)
            worldcover_img1 = self.transform(worldcover_img1)

        keypoint_transform = KeypointTransform(self.img_transform)

        if self.img_transform:
            # fix this to work with ground_keypoints and satellite_keypoints
            ground_img1, naip_img1, worldcover_img1, ground_keypoints, satellite_keypoints = keypoint_transform.apply_all(ground_img1, naip_img1, worldcover_img1, ground_keypoints, satellite_keypoints)

        # normalize images to imagenet mean and std
        # Mean: [0.485, 0.456, 0.406]
        # Standard Deviation: [0.229, 0.224, 0.225]
        # Define ImageNet mean and standard deviation
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Reshape the mean and std arrays
        mean = mean[:, None, None]
        std = std[:, None, None]

        ground_img1 = (ground_img1 - mean) / std
        naip_img1 = (naip_img1 - mean) / std
        worldcover_img1 = (worldcover_img1 - mean) / std
        
        # convert to double
        ground_img1 = ground_img1.float()
        naip_img1 = naip_img1.float()
        worldcover_img1 = worldcover_img1.float()

        # Create padding masks for each input image type
        ground_padding_mask = self.create_padding_mask(ground_img1)
        naip_padding_mask = self.create_padding_mask(naip_img1)
        worldcover_padding_mask = self.create_padding_mask(worldcover_img1)

        return ground_img1, naip_img1, worldcover_img1, ground_keypoints, satellite_keypoints, ground_padding_mask, naip_padding_mask, worldcover_padding_mask

