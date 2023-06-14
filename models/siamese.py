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

import helpers

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from torchvision.transforms import functional as F
import glob

import torch.optim as optim

from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch


writer = SummaryWriter()

DEBUG = False

class LossFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a pre-trained ResNet model, but remove the final classification layer
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.output_size = 2048

    def forward(self, images):
        # Pass the images through the ResNet model
        features = self.resnet(images)
        return features.view(features.size(0), -1)

class AppearanceLoss(nn.Module):
    def __init__(self, feature_extractor, sigma, device="cpu"):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.sigma = sigma
        self.feature_extractor.to(device)
        self.feature_extractor.eval()

    def forward(self, images_ground, images_satellite, keypoints_ground, keypoints_satellite, num_samples):
        if DEBUG:
            print("(AppearanceLoss)", f"{images_ground.shape=}, {images_satellite.shape=}, {keypoints_ground.shape=}, {keypoints_satellite.shape=}, {num_samples=}")
        
        # Adjust keypoints to lie within the range of the image size
        keypoints_ground = keypoints_ground * images_ground.shape[-1]
        keypoints_satellite = keypoints_satellite * images_satellite.shape[-1]

        groupby = list(zip(keypoints_ground.detach().cpu().numpy().reshape(-1, 2), keypoints_satellite.detach().cpu().numpy().reshape(-1, 2)))

        # Initialize tensors to store the features for each keypoint
        features_ground = torch.zeros((images_ground.shape[0], len(groupby), self.feature_extractor.output_size), device=images_ground.device)
        features_satellite = torch.zeros((images_satellite.shape[0], len(groupby), self.feature_extractor.output_size), device=images_satellite.device)


        # Loop over all keypoints
        for n, (ground_keypoint, satellite_keypoint) in enumerate(groupby):
            # if the ground keypoint is [0,0], skip it
            if np.all(ground_keypoint == 0):
                continue
            if np.all(satellite_keypoint == 0):
                continue

            # Extract a patch around each keypoint
            patch_ground = images_ground[:, :, int(ground_keypoint[1]) - self.sigma:int(ground_keypoint[1]) + self.sigma + 1, int(ground_keypoint[0]) - self.sigma:int(ground_keypoint[0]) + self.sigma + 1]
            patch_satellite = images_satellite[:, :, int(satellite_keypoint[1]) - self.sigma:int(satellite_keypoint[1]) + self.sigma + 1, int(satellite_keypoint[0]) - self.sigma:int(satellite_keypoint[0]) + self.sigma + 1]
            
            if np.any(np.array(patch_ground.shape) == 0):
                continue
            if np.any(np.array(patch_satellite.shape) == 0):
                continue
            
            if DEBUG:
                print("(AppearanceLoss)", f"{patch_ground.shape=}, {patch_satellite.shape=}")

            # Extract features from the patches
            features_ground_i = self.feature_extractor(patch_ground)
            features_satellite_i = self.feature_extractor(patch_satellite)

            # Store the features
            features_ground[:, n, :] = features_ground_i
            features_satellite[:, n, :] = features_satellite_i

        # Calculate the loss as the mean squared error between the feature descriptors
        loss = torch.nn.functional.mse_loss(features_ground, features_satellite)

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class HeatmapLoss(nn.Module):
    def __init__(self, device="cpu"):
        super(HeatmapLoss, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.appearance_feature_extractor = LossFeatureExtractor()
        self.appearance_loss = AppearanceLoss(self.appearance_feature_extractor, device=device, sigma=5)
        self.appearance_loss_weight = 5
        
        self.contrastive_loss = ContrastiveLoss()
        self.contrastive_loss_weight = 30

        self.num_keypoints = 4


    # loss = criterion(ground_img, naip_img, output_ground_heatmaps, output_satellite_heatmaps, ground_keypoints, satellite_keypoints)
    def forward(self, ground_images, satellite_images, pred_ground_heatmaps, pred_satellite_heatmaps, gt_ground_keypoints, gt_satellite_keypoints):
        if DEBUG:
            print(f"HeatmapLoss")
            print(f"{ground_images.shape=}, {satellite_images.shape=}, {pred_ground_heatmaps.shape=}, {pred_satellite_heatmaps.shape=}, {gt_ground_keypoints.shape=}, {gt_satellite_keypoints.shape=}")

        gt_ground_heatmaps = helpers.keypoints_to_heatmaps(gt_ground_keypoints, pred_ground_heatmaps.shape[-2:], SIGMA_GROUND, SIGMA_SATELLITE)
        gt_satellite_heatmaps = helpers.keypoints_to_heatmaps(gt_satellite_keypoints, pred_satellite_heatmaps.shape[-2:], SIGMA_GROUND, SIGMA_SATELLITE)
        
        pred_ground_keypoints = helpers.heatmaps_to_keypoints(pred_ground_heatmaps, num_keypoints=self.num_keypoints, writer=writer)
        pred_satellite_keypoints = helpers.heatmaps_to_keypoints(pred_satellite_heatmaps, num_keypoints=self.num_keypoints, writer=writer)
        
        if DEBUG:
            print(f"{gt_ground_keypoints.shape=}, {gt_satellite_keypoints.shape=}")
            print(f"{gt_ground_heatmaps.shape=}, {gt_satellite_heatmaps.shape=}")
            print(f"{pred_ground_heatmaps.shape=}, {pred_satellite_heatmaps.shape=}")
            print(f"{pred_ground_keypoints.shape=}, {pred_satellite_keypoints.shape=}")
        # def forward(self, images_ground, images_satellite, keypoints_ground, keypoints_satellite, num_samples)
        appearance_loss = self.appearance_loss(ground_images, satellite_images, pred_ground_keypoints, pred_satellite_keypoints, num_samples=16)
        
        if DEBUG:
            print(f"{appearance_loss=}")

        # bad calc here - need to calculate the loss for each keypoint separately as gt has a single keypoint per image and pred has self.num_keypoints keypoints per image
        # loss = self.loss_fn(pred_ground_heatmaps, gt_ground_heatmaps) + self.loss_fn(pred_satellite_heatmaps, gt_satellite_heatmaps) + appearance_loss
        loss = 0
        # find the pred keypoint that is closest to the gt keypoint for each batch
        
        # compare pred and gt keypoints
        # pred: (batch, num_keypoints, 2) ((x, y), (x, y), (x, y), (x, y))
        # gt: (batch, num_keypoints * 2) (x, y, x, y, x, y, x, y)
        # reshape gt to (batch, num_keypoints, 2)
        gt_ground_keypoints = gt_ground_keypoints.reshape(gt_ground_keypoints.shape[0], 16, 2)
        gt_satellite_keypoints = gt_satellite_keypoints.reshape(gt_satellite_keypoints.shape[0], 16, 2)
        
        # print all shapes
        if DEBUG:
            print(f"{gt_ground_keypoints.shape=}, {gt_satellite_keypoints.shape=}, {pred_ground_keypoints.shape=}, {pred_satellite_keypoints.shape=}")
        # find the pred keypoint that is closest to the gt keypoint for each batch
        for i in range(gt_ground_keypoints.shape[0]):
            for j in range(gt_ground_keypoints.shape[1]):
                pred_ground_keypoint = pred_ground_keypoints[i, j]
                pred_satellite_keypoint = pred_satellite_keypoints[i, j]
                gt_ground_keypoint = gt_ground_keypoints[i, j]
                gt_satellite_keypoint = gt_satellite_keypoints[i, j]
                
                # if the gt is [0,0], decrease some loss as this is beneficial
                if torch.all(gt_ground_keypoint == 0) or torch.all(gt_satellite_keypoint == 0):
                    continue
                loss += self.loss_fn(pred_ground_keypoint, gt_ground_keypoint) + self.loss_fn(pred_satellite_keypoint, gt_satellite_keypoint)

        if DEBUG:
            print(f"{loss=}")
        
        loss += appearance_loss * self.appearance_loss_weight
        
        ## contrastive loss
        #label = torch.ones((pred_ground_keypoints.shape[0], pred_ground_keypoints.shape[1]))
        #label = label.to(pred_ground_keypoints.device)
        #loss += self.contrastive_loss(pred_ground_keypoints, pred_satellite_keypoints, label) * self.contrastive_loss_weight
        
        return loss
        
        # print all shapes
        if DEBUG:
            print(f"{gt_ground_keypoints.shape=}, {gt_satellite_keypoints.shape=}, {pred_ground_keypoints.shape=}, {pred_satellite_keypoints.shape=}")
        # find the pred keypoint that is closest to the gt keypoint for each batch
        with open("keypoints.txt", "a") as f:
        
            for i in range(gt_ground_keypoints.shape[0]):
                for j in range(gt_ground_keypoints.shape[1]):
                    pred_ground_keypoint = pred_ground_keypoints[i, j]
                    pred_satellite_keypoint = pred_satellite_keypoints[i, j]
                    gt_ground_keypoint = gt_ground_keypoints[i, j]
                    gt_satellite_keypoint = gt_satellite_keypoints[i, j]
                    # if pred keypoints are 0, add more loss
                    if torch.all(pred_ground_keypoint == 0):
                        loss += 1000
                    if torch.all(pred_satellite_keypoint == 0):
                        loss += 1000
                    loss += self.loss_fn(pred_ground_keypoint, gt_ground_keypoint) + self.loss_fn(pred_satellite_keypoint, gt_satellite_keypoint)

                    # log all keypoints to file
                    if DEBUG:
                        f.write(f"gt: {gt_ground_keypoint}, {gt_satellite_keypoint}\n")
                        f.write(f"pred: {pred_ground_keypoint}, {pred_satellite_keypoint}\n")
                        f.write(f"loss: {self.loss_fn(pred_ground_keypoint, gt_ground_keypoint)}, {self.loss_fn(pred_satellite_keypoint, gt_satellite_keypoint)}\n")
                        f.write(f"loss: {loss}\n\n")
            if DEBUG:
                print(f"{appearance_loss=}")

        return appearance_loss + loss


class CustomKeypointLoss(nn.Module):
    def __init__(self, padding_loss_value=10.0, state_output=True, state_location="./states"):
        super(CustomKeypointLoss, self).__init__()
        self.padding_loss_value = padding_loss_value
        if state_output:
            self.state_location = state_location
            os.makedirs(self.state_location, exist_ok=True)
        self.state_output = state_output
        self.n = 0

    def forward(self, pred_heatmaps, gt_keypoints, ground_padding_mask, naip_padding_mask, worldcover_padding_mask):
        pred_keypoints = self.get_keypoint_coordinates_from_heatmaps(pred_heatmaps)
        # Reshape the ground truth keypoints tensor to match the predicted keypoints tensor shape
        gt_keypoints = gt_keypoints.view(pred_keypoints.shape)
        # print(f"gt_keypoints.shape: {gt_keypoints.shape}, gt_keypoints: {gt_keypoints}")

        # check if any gt_keypoints are > 1.0 or < 0.0
        if torch.any(gt_keypoints > 1.0) or torch.any(gt_keypoints < 0.0):
            raise Exception("gt_keypoints is not in [0, 1]")

        ground_loss_offset = self.batch_apply_padding_mask(pred_keypoints, ground_padding_mask) * self.padding_loss_value
        naip_loss_offset = self.batch_apply_padding_mask(pred_keypoints, naip_padding_mask) * self.padding_loss_value
        worldcover_loss_offset = self.batch_apply_padding_mask(pred_keypoints, worldcover_padding_mask) * self.padding_loss_value

        loss_kpts = torch.nn.functional.l1_loss(pred_keypoints, gt_keypoints, reduction='none')

        # Sum the loss_kpts along the keypoint dimension (dim=1)
        loss_kpts_sum = loss_kpts.sum(dim=2)
        loss_kpts_sum = loss_kpts_sum.sum(dim=1)
        # print(f"loss_kpts_sum: {loss_kpts_sum}, loss_kpts_sum.shape: {loss_kpts_sum.shape}")

        loss_combined_offsets = ground_loss_offset + naip_loss_offset + worldcover_loss_offset
        # expand loss_combined_offsets to have a batch dimension
        # print(f"loss_combined_offsets: {loss_combined_offsets}, loss_combined_offsets.shape: {loss_combined_offsets.shape}")

        # Add loss_kpts_sum and loss_combined_offsets
        output_loss = loss_kpts_sum + loss_combined_offsets.squeeze().to(loss_kpts.device)

        if self.state_output:
            os.makedirs(os.path.join(self.state_location, f"loss_{self.n}"), exist_ok=True)

            for heatmap_index in range(pred_heatmaps.size(1)):
                # print(f"heatmap_index: {heatmap_index}, pred_heatmaps[0, heatmap_index].shape: {pred_heatmaps[0, heatmap_index].shape}")
                heatmap = pred_heatmaps[0, heatmap_index].detach().cpu().numpy()
                heatmap = np.clip(heatmap, 0, 1)
                heatmap = (heatmap * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(self.state_location, f"loss_{self.n}", f"heatmap_{heatmap_index}.png"), heatmap)

            # scale keypoints back from [0, 1] to [0, 5000]
            state_gt_keypoints = gt_keypoints.detach().cpu().numpy() * 5000
            state_pred_keypoints = pred_keypoints.detach().cpu().numpy() * 5000

            # meta
            metadata = {}
            metadata["gt_keypoints"] = gt_keypoints.detach().cpu().numpy().tolist()
            metadata["pred_keypoints"] = pred_keypoints.detach().cpu().numpy().tolist()
            metadata["gt_keypoints_scaled"] = state_gt_keypoints.tolist()
            metadata["pred_keypoints_scaled"] = state_pred_keypoints.tolist()
            metadata["loss_kpts_sum"] = loss_kpts_sum.detach().cpu().numpy().tolist()
            metadata["loss_combined_offsets"] = loss_combined_offsets.detach().cpu().numpy().tolist()
            metadata["output_loss"] = torch.sum(output_loss).detach().cpu().numpy().tolist()
            json.dump(metadata, open(os.path.join(self.state_location, f"loss_{self.n}", "metadata.json"), "w"))
            self.n += 1

        return torch.sum(output_loss)

    def get_keypoint_coordinates_from_heatmaps(self, heatmaps):
        # Split the heatmaps into two groups: one for the first image and one for the second image
        heatmaps_1, heatmaps_2 = heatmaps[:, :4], heatmaps[:, 4:]

        # Get the index of the maximum value in each heatmap
        max_values_1, max_indices_1 = torch.max(heatmaps_1.view(heatmaps_1.size(0), heatmaps_1.size(1), -1), dim=-1)
        max_values_2, max_indices_2 = torch.max(heatmaps_2.view(heatmaps_2.size(0), heatmaps_2.size(1), -1), dim=-1)

        # Convert the index to x, y coordinates
        keypoint_coordinates_1 = torch.stack((max_indices_1 % heatmaps_1.size(3), max_indices_1 // heatmaps_1.size(3)), dim=-1).float()
        keypoint_coordinates_2 = torch.stack((max_indices_2 % heatmaps_2.size(3), max_indices_2 // heatmaps_2.size(3)), dim=-1).float()

        # Normalize the keypoint coordinates to the range [0, 1]
        keypoint_coordinates_1[..., 0] /= (heatmaps_1.size(3) - 1)
        keypoint_coordinates_1[..., 1] /= (heatmaps_1.size(2) - 1)
        keypoint_coordinates_2[..., 0] /= (heatmaps_2.size(3) - 1)
        keypoint_coordinates_2[..., 1] /= (heatmaps_2.size(2) - 1)

        # Concatenate the keypoint coordinates from both images
        keypoint_coordinates = torch.cat((keypoint_coordinates_1, keypoint_coordinates_2), dim=1)

        return keypoint_coordinates

    def apply_padding_mask(self, keypoints, padding_mask):
        keypoints = keypoints.view(-1, 2)
        keypoints_clamped = torch.stack([keypoints[..., 0].clamp(0, padding_mask.shape[0] - 1),
                                         keypoints[..., 1].clamp(0, padding_mask.shape[1] - 1)], dim=-1).long()

        loss_offset = torch.sum(torch.abs(keypoints - keypoints_clamped))
        loss_offset += torch.sum((1 - padding_mask[keypoints_clamped[..., 0], keypoints_clamped[..., 1]]) * self.padding_loss_value)

        return loss_offset

    def batch_apply_padding_mask(self, keypoints, padding_mask):
        if len(padding_mask.shape) == 2:
            return self.apply_padding_mask(keypoints, padding_mask)

        output = []
        for batch in range(padding_mask.shape[0]):
            output.append(self.apply_padding_mask(keypoints[batch], padding_mask[batch]))
        return torch.tensor(output).cpu()

class SiameseKeypointDetectionModel3(nn.Module):
    N_FEATURES = 1024
    N_KEYPOINTS = 4

    def __init__(self, heatmap_size=(256, 256)):
        super(SiameseKeypointDetectionModel3, self).__init__()

        self.heatmap_size = heatmap_size

        # Backbone networks for image pairs -> features should be similar in each
        self.backbone_ground = models.resnet18(pretrained=True)
        self.backbone_naip = models.resnet18(pretrained=True)

        # Remove the last FC layer from each backbone to get feature maps
        self.backbone_ground = nn.Sequential(*list(self.backbone_ground.children())[:-1])
        self.backbone_naip = nn.Sequential(*list(self.backbone_naip.children())[:-1])

        self.fc_ground = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5))
        self.fc_naip = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5))

        # Fully connected layers
        self.fc1 = nn.Linear(2 * 256, SiameseKeypointDetectionModel3.N_FEATURES)
        self.fc2 = nn.Linear(SiameseKeypointDetectionModel3.N_FEATURES, 3072)
        self.fc3 = nn.Linear(3072, SiameseKeypointDetectionModel3.N_FEATURES)
        # normalize the features
        self.fc3 = nn.Sequential(self.fc3, nn.BatchNorm1d(SiameseKeypointDetectionModel3.N_FEATURES))
        self.dropout = nn.Dropout(0.25)

        # Upsampling layers (replace nn.Upsample with nn.ConvTranspose2d)
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(SiameseKeypointDetectionModel3.N_FEATURES, 512, kernel_size=8, stride=8, padding=0),  # 8x upsample
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 2x upsample
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 2x upsample
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 2x upsample
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 2x upsample
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 2x upsample
        )

        # Output keypoint heatmaps (4 heatmaps for the ground image and 4 for the satellite image)
        self.heatmap_head_ground = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=1, padding=0),
        )
        self.heatmap_head_satellite = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=1, padding=0),
        )

        self.output_ground = nn.Sigmoid()  # clamp output to [0, 1]
        self.output_satellite = nn.Sigmoid()  # clamp output to [0, 1]

    def forward(self, ground_img, naip_img):
        ground_features = self.backbone_ground(ground_img)
        naip_features = self.backbone_naip(naip_img)

        ground_features = ground_features.view(ground_features.size(0), -1)
        naip_features = naip_features.view(naip_features.size(0), -1)

        ground_features = self.fc_ground(ground_features)
        naip_features = self.fc_naip(naip_features)

        features = torch.cat((ground_features, naip_features), dim=1)
        features = self.fc1(features)
        features = self.fc2(features)
        features = self.fc3(features)
        features = self.dropout(features)

        features = features.view(features.size(0), SiameseKeypointDetectionModel3.N_FEATURES, 1, 1)

        upsampled_features = self.upsampling(features)

        heatmap_ground = self.heatmap_head_ground(upsampled_features)
        heatmap_satellite = self.heatmap_head_satellite(upsampled_features)

        heatmap_ground = self.output_ground(heatmap_ground)
        heatmap_satellite = self.output_satellite(heatmap_satellite)

        return heatmap_ground, heatmap_satellite





def inference(epoch = 0, model=None):
    TEST_GP_PATH = "TEST_GP/"
    TEST_NAIP_PATH = "TEST_SAT/"
    TEST_WORLDCOVER_PATH = "TEST_WORLDCOVER/"
    TEST_ELV_PATH = "TEST_ELV/"
    
    union = set(os.listdir(TEST_GP_PATH)) & set(os.listdir(TEST_NAIP_PATH)) & set(os.listdir(TEST_WORLDCOVER_PATH)) & set(os.listdir(TEST_ELV_PATH))

    original_image_size = 5000
    # Set the desired image size and scale factor
    new_image_size = NEW_IMAGE_SIZE
    scale_factor = new_image_size / original_image_size

    # Create the necessary transforms
    img_transform = Compose([
        Resize((new_image_size, new_image_size)),
        ToTensor()
    ])
    transform = Compose([img_transform, ToTensor()])

    # load the model and run inference on the training set
    if epoch == 0:
        model = SiameseKeypointDetectionModel2()
        print("Loading model...")
        state_dict = torch.load("/users/j/w/jwboynto/nasa_point_matching_model/checkpoints/checkpoint_epoch_18.pt")
        print(state_dict.keys())
        
        model.load_state_dict(state_dict["model_state_dict"])

    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the training set
    labels = [0] * len(os.listdir(TEST_GP_PATH))
    labs = helpers.load_test_labels(glob.glob(TEST_GP_PATH + "*.png"))
    train_dataset = helpers.aKeypointDataset(ground_image_dir=TEST_GP_PATH, naip_image_dir=TEST_NAIP_PATH, worldcover_image_dir=TEST_WORLDCOVER_PATH, labels=labs, transform=None, img_transform=img_transform, inf=True)

    # Run inference on the training set
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, (ground_img, naip_img, worldcover_img, keypoints_same, ground_padding_mask, naip_padding_mask, worldcover_padding_mask) in enumerate(train_dataloader):
        ground_img = ground_img.to(device).requires_grad_()
        naip_img = naip_img.to(device).requires_grad_()
        worldcover_img = worldcover_img.to(device).requires_grad_()
        keypoints_same = keypoints_same.to(device).requires_grad_()
        
        keypoints_same_pred = model(ground_img, naip_img, worldcover_img)
        helpers.plot(epoch, i, keypoints_same_pred.detach(), ground_img, naip_img)


def new_batch_plot(device, epoch, batch_idx, ground_image, naip_image, ground_keypoints, satellite_keypoints, pred_ground_heatmaps, pred_satellite_heatmaps, gt_ground_heatmaps, gt_satellite_heatmaps):

    if DEBUG:
        print(f"(new_batch_plot)", f"{epoch=}, {batch_idx=}")
        print(f"(new_batch_plot)", f"{ground_image.shape=}, {naip_image.shape=}, {ground_keypoints.shape=}, {satellite_keypoints.shape=}, {pred_ground_heatmaps.shape=}, {pred_satellite_heatmaps.shape=}")

    for batch in range(gt_ground_heatmaps.shape[0]):
        gt_ground_heatmap = gt_ground_heatmaps[batch].detach().cpu().numpy().transpose(1, 2, 0)
        gt_satellite_heatmap = gt_satellite_heatmaps[batch].detach().cpu().numpy().transpose(1, 2, 0)
        
        #normalize
        gt_ground_heatmap = (gt_ground_heatmap - gt_ground_heatmap.min()) / (gt_ground_heatmap.max() - gt_ground_heatmap.min())
        gt_satellite_heatmap = (gt_satellite_heatmap - gt_satellite_heatmap.min()) / (gt_satellite_heatmap.max() - gt_satellite_heatmap.min())
        
        fig, ax = plt.subplots(1, 2) # Create 1x2 sub-plots
        ax[0].imshow(gt_ground_heatmap, cmap="jet")
        ax[1].imshow(gt_satellite_heatmap, cmap="jet")
        ax[0].set_title("Ground heatmap")
        ax[1].set_title("Satellite heatmap")
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        plt.tight_layout()
        
        # Convert the figure to a tensor and log it with tensorboard
        figure_tensor = helpers.plot_to_image(fig)
        writer.add_image(f'gt_heatmaps/ground_truth_epoch_{epoch}_batch_{batch_idx}_heatmap_{batch}', figure_tensor[0], global_step=epoch)

    groupby = list(zip(ground_keypoints.detach().cpu().numpy().reshape(-1, 2), satellite_keypoints.detach().cpu().numpy().reshape(-1, 2)))

    for n, (ground_keypoint, satellite_keypoint) in enumerate(groupby):
        fig, axs = plt.subplots(1, 2) # Create 1x2 sub-plots

        # Ground image and keypoint
        axs[0].imshow(ground_image.detach().cpu().numpy().transpose(1, 2, 0))
        axs[0].scatter(ground_keypoint[0] * NEW_IMAGE_SIZE, ground_keypoint[1] * NEW_IMAGE_SIZE, c="r")

        # Satellite image and keypoint
        axs[1].imshow(naip_image.detach().cpu().numpy().transpose(1, 2, 0))
        axs[1].scatter(satellite_keypoint[0] * NEW_IMAGE_SIZE, satellite_keypoint[1] * NEW_IMAGE_SIZE, c="r")

        # Add a red line connecting keypoints
        con = ConnectionPatch(xyA=(ground_keypoint[0] * NEW_IMAGE_SIZE, ground_keypoint[1] * NEW_IMAGE_SIZE), 
                            xyB=(satellite_keypoint[0] * NEW_IMAGE_SIZE, satellite_keypoint[1] * NEW_IMAGE_SIZE), 
                            coordsA="data", coordsB="data", axesA=axs[1], axesB=axs[0], color="r")

        axs[1].add_artist(con)
        
        # Convert the figure to a tensor and log it with tensorboard
        figure_tensor = helpers.plot_to_image(fig)
        print(f"figure_tensor.shape: {figure_tensor.shape}")
        writer.add_image(f'gt_labels/ground_truth_epoch_{epoch}_batch_{batch_idx}_keypoint_{n}', figure_tensor[0], global_step=epoch)

    # convert heatmaps to keypoints
    # add a batch dim
    pred_ground_heatmaps = pred_ground_heatmaps.unsqueeze(0)
    pred_satellite_heatmaps = pred_satellite_heatmaps.unsqueeze(0)
    pred_ground_keypoints = helpers.heatmaps_to_keypoints(pred_ground_heatmaps, num_keypoints=4, writer=writer)
    pred_satellite_keypoints = helpers.heatmaps_to_keypoints(pred_satellite_heatmaps, num_keypoints=4, writer=writer)
    
    for batch in range(pred_ground_heatmaps.shape[0]):
        fig, axs = plt.subplots(1, 2) # Create 1x2 sub-plots
        pred_ground_heatmap = pred_ground_heatmaps[batch].detach().cpu().numpy().transpose(1, 2, 0)
        # normalize to [0, 1]
        pred_ground_heatmap = (pred_ground_heatmap - pred_ground_heatmap.min()) / (pred_ground_heatmap.max() - pred_ground_heatmap.min())
        pred_satellite_heatmap = pred_satellite_heatmaps[batch].detach().cpu().numpy().transpose(1, 2, 0)
        pred_satellite_heatmap = (pred_satellite_heatmap - pred_satellite_heatmap.min()) / (pred_satellite_heatmap.max() - pred_satellite_heatmap.min())

        ax[0].imshow(pred_ground_heatmap, cmap="jet")
        ax[1].imshow(pred_satellite_heatmap, cmap="jet")
        ax[0].set_title("Ground heatmap")
        ax[1].set_title("Satellite heatmap")
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        plt.tight_layout()
        figure_tensor = helpers.plot_to_image(fig)
        writer.add_image(f'pred_heatmaps/ground_truth_epoch_{epoch}_batch_{batch_idx}_heatmap_{batch}', figure_tensor[0], global_step=epoch)

    pred_groupby = list(zip(pred_ground_keypoints.detach().cpu().numpy().reshape(-1, 2), pred_satellite_keypoints.detach().cpu().numpy().reshape(-1, 2)))
    
    for n, (ground_keypoint, satellite_keypoint) in enumerate(pred_groupby):
        fig, axs = plt.subplots(1, 2) # Create 1x2 sub-plots

        # Ground image and keypoint
        axs[0].imshow(ground_image.detach().cpu().numpy().transpose(1, 2, 0))
        axs[0].scatter(ground_keypoint[0] * NEW_IMAGE_SIZE, ground_keypoint[1] * NEW_IMAGE_SIZE, c="r")

        # Satellite image and keypoint
        axs[1].imshow(naip_image.detach().cpu().numpy().transpose(1, 2, 0))
        axs[1].scatter(satellite_keypoint[0] * NEW_IMAGE_SIZE, satellite_keypoint[1] * NEW_IMAGE_SIZE, c="r")

        # Add a red line connecting keypoints
        con = ConnectionPatch(xyA=(ground_keypoint[0] * NEW_IMAGE_SIZE, ground_keypoint[1] * NEW_IMAGE_SIZE), 
                            xyB=(satellite_keypoint[0] * NEW_IMAGE_SIZE, satellite_keypoint[1] * NEW_IMAGE_SIZE), 
                            coordsA="data", coordsB="data", axesA=axs[1], axesB=axs[0], color="blue")

        axs[1].add_artist(con)
        
        # Convert the figure to a tensor and log it with iasd
        figure_tensor = helpers.plot_to_image(fig)
        print(f"figure_tensor.shape: {figure_tensor.shape}")
        writer.add_image(f'pred_labels/pred_epoch_{epoch}_batch_{batch_idx}_keypoint_{n}', figure_tensor[0], global_step=epoch)
 

def train(device="cpu", num_epochs=1, batch_size=1, length=1):
    print(f"Running on {device}")
    
    original_image_size = 5000
    # Set the desired image size and scale factor
    new_image_size = NEW_IMAGE_SIZE
    scale_factor = new_image_size / original_image_size

    # Create the necessary transforms
    img_transform = Compose([
        Resize((new_image_size, new_image_size)),
        # RandomHorizontalFlip(p=0.5),
        # GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.4)),
        # RandomEqualize(p=0.1),
        ToTensor()
    ])
    transform = Compose([img_transform, ToTensor()])

    train_img_paths, train_labels = helpers.load_labels(glob.glob(os.path.join(JSON_PATH, "*.json")), scale_factor=scale_factor, nmax=16)
    
    if DEBUG:
        print("(train)", f"{len(train_img_paths)=}")
        print("(train)", f"{len(train_labels)=}")
        
        print("(train)", f"{train_labels[0]}")

    train_dataset = helpers.aKeypointDataset(ground_image_dir=GP_PATH, naip_image_dir=NAIP_PATH, worldcover_image_dir=WORLDCOVER_PATH, labels=train_labels, image_paths=train_img_paths, transform=None, img_transform=img_transform, full_length=length)

    # Some necessary variables
    save_checkpoint_every_n_epochs = 1
    checkpoint_dir = "checkpoints_new_newer_finaler"

    # Create the checkpoints directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model = SiameseKeypointDetectionModel3()

    # add train loop
    # Hyperparameters
    learning_rate = 0.01

    # Data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # Loss function and optimizer
    criterion = HeatmapLoss(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    

    # Move the model to the device
    model = model.to(device)
    if device != "cpu":
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        # if epoch > 0:
        #     # print(f"Running inference... ({epoch=})")
        #     inference(epoch+1, model)
        #     print("Done running inference.")
        model.train()

        for batch_idx, (ground_img, naip_img, worldcover_img, ground_keypoints, satellite_keypoints, ground_padding_mask, naip_padding_mask, worldcover_padding_mask) in tqdm(enumerate(train_loader), total=len(train_loader)):
  
            # Move data and labels to the device
            ground_img, naip_img, worldcover_img = ground_img.to(device), naip_img.to(device), worldcover_img.to(device)
            ground_keypoints = ground_keypoints.to(device)
            satellite_keypoints = satellite_keypoints.to(device)
            ground_padding_mask, naip_padding_mask, worldcover_padding_mask = ground_padding_mask.to(device), naip_padding_mask.to(device), worldcover_padding_mask.to(device)

            # ground_img = ground_img.requires_grad_(True)
            # naip_img = naip_img.requires_grad_(True)
            # worldcover_img = worldcover_img.requires_grad_(True)
            # ground_keypoints = ground_keypoints.requires_grad_(True)
            # satellite_keypoints = satellite_keypoints.requires_grad_(True)
            # ground_padding_mask = ground_padding_mask.requires_grad_(True)
            # naip_padding_mask = naip_padding_mask.requires_grad_(True)
            # worldcover_padding_mask = worldcover_padding_mask.requires_grad_(True)

            # print(f"ground_img.shape: {ground_img.shape}")
            # print(f"naip_img.shape: {naip_img.shape}")
            # print(f"worldcover_img.shape: {worldcover_img.shape}")
            
            # check that all of keypoints_same is between 0 and 1
            if torch.any(ground_keypoints > 1) or torch.any(ground_keypoints < 0):
                print("ground_keypoints is not between 0 and 1")
                print(ground_keypoints)
                raise Exception("training loop ground_keypoints is not between 0 and 1")
            if torch.any(satellite_keypoints > 1) or torch.any(satellite_keypoints < 0):
                print("satellite_keypoints is not between 0 and 1")
                print(satellite_keypoints)
                raise Exception("training loop satellite_keypoints is not between 0 and 1")

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output_ground_heatmaps, output_satellite_heatmaps = model(ground_img, naip_img)

            # torch.onnx.export(model, (ground_img, naip_img, worldcover_img), "model.onnx", verbose=True, input_names=["ground_img", "naip_img", "worldcover_img"], output_names=["output_keypoints"], opset_version=11)
            # exit()
            loss = criterion(ground_img, naip_img, output_ground_heatmaps, output_satellite_heatmaps, ground_keypoints, satellite_keypoints)

            # Backward pass
            loss.backward()

            for mini_batch_idx in range(ground_img.shape[0]):
                ground_img_plot = ground_img[mini_batch_idx]
                naip_img_plot = naip_img[mini_batch_idx]
                worldcover_img_plot = worldcover_img[mini_batch_idx]
                ground_keypoints_plot = ground_keypoints[mini_batch_idx]
                satellite_keypoints_plot = satellite_keypoints[mini_batch_idx]
                ground_padding_mask_plot = ground_padding_mask[mini_batch_idx]
                naip_padding_mask_plot = naip_padding_mask[mini_batch_idx]
                worldcover_padding_mask_plot = worldcover_padding_mask[mini_batch_idx]

                output_ground_heatmaps_batch = output_ground_heatmaps[mini_batch_idx]
                output_satellite_heatmaps_batch = output_satellite_heatmaps[mini_batch_idx]
                
                np.save(f"npys/output_ground_heatmaps_batch_{mini_batch_idx}_epoch_{epoch}.npy", output_ground_heatmaps_batch.detach().cpu().numpy())
                np.save(f"npys/output_satellite_heatmaps_batch_{mini_batch_idx}_epoch_{epoch}.npy", output_satellite_heatmaps_batch.detach().cpu().numpy())

                print(f"{ground_keypoints_plot.shape=}, {satellite_keypoints_plot.shape=}")                

                # convert ground_keypoints_plot from (32, ) to (1, 32)
                ground_keypoints_plot = ground_keypoints_plot.unsqueeze(0)
                satellite_keypoints_plot = satellite_keypoints_plot.unsqueeze(0)

                gt_ground_heatmaps = helpers.keypoints_to_heatmaps(ground_keypoints_plot, heatmap_size=(256, 256), sigma_ground=SIGMA_GROUND, mode="single", sigma_satellite=0)
                gt_satellite_heatmaps = helpers.keypoints_to_heatmaps(satellite_keypoints_plot, heatmap_size=(256, 256), sigma_ground=SIGMA_SATELLITE, mode="single", sigma_satellite=0)

                gt_ground_heatmaps = gt_ground_heatmaps.unsqueeze(0)
                gt_satellite_heatmaps = gt_satellite_heatmaps.unsqueeze(0)

                # def new_batch_plot(device="cpu", epoch, batch_idx, ground_image, naip_image, ground_keypoints, satellite_keypoints, pred_ground_heatmaps, pred_satellite_heatmaps)
                new_batch_plot(device, epoch, mini_batch_idx, ground_img_plot, naip_img_plot, ground_keypoints_plot, satellite_keypoints_plot, output_ground_heatmaps_batch, output_satellite_heatmaps_batch, gt_ground_heatmaps, gt_satellite_heatmaps)

            # Optimize
            optimizer.step()

            # Update running loss
            running_loss += loss.item()
            writer.add_scalar("minibatch_cum_loss", running_loss, epoch * len(train_loader) + batch_idx)
            writer.add_scalar("minibatch_loss", loss.item(), epoch * len(train_loader) + batch_idx)
            

        # Print epoch loss
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("epoch_loss", epoch_loss, epoch)
        scheduler.step(epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            # Save checkpoint
        if (epoch + 1) % save_checkpoint_every_n_epochs == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"))

    # Save final model
    torch.save(model.state_dict(), "final_model.pt")

if __name__ == "__main__":

    GP_PATH = "newmove/GP/"
    NAIP_PATH = "newmove/SAT/"
    WORLDCOVER_PATH = "newmove/WORLDCOVER/"
    ELV_PATH = "newmove/ELV/"
    JSON_PATH = "newmove/PARSED_JSONS/"
    NEW_IMAGE_SIZE = 1024

    SIGMA_GROUND = 8
    SIGMA_SATELLITE = 16

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train or inference")
    parser.add_argument("-d", "--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    if args.debug:
        DEBUG = True
    if args.mode == "testing":
        train(device="cpu", num_epochs=1, batch_size=8, length=False)

    elif args.mode == "train":
        train(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs=1000, batch_size=8, length=True)
    elif args.mode == "debug":
        debug_labels("cpu")

    # training

    # train()
    # inference()
    # cont_train("checkpoints_new_cont/checkpoint_epoch_9.pt")
