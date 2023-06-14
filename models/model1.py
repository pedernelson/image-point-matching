import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

import helpers

"""

(x1_g1, y1_g1) and (x1_s1, y1_s1) are the same object in real life

ground_keypoints = [
    [(x1_g1, y1_g1), (x2_g1, y2_g1), (x3_g1, y3_g1), (x4_g1, y4_g1), (x5_g1, y5_g1)],
    [(x1_g2, y1_g2), (x2_g2, y2_g2), (x3_g2, y3_g2), (x4_g2, y4_g2), (x5_g2, y5_g2)],
    [(x1_g3, y1_g3), (x2_g3, y2_g3), (x3_g3, y3_g3), (x4_g3, y4_g3), (x5_g3, y5_g3)]
]

satellite_keypoints = [
    [(x1_s1, y1_s1), (x2_s1, y2_s1), (x3_s1, y3_s1), (x4_s1, y4_s1), (x5_s1, y5_s1)],
    [(x1_s2, y1_s2), (x2_s2, y2_s2), (x3_s2, y3_s2), (x4_s2, y4_s2), (x5_s2, y5_s2)],
    [(x1_s3, y1_s3), (x2_s3, y2_s3), (x3_s3, y3_s3), (x4_s3, y4_s3), (x5_s3, y5_s3)]
]


Siamse Network:
1. The model takes N images as input, where N is the number of images in the dataset.
   For each image, the model extracts features using a pre-trained deep learning model, (tandem training)
2. The model then uses a Siamese network to learn the similarities between the features extracted from the images.
3. The model outputs the coordinates of the keypoints in each image. Where the coordinates are the same for the same object in real life.

Data preparation:

Collect and preprocess the dataset, including ground photo images, NAIP satellite images, and Google WorldCover land cover satellite images. Align and scale the images as needed.
Label the keypoints in the images (e.g., building corners, unique trees, road corners, bridges) and their corresponding coordinates.
Split the dataset into training, validation, and test sets.
Model architecture:

Use a pre-trained deep learning model, such as a ResNet or VGG model, as the backbone for feature extraction. You can use these models in combination with Siamese or Triplet networks for learning similarities between keypoints.
Create three input branches, one for each type of image (ground photo, NAIP satellite, and Google WorldCover). Each branch will process the corresponding image type and extract relevant features.
Fuse the extracted features using concatenation or other techniques, and add a few fully connected layers to process the combined features.
Use a final output layer to predict the coordinates of the keypoints in each image.
Loss function and optimization:

Define an appropriate loss function to minimize the difference between predicted and ground truth keypoints, such as Mean Squared Error (MSE) or Huber loss.
Choose an optimizer, like Adam or SGD, to minimize the loss.
Model training:

Train the model using the training dataset, and monitor its performance on the validation dataset.
Use techniques such as learning rate scheduling, early stopping, and data augmentation to improve the model's generalization capabilities.
Model evaluation:

Evaluate the model's performance on the test dataset using appropriate metrics, like Euclidean distance or Mean Absolute Error (MAE) between predicted and ground truth keypoints.
Post-processing:

For each image pair, identify the keypoints and their corresponding coordinates in each image.
Filter the keypoints by finding those that correspond to the same object in real life, using a threshold based on the distance metric.

https://github.com/spgriffin/land_cover/blob/master/notebooks/Step%203.%20Train_U_Net.ipynb

"""

def contrastive_loss(ground_preds, satellite_preds, margin=1.0):
    """To further enforce the consistency between the predicted keypoints
       in the ground and satellite images, you can modify the loss function
       to include a term that encourages the model to learn the correspondences
       better. One way to do this is to use a contrastive loss or a triplet loss
       to minimize the distance between the corresponding keypoints and maximize
       the distance between non-corresponding keypoints.

    Args:
        ground_preds (_type_): _description_
        satellite_preds (_type_): _description_
        margin (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    distances = [torch.norm(g - s, dim=1) for g, s in zip(ground_preds, satellite_preds)]
    positive_loss = sum([d.pow(2).mean() for d in distances])
    negative_loss = sum([(margin - d).clamp(min=0).pow(2).mean() for d in distances])

    return positive_loss + negative_loss


# a data class for paired image keypoints
class PairedImagesDataset(Dataset):
    def __init__(self, ground_photos, satellite_images, ground_keypoints, satellite_keypoints, output_size, transform=None):
        self.ground_photos = ground_photos
        self.satellite_images = satellite_images
        self.ground_keypoints = ground_keypoints
        self.satellite_keypoints = satellite_keypoints
        self.output_size = output_size
        self.transform = transform

    def __len__(self):
        return len(self.ground_photos)

    def __getitem__(self, idx):
        ground_photo = self.ground_photos[idx]
        satellite_image = self.satellite_images[idx]
        ground_keypoints = self.ground_keypoints[idx]
        satellite_keypoints = self.satellite_keypoints[idx]

        if self.transform:
            ground_photo = self.transform(ground_photo)
            satellite_image = self.transform(satellite_image)

        ground_heatmaps = keypoints_to_heatmaps(ground_keypoints, self.output_size[0], self.output_size[1])
        satellite_heatmaps = keypoints_to_heatmaps(satellite_keypoints, self.output_size[0], self.output_size[1])

        return ground_photo, satellite_image, torch.tensor(ground_heatmaps), torch.tensor(satellite_heatmaps)


class FPN(nn.Module):
    def __init__(self, backbone):
        super(FPN, self).__init__()
        self.backbone = backbone
        self.lat_layers = nn.ModuleList([nn.Conv2d(c, 256, 1) for c in self.backbone.out_channels])
        self.pred_layers = nn.ModuleList([nn.Conv2d(256, 256, 3, padding=1) for _ in range(len(self.backbone.out_channels))])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        features = self.backbone(x)
        latents = [lat_layer(f) for lat_layer, f in zip(self.lat_layers, features)]

        outputs = []
        prev_latent = latents[-1]
        for lat, pred in zip(reversed(latents[:-1]), reversed(self.pred_layers[:-1])):
            prev_latent = self.upsample(prev_latent)
            lat_sum = prev_latent + lat
            outputs.append(pred(lat_sum))
            prev_latent = lat_sum

        outputs.append(self.pred_layers[-1](latents[-1]))
        return list(reversed(outputs))

class PointCorrespondenceModel(nn.Module):
    def __init__(self, num_keypoints):
        super(PointCorrespondenceModel, self).__init__()
        self.backbone = models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-3])
        self.backbone.out_channels = [64, 128, 256, 512]
        self.fpn = FPN(self.backbone)
        self.prediction_layer = nn.Conv2d(256, num_keypoints, 1)

    def forward(self, ground_photo, satellite_image):
        ground_features = self.fpn(ground_photo)
        satellite_features = self.fpn(satellite_image)

        ground_predictions = [self.prediction_layer(f) for f in ground_features]
        satellite_predictions = [self.prediction_layer(f) for f in satellite_features]

        return ground_predictions, satellite_predictions


# training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_keypoints = len(ground_keypoints[0])
model = PointCorrespondenceModel(num_keypoints).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
alpha = 0.1  # Weight for the contrastive loss term

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for ground_photo, satellite_image, ground_heatmaps, satellite_heatmaps in train_loader:
        ground_photo, satellite_image = ground_photo.to(device), satellite_image.to(device)
        ground_heatmaps, satellite_heatmaps = ground_heatmaps.to(device), satellite_heatmaps.to(device)

        optimizer.zero_grad()

        ground_predictions, satellite_predictions = model(ground_photo, satellite_image)
        heatmap_loss = sum([criterion(ground_pred, ground_heatmaps) + criterion(satellite_pred, satellite_heatmaps)
                            for ground_pred, satellite_pred in zip(ground_predictions, satellite_predictions)])

        corr_loss = contrastive_loss(ground_predictions, satellite_predictions)
        total_loss = heatmap_loss + alpha * corr_loss

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print("Training complete!")