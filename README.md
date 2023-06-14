# Image Point Matching


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

