# Image-Segmentation-With-ResNet101
Segmenting five images with ResNet101

This code was created for a Computer Vision class. All code was run on Google Colab

Steps Taken:
1. Import modules and libraries
2. Load the ResNet101 model wiith pretrained weights and set it to inference mode
3. Load and display five random images found on the internet containing at least three objects belonging 
   to the Resnet101 classes.
4. Transform the images to be of uniform size, center cropped, converted to tensor objects, and normalized
   based on the mean and standard deviation of ImageNet
5. Unsqueeze the images so they can pass through the network
6. Complete the forward pass (i.e. send the inputs through the network)
7. Display the feature maps for each image (I created a helper function to do this)
8. Get the max for each pixel and assign that pixel to a class for each image
9. Generate the segmentation maps (I created a helper function to assign each class of the network a color)
10. Show the image segmentation maps
