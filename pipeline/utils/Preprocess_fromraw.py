# here, we just put a black circular mask on the experimental images to crop out the plate boundaries
# no additional rotation or cropping 
import cv2
import numpy as np
import os
from Data_augmentation.utils.local_config import MUTANT_EXP_FOLDER

def crop_experimental(img):
    """Optimized experimental image rotation using OpenCV warpAffine. Takes raw experimental images with plate boundaries 
    Steps:
    - detect plate circle with HoughCircles
    - mask and adjust contrast/brightness
    - re-apply circular mask and return
    """

    # similar code as some of the preprocessing functions
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find edges in the image using Canny edge detection
    edges = cv2.Canny(img_gray, 100, 200)

    # Find circles in the image using HoughCircles method
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=20, maxRadius=600)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Take the first circle found
        # Assuming first circle detected by Hough transform is the plate
        x_center, y_center, radius = circles[0]
        new_radius = radius - 82

        # Create a mask where all values are set to zero (black)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.circle(mask, (int(x_center), int(y_center)), int(new_radius), 255, -1)
        img_masked = cv2.bitwise_and(img, img, mask=mask)
        
        # Adjust contrast and brightness
        alpha = 1.5
        beta = 50
        adjusted_image = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)
        # no rotation here, just return cropped image
        output_img =cv2.bitwise_and(adjusted_image, adjusted_image, mask=mask)
    else:
        # no circle found -> return empty image of same shape
        output_img = np.zeros_like(img)

    return output_img
 

# Define the folders
experimental_folder = MUTANT_EXP_FOLDER

# save by adding _AUG_DUP to the EXPERIMENTAL AND SIMULATED folder names
# extract the base paths and append new folder names
new_experimental_folder = experimental_folder + '_preprocess_noaug'

# Ensure the new folders exist
os.makedirs(new_experimental_folder, exist_ok=True)

# Get the file lists, only files, no directories
experimental_files = [f for f in sorted(os.listdir(experimental_folder)) 
                      if not os.path.isdir(os.path.join(experimental_folder, f))]



# Process each pair of images
for filename in sorted(experimental_files):
    
    print(filename)
    
    # Read the images ONCE per file (moved outside the angle loop to avoid redundant loading)
    exp_img = cv2.imread(os.path.join(experimental_folder, filename))

    # additional edit made for Dongheon's 2000x2000 images to resize to 1001x1001 first so that HoughCircles can work
    if exp_img.shape[0] > 1001 or exp_img.shape[1] > 1001:
        exp_img= cv2.resize(exp_img, (1001,1001))

    # Crop the experimental image
    cropped_exp_img = crop_experimental(exp_img)

    # Save the new images in the new folders with the angle in the filename
    new_exp_filename = f"{os.path.splitext(filename)[0]}{os.path.splitext(filename)[1]}"

    cv2.imwrite(os.path.join(new_experimental_folder, new_exp_filename), cropped_exp_img)

    # print("Processed and wrote first file")
    # break

print("Processing completed.")
