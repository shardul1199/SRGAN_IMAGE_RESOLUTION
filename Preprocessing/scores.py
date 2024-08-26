import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os

# Define the paths to the folders containing images
original_folder = 'original'
bilinear_folder = 'bilinear'
bicubic_folder = 'bicubic'
lanczos_folder = 'lanczos'

# Initialize variables to accumulate scores
psnr_scores_bilinear = []
psnr_scores_bicubic = []
psnr_scores_lanczos = []

ssim_scores_bilinear = []
ssim_scores_bicubic = []
ssim_scores_lanczos = []

mse_scores_bilinear = []
mse_scores_bicubic = []
mse_scores_lanczos = []

# Get the list of file names in the original folder
file_names = os.listdir(original_folder)

# Iterate over the file names
for file_name in file_names:
    # Load the original image
    original_image = cv2.imread(os.path.join(original_folder, file_name), cv2.IMREAD_COLOR)

    # Load the downscaled images produced by different methods
    downscaled_image_bilinear = cv2.imread(os.path.join(bilinear_folder, file_name), cv2.IMREAD_COLOR)
    downscaled_image_bicubic = cv2.imread(os.path.join(bicubic_folder, file_name), cv2.IMREAD_COLOR)
    downscaled_image_lanczos = cv2.imread(os.path.join(lanczos_folder, file_name), cv2.IMREAD_COLOR)

    # Convert the images to grayscale (if necessary)
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    downscaled_gray_bilinear = cv2.cvtColor(downscaled_image_bilinear, cv2.COLOR_BGR2GRAY)
    downscaled_gray_bicubic = cv2.cvtColor(downscaled_image_bicubic, cv2.COLOR_BGR2GRAY)
    downscaled_gray_lanczos = cv2.cvtColor(downscaled_image_lanczos, cv2.COLOR_BGR2GRAY)

    # Compute PSNR
    psnr_scores_bilinear.append(psnr(original_gray, downscaled_gray_bilinear))
    psnr_scores_bicubic.append(psnr(original_gray, downscaled_gray_bicubic))
    psnr_scores_lanczos.append(psnr(original_gray, downscaled_gray_lanczos))

    # Compute SSIM
    ssim_scores_bilinear.append(ssim(original_gray, downscaled_gray_bilinear))
    ssim_scores_bicubic.append(ssim(original_gray, downscaled_gray_bicubic))
    ssim_scores_lanczos.append(ssim(original_gray, downscaled_gray_lanczos))

    # Compute MSE
    mse_scores_bilinear.append(np.mean((original_gray - downscaled_gray_bilinear) ** 2))
    mse_scores_bicubic.append(np.mean((original_gray - downscaled_gray_bicubic) ** 2))
    mse_scores_lanczos.append(np.mean((original_gray - downscaled_gray_lanczos) ** 2))

# Calculate average scores
avg_psnr_bilinear = np.mean(psnr_scores_bilinear)
avg_psnr_bicubic = np.mean(psnr_scores_bicubic)
avg_psnr_lanczos = np.mean(psnr_scores_lanczos)

avg_ssim_bilinear = np.mean(ssim_scores_bilinear)
avg_ssim_bicubic = np.mean(ssim_scores_bicubic)
avg_ssim_lanczos = np.mean(ssim_scores_lanczos)

avg_mse_bilinear = np.mean(mse_scores_bilinear)
avg_mse_bicubic = np.mean(mse_scores_bicubic)
avg_mse_lanczos = np.mean(mse_scores_lanczos)

# Print the average scores
print("Average PSNR (Bilinear):", avg_psnr_bilinear)
print("Average PSNR (Bicubic):", avg_psnr_bicubic)
print("Average PSNR (Lanczos):", avg_psnr_lanczos)
print("\n")
print("Average SSIM (Bilinear):", avg_ssim_bilinear)
print("Average SSIM (Bicubic):", avg_ssim_bicubic)
print("Average SSIM (Lanczos):", avg_ssim_lanczos)
print("\n")
print("Average MSE (Bilinear):", avg_mse_bilinear)
print("Average MSE (Bicubic):", avg_mse_bicubic)
print("Average MSE (Lanczos):", avg_mse_lanczos)

