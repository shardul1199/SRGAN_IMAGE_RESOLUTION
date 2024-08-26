import cv2
import csv
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from keras.models import load_model

generator = load_model('gen_e26_3.h5', compile=False)

directory = "testing_accuracy_data"

def main():
    # File for PSNR scores
    with open('psnr_scores.csv', mode='w', newline='') as psnr_file:
        psnr_writer = csv.writer(psnr_file)
   
        
        # File for SSIM scores
        with open('ssim_scores.csv', mode='w', newline='') as ssim_file:
            ssim_writer = csv.writer(ssim_file)
            
            
            # Processing each image in the directory
            for file in os.listdir(directory):
                original_image = cv2.imread(os.path.join(directory, file), cv2.IMREAD_COLOR)
                
                img_lr = cv2.resize(original_image, (32, 32), interpolation=cv2.INTER_LANCZOS4)
                img_lr = img_lr / 255.0
                img_lr = np.expand_dims(img_lr, axis=0)
                
                generated_img = generator.predict(img_lr)
                generated_img = np.clip(generated_img, 0.0, 1.0)

                # Save the generated image
                generated_img = (generated_img[0] * 255).astype(np.uint8)  # Convert back to uint8
                cv2.imwrite("generated_image.jpg", generated_img)

                sr_generated_img = cv2.imread('generated_image.jpg', cv2.IMREAD_COLOR)
                sr_generated_img = cv2.resize(sr_generated_img, (original_image.shape[1], original_image.shape[0]))

                original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                sr_generated_gray = cv2.cvtColor(sr_generated_img, cv2.COLOR_BGR2GRAY)

                psnr_score = psnr(original_gray, sr_generated_gray)
                ssim_score = ssim(original_gray, sr_generated_gray)

                # Write PSNR scores to psnr_scores.csv
                psnr_writer.writerow([psnr_score])

                # Write SSIM scores to ssim_scores.csv
                ssim_writer.writerow([ssim_score])



    psnr_scores = []
    with open('psnr_scores.csv', mode='r') as psnr_file:
        reader = csv.reader(psnr_file)
        next(reader)  # Skip header row
        for row in reader:
            psnr_scores.append(float(row[0]))

    # Read SSIM scores from ssim_scores.csv
    ssim_scores = []
    with open('ssim_scores.csv', mode='r') as ssim_file:
        reader = csv.reader(ssim_file)
        next(reader)  # Skip header row
        for row in reader:
            ssim_scores.append(float(row[0]))

    # Calculate average scores
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_ssim = sum(ssim_scores) / len(ssim_scores)

    # Write average scores to a new text file
    with open('average_scores.txt', mode='w') as avg_file:
        avg_file.write(f'Average PSNR: {avg_psnr}\n')
        avg_file.write(f'Average SSIM: {avg_ssim}\n')

if __name__ == "__main__":
    main()
