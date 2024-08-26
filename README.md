# SRGAN_Image_resolution Project for DA-526 at IIT Guwahati

## Group No - 2

### Group Members:
- Rahul Agarkar - 234101041
- Keshav Gupta - 234101023
- Shardul Nalode - 234101049
- Vikas Khurendra - 234101055

---

## Information about Dataset

Dataset Source: [MIRFLICKR](http://press.liacs.nl/mirflickr/mirdownload.html)

We use high-resolution original images from the MIRFLICKR dataset and save lower resolution versions for use with SRGAN.

- Original images are resized to 128x128 pixels for high-resolution (HR) images.
- Lower resolution (LR) images are resized to 32x32 pixels.
- Preprocessing is done in the `preprocessing` folder.
- Download the dataset and set the path as specified in `lanczos.py` file.

---

## Steps to Run the Project:

1. Download the project files and dataset.
2. Install all dependencies mentioned in `requirements.txt`.
3. Run the command `streamlit run app.py`.
4. Upload an image and the application will process it.

## For Training:

1. Open `training.ipynb`.
2. Upload the data (preprocessing creates two folders named `HR_images` and `LR_images`).
3. Set paths where necessary.
4. Adjust hyperparameters such as the number of epochs.
5. The trained model (`.h5` file) for the generator will be generated.

## Testing:

1. Use `test.py`, which will take images from the `test_dataset` directory (create this directory).
2. It will generate three files: `psnr.csv`, `ssim.csv`, and average scores.

---
