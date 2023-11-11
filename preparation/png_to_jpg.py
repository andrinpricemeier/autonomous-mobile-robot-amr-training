import cv2
import glob
import ntpath

image_folder = "C:\\Users\\toebi\\OneDrive - Hochschule Luzern\\PREN\\Trainingsdaten\\_TODO_Annotate_This"
# Load .png images
images = glob.glob(f"{image_folder}\\*.png")
for image in images:
    filename = ntpath.basename(image)
    image_file = cv2.imread(image)
    without_ext = filename.split('.')[0]

    cv2.imwrite(f"{image_folder}\\{without_ext}.jpg", image_file, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    

