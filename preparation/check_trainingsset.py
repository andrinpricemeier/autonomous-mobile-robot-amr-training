import glob
import ntpath
import os

image_folder = "C:\\Users\Andrin\\OneDrive - Hochschule Luzern\\PREN\\Trainingsdaten\\1_Originaldaten_Trainingsset_und_Validierungsset"
images = glob.glob(f"{image_folder}\\*.jpg")
for image in images:
    filename = ntpath.basename(image)
    without_ext = filename.split('.')[0]
    if not os.path.isfile(os.path.join(image_folder, without_ext + ".xml")):
        print("DOESNT EXIST: " + without_ext)
        os.remove(image)