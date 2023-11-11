#this file 1)compare the 
# images : "./yolo5_cer/train/images"
# and
# labels ="./yolo5_cer/train/labels"
# by file names. 
# Will remove  files in images dir which are not exist in labels dir

import os

images = os.listdir("./yolo5_cer/train/images")
labels = os.listdir("./yolo5_cer/train/labels")
images_name=[]
labels_name=[]
for f in images:
    name, ext = f.split(".")
    images_name.append(name)
for f in labels:
    name, ext = f.split(".")
    labels_name.append(name)
print("images_name ====")    
print(images_name)
for f in images_name:
    try:
        labels_name.index(f)  
        print("found label: "+ f)
    except ValueError:
        print("Not found  : "+ f)
        file ="./yolo5_cer/train/images/"+ f+".jpg"
        print("deleteed "+ file)
        os.remove(file)