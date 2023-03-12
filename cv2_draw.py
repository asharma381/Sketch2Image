import cv2
import matplotlib.pyplot as plt
import os, os.path

REAL_DIR = "lip_imgs/"
DRAW_DIR = "lip_imgs_cv2draw/"

files = os.listdir(REAL_DIR)

for i in files:
    if(i[-3:] == "jpg"):
        real = REAL_DIR + i # "lip_imgs/183114.jpg"
        draw = DRAW_DIR + i # "lip_imgs_cv2draw/183114.jpg"
        
        # pencil sketch with cv2
        image = cv2.imread(real)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv_img = 255 - gray_img
        blurred = cv2.GaussianBlur(inv_img, (21,21), 0)
        inv_blur = 255 - blurred
        pencil_sketch = cv2.divide(gray_img, inv_blur, scale=256.0)
        cv2.imwrite(draw, cv2.cvtColor(pencil_sketch, cv2.COLOR_BGR2RGB))
        
print(len(files))