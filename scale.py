import cv2
import numpy as np
import os, os.path
import random

REAL_DIR = "lip_imgs/"
DRAW_DIR = "lip_imgs_cv2draw/"
DEST_DIR = "pytorch-CycleGAN-and-pix2pix/datasets/lip/"
SCALE = 500
TRA, VAL, TST = 0.8, 0.1, 0.1
count = 0

files = os.listdir(DRAW_DIR)
random.shuffle(files)

TOTAL_NUM_FILE = len(files)
print(TOTAL_NUM_FILE)

for i in files:
    if(i[-3:] == "jpg"):
        real = REAL_DIR + i # "lip_imgs/183114.jpg"
        draw = DRAW_DIR + i # "lip_imgs_cv2draw/183114.jpg"
        img1 = cv2.imread(real)
        img2 = cv2.imread(draw)
        resized1 = cv2.resize(img1, (SCALE, SCALE))
        resized2 = cv2.resize(img2, (SCALE, SCALE))
        vis = np.concatenate([resized1, resized2], axis=1)
        
        # save image to "train/", "test/", "val/"
        if count < TOTAL_NUM_FILE * TRA:
            cv2.imwrite(DEST_DIR + "train/" + i, vis)
        elif count >= TOTAL_NUM_FILE * TRA and count < TOTAL_NUM_FILE * (1-TST):
            cv2.imwrite(DEST_DIR + "test/" + i, vis)
        else:
            cv2.imwrite(DEST_DIR + "val/" + i, vis)
        count += 1
        
print(len(files))
print(len(os.listdir(DEST_DIR + "train/")))
print(len(os.listdir(DEST_DIR + "test/")))
print(len(os.listdir(DEST_DIR + "val/")))