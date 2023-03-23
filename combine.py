import cv2
import matplotlib.pyplot as plt
import os, os.path
import numpy as np

REAL_DIR = "data/birds/real/"
DRAW_DIR = "pytorch-CycleGAN-and-pix2pix/datasets/bird/val/"
DEST_DIR = "pytorch-CycleGAN-and-pix2pix/datasets/bird-draw-real/val/"

files = os.listdir(DRAW_DIR)


for i in files:
    if(i[-3:] == "jpg"):
        real = REAL_DIR + i # "data/birds/real/ca_001.jpg"
        draw = DRAW_DIR + i # "pytorch-CycleGAN-and-pix2pix/datasets/bird/train/ca_001.jpg"
        img1 = cv2.imread(real)
        img2 = cv2.imread(draw)
        vis = np.concatenate([img1, img2], axis=1)
        cv2.imwrite(DEST_DIR + i, vis)
        
print(len(files))
print(len(os.listdir(DEST_DIR)))