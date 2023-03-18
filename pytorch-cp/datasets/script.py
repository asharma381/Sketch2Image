import os, os.path
import random
import shutil

files = os.listdir(".")
random.shuffle(files)
TOTAL_NUM_FILE = len(files)
print(TOTAL_NUM_FILE)
TRA, VAL, TST = 0.8, 0.1, 0.1
DEST_DIR = "../../../pytorch-CycleGAN-and-pix2pix/datasets/person/"
count = 0
for f in files:
    if count < TOTAL_NUM_FILE * TRA:
        shutil.move(f, DEST_DIR + "train/" + f)
    elif count >= TOTAL_NUM_FILE * TST and  count < TOTAL_NUM_FILE * (1-TST):
        shutil.move(f, DEST_DIR + "test/" + f)
    else:
        shutil.move(f, DEST_DIR + "val/" + f)
    count+=1
    