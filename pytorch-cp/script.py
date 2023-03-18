import os
import shutil

files = os.listdir(".")

p = set([x[:-4] for x in files])
print(p)

all_pics = os.listdir("../../VOCdevkit/VOC2010/JPEGImages")
print(all_pics[:5])

out_dir = "../pascal_person_part_real/"

for i in all_pics:
    img = i[:-4]
    if img in p:
        shutil.move("../../VOCdevkit/VOC2010/JPEGImages/" + i, out_dir + img + ".png")
