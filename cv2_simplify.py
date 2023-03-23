import cv2
import matplotlib.pyplot as plt
import os, os.path
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from model_gan import model_gan, immean, imstd

REAL_DIR = "control_imgs/"
DRAW_DIR = "control_simp_test/"

files = os.listdir(REAL_DIR)

use_cuda = torch.cuda.device_count() > 0

# cache  = load_lua( "../../sketch_simplification/model_gan.t7" )

model = model_gan
model.load_state_dict(torch.load("../../sketch_simplification/model_gan.pth"))
model.eval()

# model  = cache.model
# immean = cache.mean
# imstd  = cache.std

for name in ["1276362.jpg"]:
    if "jpg" in name or "png" in name:
        real = REAL_DIR + name # "lip_imgs/183114.jpg"
        draw = DRAW_DIR + name # "lip_imgs_cv2draw/183114.jpg"
        
        # pencil sketch with cv2
        # image = cv2.imread(real)
        # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # inv_img = 255 - gray_img
        # blurred = cv2.GaussianBlur(inv_img, (21,21), 0)
        # inv_blur = 255 - blurred
        # pencil_sketch = cv2.divide(gray_img, inv_blur, scale=256.0)
        # cv2.imwrite(draw, cv2.cvtColor(pencil_sketch, cv2.COLOR_BGR2RGB))

        pencil_sketch = cv2.imread(real)
        
        pencil_sketch = cv2.cvtColor(pencil_sketch, cv2.COLOR_BGR2GRAY)
        pencil_sketch[pencil_sketch > 128] = 256
        pencil_sketch[pencil_sketch < 128] = 0
        cv2.imwrite(DRAW_DIR + "thresholded.jpg", pencil_sketch)
        # cv2.imwrite(DRAW_DIR + "gray_" + name, pencil_sketch)
        print(pencil_sketch.shape)
        data = Image.fromarray(pencil_sketch)
        w, h  = data.size[0], data.size[1]
        pw    = 8-(w%8) if w%8!=0 else 0
        ph    = 8-(h%8) if h%8!=0 else 0
        data  = ((transforms.ToTensor()(data)-immean)/imstd).unsqueeze(0)
        if pw!=0 or ph!=0:
            data = torch.nn.ReplicationPad2d( (0,pw,0,ph) )( data ).data

        if use_cuda:
            pred = model.cuda().forward( data.cuda() ).float()
        else:
            pred = model.forward( data )

        save_image( pred[0], draw)



        
print(len(files))