# %%

import os

# %%
imgs = os.listdir("JPEGImages")
masks = os.listdir("pascal_person_part_gt")

print(len(imgs))
print(len(masks))

# %%
to_delete = []

fixed_masks = set([mask.split(".")[0] for mask in masks])
fixed_imgs = [img.split(".")[0] for img in imgs]
print(masks[0])


print(len(masks))
print(len(fixed_masks))

print("2008_004677.png" in fixed_masks)


# %%
to_delete = []
print(fixed_imgs[0])
print(imgs[0])
for img in fixed_imgs:
    if img not in fixed_masks:
        to_delete.append(img)
# %%

print(len(to_delete))
print(to_delete[0])
# %%

for to_del in to_delete:
    os.remove("JPEGImages/" + to_del + ".jpg")
# %%


import numpy as np
import cv2
from matplotlib import pyplot as plt

# %%
# mask = cv2.imread('imgs/2010_006051.jpg')
# print(img.shape)
# img_rgb = img[:, :, ::-1]
# plt.imshow(img_rgb)
import scipy.misc
from PIL import Image

head = [1,2,4,13]
upper = [3,5,6,7,10,11,14,15]
lower = [6,9,10,12,16,17]
feet = [8,18,19]
img = Image.open('lip_imgs/1276362.jpg')

x1 = 75
y1 = 100
tw = 100
th = 50

from torchvision.transforms.functional import hflip


img = hflip(img)

img.show()


print(tuple(0.5 for i in range(3)))


# %%
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
# import os
# print(os.getcwd())
thing = torch.load("aditya/cyclegan/pytorch-cp/TEST_THING")

image_numpy = thing.cpu().numpy()[0,:,:,:]
image_numpy = image_numpy.transpose([1,2,0])[:,:,:]
# print(image_numpy.shape)
plt.imsave("Name1.png", image_numpy[:,:,4])
# np.uint8(mat * 255)
# img = Image.fromarray(np.uint8(image_numpy[:,:,0] * 255), mode="L")
# img.show()


# mask = cv2.cvtColor(cv2.imread('lip_masks_gray/1276362.png'), cv2.COLOR_BGR2GRAY)
# print(np.unique(mask))
# new_shape = [mask.shape[0], mask.shape[1], 7]



# new = np.zeros(new_shape)
# new[:,:,0] = np.isin(mask, head)
# new[:,:,1] = np.isin(mask, upper)
# new[:,:,2] = np.isin(mask, lower)
# new[:,:,3] = np.isin(mask, feet)
# new[:,:,4:] = img

# plt.imshow(new[:,:,4])
# # scipy.misc.imsave("lip_masks_merged/test.png", new)

# print(np.unique(np.isin(mask, upper)))
# # plt.imshow(img_rgb)
# plt.imshow(mask)
# new = (mask != 0)
# img_rgb[new] = 255
# plt.imshow(img_rgb)
# plt.show()

# %%

img = cv2.imread("lip_masks_merged/test.png")
plt.imshow(img[:,:,3])
# %%

cv2.destroyAllWindows()

# %%

def make_edges(name):
    img_loc = f"aditya/cyclegan/control_imgs/{name}.jpg"
    mask_loc = f"lip_masks_gray/{name}.png"
    print(img_loc)
    
    img = cv2.imread(img_loc)[:, :, ::-1]
    mask = cv2.imread(mask_loc)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    print(img.shape, mask.shape)
    foreground_idx = (mask != 0)
    background_idx = (mask == 0)

    foreground = img.copy()
    foreground[background_idx] = 255
    background = img.copy()
    background[foreground_idx] = 255

    all_edges = cv2.Canny(img, 70, 150, apertureSize=3)
    foreground_edges = cv2.Canny(foreground, 70, 150, apertureSize=3)
    background_edges = cv2.Canny(background, 100, 200, apertureSize=3)

    # plt.imshow(img)
    # plt.show()
    # plt.imshow(255 - all_edges, cmap="gray")
    # plt.show()
    plt.imshow(background, cmap="gray")
    plt.show()
    plt.imshow(background_edges, cmap="gray")
    plt.show()

    total = all_edges
    total[(total == 254)] = 255
    # print(np.unique(total))
    # cv2.imwrite(f"lip_cc/{name}.png", 255 - total)
    plt.imshow(255 - total, cmap="gray")
    plt.show()
    
    # print(mask[50,60])
    # print(np.unique(foreground_edges + background_edges))
    # print(np.unique(all_edges))


def make_drawing(name):
    img_loc = f"aditya/cyclegan/control_imgs/{name}.jpg"
    mask_loc = f"lip_masks/{name}.png"

    image = cv2.imread(img_loc)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv_img = 255 - gray_img
    blurred = cv2.GaussianBlur(inv_img, (21,21), 0)
    inv_blur = 255 - blurred
    pencil_sketch = cv2.divide(gray_img, inv_blur, scale=256.0)
    cv2.imshow(cv2.cvtColor(pencil_sketch, cv2.COLOR_BGR2RGB))
    # cv2.imwrite(f"lip_draw1/{name}.png", cv2.cvtColor(pencil_sketch, cv2.COLOR_BGR2RGB))
    

    
# os.listdir("lip_imgs")[:5] + 
for name in ["1276362.jpg"]:
    make_edges(name.split(".")[0])
    


# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
image = cv2.imread("lip_imgs/183114.jpg")
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
inv_img = 255 - gray_img
blurred = cv2.GaussianBlur(inv_img, (21,21), 0)
inv_blur = 255 - blurred
pencil_sketch = cv2.divide(gray_img, inv_blur, scale=256.0)

blurred1 = cv2.GaussianBlur(gray_img, (21,21), 0)
pencil_sketch1 = cv2.divide(gray_img, blurred1, scale=256.0)


plt.imshow(pencil_sketch, cmap="gray")
print(np.allclose( pencil_sketch, pencil_sketch1, 0.1))
print(np.min(pencil_sketch), np.max(pencil_sketch))

# %%


from tensorbay import GAS
from tensorbay.dataset import Dataset

# Authorize a GAS client.
gas = GAS('ACCESSKEY-a99e7df960ef94c9b211367dc0d6b0b9')

# Get a dataset.
dataset = Dataset("LIP", gas)

# List dataset segments.
segments = dataset.keys()
segment = dataset["val"]


from PIL import Image
# %%

head = [1,2,4,13]
upper = [3,5,6,7,10,11,14,15]
lower = [6,9,10,12,16,17]
feet = [8,18,19]

SEMANTIC_MASK = {
    "categories": [
        {
            "categoryId": 0,
            "description": "",
            "name": "Background"
        },
        {
            "categoryId": 1,
            "description": "",
            "name": "Hat"
        },
        {
            "categoryId": 2,
            "description": "",
            "name": "Hair"
        },
        {
            "categoryId": 3,
            "description": "",
            "name": "Glove"
        },
        {
            "categoryId": 4,
            "description": "",
            "name": "Sunglasses"
        },
        {
            "categoryId": 5,
            "description": "",
            "name": "UpperClothes"
        },
        {
            "categoryId": 6,
            "description": "",
            "name": "Dress"
        },
        {
            "categoryId": 7,
            "description": "",
            "name": "Coat"
        },
        {
            "categoryId": 8,
            "description": "",
            "name": "Socks"
        },
        {
            "categoryId": 9,
            "description": "",
            "name": "Pants"
        },
        {
            "categoryId": 10,
            "description": "",
            "name": "Jumpsuits"
        },
        {
            "categoryId": 11,
            "description": "",
            "name": "Scarf"
        },
        {
            "categoryId": 12,
            "description": "",
            "name": "Skirt"
        },
        {
            "categoryId": 13,
            "description": "",
            "name": "Face"
        },
        {
            "categoryId": 14,
            "description": "",
            "name": "Left-arm"
        },
        {
            "categoryId": 15,
            "description": "",
            "name": "Right-arm"
        },
        {
            "categoryId": 16,
            "description": "",
            "name": "Left-leg"
        },
        {
            "categoryId": 17,
            "description": "",
            "name": "Right-leg"
        },
        {
            "categoryId": 18,
            "description": "",
            "name": "Left-shoe"
        },
        {
            "categoryId": 19,
            "description": "",
            "name": "Right-shoe"
        }
    ],
}



# %%

def good(mask):
    img, num = label(mask)
    uniquel, countl = np.unique(img, return_counts=True)
    if len(countl) > 2 and ((countl[2] / countl[1]) > 0.1):
        # print(f"More than 1 component ({num})")
        # plt.imshow(img)
        # plt.show()
        return False
    unique = set(np.unique(mask))

    if 13 not in unique:
        return False

    if not (8 in unique or (18 in unique and 19 in unique)):
        return False

    
    return True


mask = None
saved = 0
total = len(segment)


# %%
good_masks = os.listdir("lip_masks_gray")

print(len(good_masks))
# import shutil


# for name in good_masks:
#     shutil.copyfile(f"semantic_mask/{name}", f"lip_masks_gray/{name}")


# %%
for i, data in enumerate(segment):
    if i > 5:
        break
    # fp = data.label.semantic_mask.open()
    # image = Image.open(fp)
    # plt.imshow(image)

    mfp = data.label.semantic_mask.open()
    mask = np.array(Image.open(mfp))
    cv2.imwrite(f"lip_masks_gray/{data.label.semantic_mask.path}", mask)



    # with data.open() as fp:
    #     mfp = data.label.semantic_mask.open()
    #     mask = np.array(Image.open(mfp))
    #     print(mask.shape)
    #     image = np.array(Image.open(fp))
    #     # if not good(mask):
    #     #     # print("Not good")
    #     #     # plt.imshow(image)
    #     #     # plt.show()

    #     #     continue
    #     # plt.imsave(f"test_imgs/{data.path}", image)
    #     cv2.imwrite(f"lip_masks_gray/{data.label.semantic_mask.path}", mask)
    #     saved += 1

    #     if i % 50 == 0:
    #         print(f"{i}/{total}, saved = {saved}, not saved = {i - saved}")
    #     # plt.imshow(image)
    #     # plt.show()
    #     # plt.colorbar()
    #     # cv2.imwrite(f"lip_imgs/{data.path}", image)
    #     # print(data.label)
        # plt.imshow(mask)
        # plt.colorbar()
        # print(np.unique(mask))
        # print(data.label.dumps())
        # print(vars(data.label.semantic_mask.))

# %%

img = cv2.imread("test_masks/1189583.png")
cv2.imwrite("test_masks/1189583gray.png", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# %%

img = cv2.imread("test_masks/1189583gray.png")

print(np.unique(img))


# %%
from scipy.ndimage import label
plt.imshow(mask)
labeled_array, num_features = label(mask)
print(num_features)
plt.imshow(labeled_array)
print(set(np.unique(labeled_array)))


# %%

img = cv2.imread('imgs/2010_006051.jpg')
print(img.shape)
img_rgb = img[:, :, ::-1]
plt.imshow(img_rgb)
# %%

import os
print(len(os.listdir("lip_imgs")))
print(len(os.listdir("lip_masks")))

# %%

import os
from collections import Counter
import random
random.seed(42)

# %%

# folders = list(filter(lambda x: "lip" in x, os.listdir("aditya/cyclegan/pytorch-cp/results/")))
# print(folders)
# random.shuffle(folders)
# print(folders)

folders = [ 
    'lips_seg', 
    'lips_pix2pix', 
    'lipcv2_pix2pix', 
    'lip_cv2_seg', 
    'lipcnet_pix2pix', 
    'lip_canny_seg', 
    'lip_cnet_seg', 
    'lipcanny_pix2pix'
]

# %%


files = os.listdir("aditya/cyclegan/pytorch-cp/results/lips_seg/test_latest/images")
files = list(filter(lambda x: "fake" in x, files))

selected = random.sample(files, 30)
print(selected)


counter = Counter()

# %%
import cv2
import matplotlib.pyplot as plt
for file in selected[:1]:
    fig, ax = plt.subplots(2, 4)
    for i, folder in enumerate(folders):
        # print(f"aditya/cyclegan/pytorch-cp/results/{folder}/test_latest/images/{file}")
        img = cv2.imread(f"aditya/cyclegan/pytorch-cp/results/{folder}/test_latest/images/{file}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(ax.shape)
        ax[i // 4, i % 4].title.set_text("Image " + str(i))
        ax[i // 4, i % 4].imshow(img)
        ax[i // 4, i % 4].get_xaxis().set_visible(False)
        ax[i // 4, i % 4].get_yaxis().set_visible(False)

    plt.show()
    counter[input("Which was the best?")] += 1

print(counter)


# %%

real_results = {
    7: 1, 
    5: 0,
    2: 11, 
    3: 13, 
    1: 3, 
    0: 1, 
    4: 0,
    6: 1,
}
folders_nice = [ 
    'OpenCV\nSimplify\n+Seg', 
    'OpenCV\nSimplify', 
    'OpenCV', 
    'OpenCV\n+Seg', 
    'Control', 
    'Canny\n+Seg', 
    'Control\n+Seg', 
    'Canny'
]

import seaborn as sns
import matplotlib.pyplot as plt

keys = list(real_results.keys())
vals = [real_results[key] for key in keys]
keys_names = [folders_nice[key] for key in keys]
sns.barplot(x=keys_names, y=vals)
