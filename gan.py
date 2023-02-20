import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

############################# Discriminator Model #############################

class Block(nn.Module):
    """
        Discriminator Block 4x4 convolution-InstanceNorm-LeakyReLU
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels, 
                kernel_size=4, 
                stride=stride, 
                padding=1, 
                bias=True, 
                padding_mode="reflect"
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    """
        Discriminator Architecture is C64-C128-C256-C512
    """ 
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                Block(
                    in_channels=in_channels,
                    out_channels=feature,
                    stride=1 if feature == features[-1] else 2
                )
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect"
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))

def discriminator_test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator()
    preds = model(x)
    print(preds.shape)

############################### Generator Model ###############################

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels, 
                 down=True, 
                 use_act=True, 
                 **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                padding_mode="reflect",
                **kwargs
            ) if down
            else nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )
    
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1
            ),
            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                use_act=False,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=num_features,
                    out_channels=num_features*2,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                ConvBlock(
                    in_channels=num_features*2,
                    out_channels=num_features*4,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=num_features*4,
                    out_channels=num_features*2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    in_channels=num_features*2,
                    out_channels=num_features*1,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )
        self.last = nn.Conv2d(
            in_channels=num_features*1,
            out_channels=img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )
    
    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

def generator_test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)

################################### Dataset ###################################

class HorseZebraDataset(Dataset):
    def __init__(self, root_z, root_h):
        super().__init__()
        self.root_z = root_z
        self.root_h = root_h

        self.zebra_images = os.listdir(root_z)
        self.horse_images = os.listdir(root_h)
        self.length_dataset = max(len(self.zebra_images), len(self.horse_images))
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        zebra_img = self.zebra_images[index % self.zebra_len]
        horse_img = self.horse_images[index % self.horse_len]

        zebra_path = os.path.join(self.root_z, zebra_img)
        horse_path = os.path.join(self.root_h, horse_img)

        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        return zebra_img, horse_img

#################################### Train ####################################

def train_fn(
    disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loader):
        zebra = zebra
        horse = horse
        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * 10
                + cycle_horse_loss * 10
                + identity_horse_loss * 0.0
                + identity_zebra_loss * 0.0
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_horse * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))




def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    dis_H = Discriminator(in_channels=3)
    dis_Z = Discriminator(in_channels=3)
    gen_Z = Generator(img_channels=3, num_residuals=9)
    gen_H = Generator(img_channels=3, num_residuals=9)

    opt_dis = optim.Adam(list(dis_H.parameters())+list(dis_Z.parameters()), lr=1e-5, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(list(gen_Z.parameters())+list(gen_H.parameters()), lr=1e-5, betas=(0.5, 0.999),)

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    train_path = "data/horse2zebra"

    dataset = HorseZebraDataset(
        root_h=train_path + "/trainA",
        root_z=train_path + "/trainB",
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(10):
        train_fn(
            dis_H,
            dis_Z,
            gen_Z,
            gen_H,
            loader,
            opt_dis,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )



if __name__ == "__main__":
    # discriminator_test()
    # generator_test()
    main()