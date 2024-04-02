import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from gan_parts import Down, Up, OutConv, CBLR

class Generator(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Generator, self).__init__()
        self.conv = Down(n_channels, 16, 5, 1, 0) # b x 16 x 
        self.down1 = Down(16, 32, 5, 2, 0)
        self.down2 = Down(32, 64, 5, 2, 0)
        self.down3 = Down(64, 128, 3, 2, 0)
        self.down4 = Down(128, 128, 3, 2, 0)

        self.up1 = Up(192, 128, 3, 1, 0, 2)
        self.up2 = Up(128, 64, 5, 1, 0, 2)
        self.up3 = Up(64, 32, 5, 1, 0, 2)
        self.up4 = Up(32, 16, 5, 1, 0, 2)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)

        return output

if __name__ == '__main__':
    net = Generator(n_channels=1, n_classes=1)
    print(net)

    
class Discriminator(nn.Module):
    def __init__(self, n_channels):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            CBLR(n_channels, 64, 9, 2, 0), 
            CBLR(64, 64, 9, 2, 0), 
            CBLR(64, 64, 9, 2, 0), 
            CBLR(64, 64, 9, 2, 0), 
            CBLR(64, 64, 9, 2, 0), 
            CBLR(64, 64, 9, 2, 0), 

            CBLR(64, 1, 180, 1, 0), 
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.conv(x)
        return output
    
if __name__ == '__main__':
    net2 = Discriminator(n_channels=1)
    print(net2)

# ##################################### 測試區 ###########################################
# import os
# from gan_parts import load_dataset

# file_name = os.listdir('/home/yuheng5454/MiDAS_test/data/hdf5/DAS')
# fold = '/home/yuheng5454/MiDAS_test/data/hdf5/DAS'
# a= load_dataset(fold, file_name)
# train_dataload = DataLoader(a, batch_size=32, shuffle=True)
# b, c = next(iter(train_dataload))
# b = b.unsqueeze(1)

# pred = net(b)
# pred

# import matplotlib.pyplot as plt
# import numpy as np

# for i in range(20):
#     test = np.array(pred[i].detach())
#     fig, ax = plt.subplots(figsize=(20, 10), dpi=500, nrows=2)
#     ax[0].plot(np.arange(0,11993), test[0], linewidth=0.2, color='black')
#     ax[1].plot(np.arange(0,12001), b[i][0], linewidth=0.2, color='black')
#     fig.show()

# test = net2(pred)
# test.size()