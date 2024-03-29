import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
import numpy as np
import pandas as pd
import h5py
import os

class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x

class load_dataset(Dataset):

    def __init__(self, hdf5_fold, file_name_list):
        self.file_name = file_name_list
        self.wave_dir = hdf5_fold
        
    def __len__(self):
        self.numbers = len(self.file_name)

        return self.numbers

    def __getitem__(self, idx):
        wave_path = os.path.join(self.wave_dir, self.file_name[idx])
        waveform = h5py.File(wave_path, 'r')
        waveform = torch.tensor(waveform['waveform_data'][:])
        label = self.file_name[idx]

        return waveform, label

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Down, self).__init__()
        self.downblock = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), 
            nn.BatchNorm1d(out_channels), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.downblock(x)

class Up(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, r):
        super(Up, self).__init__()

        self.sub_pixel = PixelShuffle1D(r)

        self.upblock = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding), 
            nn.BatchNorm1d(out_channels), 
            nn.ReLU(inplace=True), 
        )

    def forward(self, x1, x2):
        print("x1first shape:", x1.shape)
        x1 = self.sub_pixel(x1)
        # input is CL
        diff = torch.tensor([x2.size()[2] - x1.size()[2]])
        x1 = nn.functional.pad(x1, (diff // 2, diff - diff // 2))
        print("x2 shape:", x2.shape)
        print("x1 shape:", x1.shape)
        x = torch.cat([x2, x1], dim=1)

        return self.upblock(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class CBLR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBLR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), 
            nn.BatchNorm1d(out_channels), 
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
##################################### 測試區 ###########################################

# file_name = os.listdir('/home/yuheng5454/MiDAS_test/data/hdf5/DAS')
# fold = '/home/yuheng5454/MiDAS_test/data/hdf5/DAS'
# a= load_dataset(fold, file_name)
# train_dataload = DataLoader(a, batch_size=32, shuffle=True)

# b, c = next(iter(train_dataload))




# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super().__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes

#         self.down1 = Down(n_channels, 32, 5, 1, 0) #len=11997
#         self.down2 = Down(32, 64, 5, 2, 0) #len=5997

#         self.up1 = Up(64, 16, 5, 1, 0, 2) 
#         self.outc = OutConv(16, n_classes)

#     def forward(self, x):
#         x1 = self.down1(x)
#         x2 = self.down2(x1)
#         x = self.up1(x2, x1)
#         final = self.outc(x)

#         return final

# net = UNet(n_channels=1, n_classes=1)
# b = b.unsqueeze(1)
# b.size()

# pred = net(b)
# pred[0].size()
# test = np.array(pred[1].detach())

# import matplotlib.pyplot as plt

# for i in range(20):
#     test = np.array(pred[i].detach())
#     fig, ax = plt.subplots(figsize=(20, 10), dpi=500, nrows=2)
#     ax[0].plot(np.arange(0,11993), test[0], linewidth=0.2, color='black')
#     ax[1].plot(np.arange(0,12001), b[i][0], linewidth=0.2, color='black')
#     fig.show()

# class Discriminator(nn.Module):
#     def __init__(self, n_channels):
#         super(Discriminator, self).__init__()
#         self.conv = nn.Sequential(
#             CBLR(n_channels, 64, 9, 2, 0), 
#             CBLR(64, 64, 9, 2, 0), 
#             CBLR(64, 64, 9, 2, 0), 
#             CBLR(64, 64, 9, 2, 0), 
#             CBLR(64, 64, 9, 2, 0), 
#             CBLR(64, 64, 9, 2, 0), 

#             CBLR(64, 1, 180, 1, 0), 
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         output = self.conv(x)
#         return output

# test = Discriminator(1)
# pred = test(b)
# pred