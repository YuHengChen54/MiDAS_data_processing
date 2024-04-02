import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from gan_parts import load_dataset
from gan_model import Generator, Discriminator
import numpy as np
import pandas as pd
import h5py
import os
import time
from tqdm import tqdm

flag_gpu = 1
# Number of workers for dataloader
workers = 0
# Batch size during training
batch_size = 64
# Number of training epochs
epochs = 1
# Learning rate for optimizers
lr = 0.00001

device = 'cuda:0' if (torch.cuda.is_available() & flag_gpu) else 'cpu'
print('GPU State:', device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

G = Generator(1, 1).to(device)
D = Discriminator(1).to(device)

g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=5, gamma=0.5)
# d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=5, gamma=0.5)

fold = '/home/yuheng5454/MiDAS_test/data/hdf5/GLZ'
file_list = os.listdir(fold)
train_set = load_dataset(fold, file_list)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)

# remove_list = []
# for i in tqdm(range(train_set.numbers)):
#     if train_set[i][0].size()[0] != 120001:
#         remove_list.append(train_set[i][1])

# save_fold_list = ['/home/yuheng5454/MiDAS_test/data/hdf5/DAS',
#                 '/home/yuheng5454/MiDAS_test/data/hdf5/GL1', 
#                 '/home/yuheng5454/MiDAS_test/data/hdf5/GL2', 
#                 '/home/yuheng5454/MiDAS_test/data/hdf5/GLZ'
#                 ]
# for file in remove_list:
#     for fold, comp in zip(save_fold_list, ['DAS', 'GL1', 'GL2', 'GLZ']):
#         if comp == 'DAS':
#             os.remove(f'{fold}/{file[0:3]}{file[7:]}')
#         else:
#             os.remove(f'{fold}/{file[0:4]}{comp}{file[7:]}')

adversarial_loss = nn.BCELoss().to(device)
G.train()
D.train()
loss_g, loss_d = [],[]
start_time= time.time()

for epoch in range(epochs):
    epoch += 1
    total_loss_g, total_loss_d = 0, 0
    count_d = 0
    for i_iter, (waveform, label) in enumerate(train_loader):
        i_iter += 1

        # train generator

        g_optimizer.zero_grad()
        noise = torch.randn(waveform.shape[0], 1, 12001)
        noise = noise.to(device)

        fake_label = torch.ones((waveform.shape[0], 1, 1), dtype=torch.float).to(device)

        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)

        loss_g_value = adversarial_loss(fake_outputs, fake_label)
        loss_g_value.backward()
        g_optimizer.step()
        total_loss_g+=loss_g_value
        loss_g.append(loss_g_value) 

        # train discriminator

        d_optimizer.zero_grad()
        real_inputs = waveform.to(device)
        real_inputs = real_inputs.unsqueeze(1)
        real_label = torch.ones((real_inputs.shape[0], 1, 1), dtype=torch.float).to(device)
        fake_label = torch.zeros((fake_inputs.shape[0], 1, 1), dtype=torch.float).to(device)

        real_loss = adversarial_loss(D(real_inputs),real_label)
        fake_loss = adversarial_loss(D(fake_inputs.detach()),fake_label)
        loss_d_value = (real_loss + fake_loss) / 2
        loss_d_value.backward()

        d_optimizer.step()
        total_loss_d+=loss_d_value
        loss_d.append(loss_d_value)
    
    total_loss_g/=len(train_loader)
    total_loss_d/=len(train_loader)

    print('[Epoch: {}/{}] D_loss: {:.3f} G_loss: {:.3f}'.format(epoch, epochs, total_loss_d.item(), total_loss_g.item()))


print('Training Finished.')
print('Cost Time: {}s'.format(time.time()-start_time))