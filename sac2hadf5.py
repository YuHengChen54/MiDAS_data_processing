import h5py
from obspy import read
import os
from tqdm import tqdm

class read_sac():

    def __init__(self, file_path):
        self.sac = read(file_path)
        self.data = self.sac[0].data
        self.time = self.sac[0].times()

    def data(self):
        return self.data
    
    def time(self):
        return self.time

class save_hdf5():

    def __init__(self, save_path, data, time):
        self.hdf5 = h5py.File(save_path, "w")
        self.d1 = self.hdf5.create_dataset('waveform_data', data=data)
        self.d2 = self.hdf5.create_dataset('waveform_time', data=time)
        self.hdf5.close()

fold_list = ['/home/yuheng5454/MiDAS_test/data/checked/1257', 
            '/home/yuheng5454/MiDAS_test/data/checked/GL1', 
            '/home/yuheng5454/MiDAS_test/data/checked/GL2', 
            '/home/yuheng5454/MiDAS_test/data/checked/GLZ'
            ]
save_fold_list = ['/home/yuheng5454/MiDAS_test/data/hdf5/DAS',
                '/home/yuheng5454/MiDAS_test/data/hdf5/GL1', 
                '/home/yuheng5454/MiDAS_test/data/hdf5/GL2', 
                '/home/yuheng5454/MiDAS_test/data/hdf5/GLZ'
                ]
#寫for把所有DAS+井下地震儀改存hdf5
for fold, save_fold in zip(fold_list, save_fold_list):
    file_list = os.listdir(fold)

    if fold[-4:] == '1257':
        for file in tqdm(file_list):
            # print(fold+'/'+file)
            readsac = read_sac(fold+'/'+file)
            data = readsac.data
            time = readsac.time

            save = save_hdf5(f'{save_fold}/{file[:-4]}.hdf5', data, time)

    else:
        for file in tqdm(file_list):
            readsac = read_sac(f'{fold}/{file}')
            data = readsac.data
            time = readsac.time

            if len(data) > 119930:
                data_split = []
                time_split = []
                for i in range(11993):
                    num = i*10
                    data_split.append(data[num])
                    time_split.append(time[num])
                data = data_split
                time = time_split

                save = save_hdf5(f'{save_fold}/{file[:-4]}.hdf5', data, time)
            else:
                continue
# for fold, save_fold in zip(fold_list[1:], save_fold_list[1:]):
#     file_list = os.listdir(fold)
#     for file in tqdm(file_list):
#         readsac = read_sac(f'{fold}/{file}')
#         data = readsac.data
#         time = readsac.time

#         data_split = []
#         time_split = []
#         for i in range(11993):
#             num = i*10
#             data_split.append(data[num])
#             time_split.append(time[num])
#         data = data_split
#         time = time_split
#         # fig, ax = plt.subplots(figsize=(20, 5), dpi=500)
#         # ax.plot(time, data)
#         # fig.show()
#         save = save_hdf5(f'{save_fold}/{file[:-4]}.hdf5', data, time)




# f = h5py.File('/home/yuheng5454/MiDAS_test/data/hdf5/DAS/new_2023-08-13_01:43:10.32.hdf5', "r")
# print(f['waveform_data'][:])
# print(f['waveform_time'][:])

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(20, 5), dpi=500)
# ax.plot(f['waveform_time'][:], f['waveform_data'][:])
# fig.show()


# wave = load_dataset('/home/yuheng5454/MiDAS_test/data/hdf5/GLZ', ['new_GLZ_2023-06-11_02:20:58.27.hdf5'])
# waveform, label = wave[0]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(20, 5), dpi=500)
# ax.plot(waveform['waveform_time'][:], waveform['waveform_data'][:], linewidth=0.1, color='black')
# fig.show()
        

####################### Remove file #######################
# for i in tqdm(save_fold_list):
#     file_list = os.listdir(i)
#     for j in tqdm(file_list):
#         try:
#             os.remove(f'{i}/{j}')
#         except:
#             continue