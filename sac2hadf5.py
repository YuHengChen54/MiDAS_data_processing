import h5py
from obspy import read
import os

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

fold_list = ['/home/yuheng5454/MiDAS_test/data/event_cut_waveform_1257', 
            '/home/yuheng5454/MiDAS_test/data/seis_event_cut_waveform_GL1', 
            '/home/yuheng5454/MiDAS_test/data/seis_event_cut_waveform_GL2', 
            '/home/yuheng5454/MiDAS_test/data/seis_event_cut_waveform_GLZ'
            ]
save_fold_list = ['/home/yuheng5454/MiDAS_test/data/hdf5/DAS',
                '/home/yuheng5454/MiDAS_test/data/hdf5/GL1', 
                '/home/yuheng5454/MiDAS_test/data/hdf5/GL2', 
                '/home/yuheng5454/MiDAS_test/data/hdf5/GLZ'
                ]
#寫for把所有DAS+井下地震儀改存hdf5
for fold, save_fold in zip(fold_list, save_fold_list):
    file_list = os.listdir(fold)
    for file in file_list:
        # print(fold+'/'+file)
        readsac = read_sac(fold+'/'+file)
        data = readsac.data
        time = readsac.time

        save = save_hdf5(f'{save_fold}/{file[:-4]}.hdf5', data, time)


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