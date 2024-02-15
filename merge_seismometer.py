import os
import subprocess
from tqdm import tqdm

filepath = os.listdir('/home/yuheng5454/MiDAS_test/data/MiDASA1/2023/GLZ')
filepath.sort()

for i in ['GLZ', 'GL1', 'GL2']:
    for j in tqdm(filepath):
        s = f'r /home/yuheng5454/MiDAS_test/data/MiDASA1/2023/{i}/{j}/* \n'
        s += 'merge overlap average \n'
        s += f'w /home/yuheng5454/MiDAS_test/data/MiDASA1/2023/{i}/{j}/TW.MDSA1..{i}.R.2023.{j}.SAC \n'
        s += 'q \n'
        subprocess.Popen(['sac'], stdin=subprocess.PIPE).communicate(s.encode())        


# #刪掉合併錯的檔案(全部)
# for i in ['GLZ', 'GL1', 'GL2']:
#     for j in tqdm(filepath):
#         try:
#             os.remove(f'/home/yuheng5454/MiDAS_test/data/MiDASA1/2023/{i}/{j}/TW.MDSA1..{i}.R.2023.{j}.SAC')
#         except:
#             continue