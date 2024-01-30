from tqdm import tqdm
import subprocess
import os
import shutil
import pandas as pd

num = 109
filepath = os.listdir('/home/yuheng5454/MiDAS_test/data/MiDAS-A_1257')
filepath.reverse()

loss_data = []

for i in tqdm(filepath[num-1::]):
    try:
        filename = os.listdir(f'/home/yuheng5454/MiDAS_test/data/MiDAS-A_1257/{i}/mseed-dt100')
        file = '/home/yuheng5454/MiDAS_test/data/MiDAS-A_1257/'+ i +'/mseed-dt100/'+filename[0]
        s = f'r {file} \n'
        s += 'ppk \n'
        s += 'q \n'
        process = subprocess.Popen(['sac'], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = process.communicate(s.encode())
        print('if DAS data is ok,  PRESS [Enter] if not PRESS [1]')
        check = input()

        if check=="1":
            shutil.copy(file, '/home/yuheng5454/MiDAS_test/data/DAS_loss/'+filename[0])
            loss_data.append(filename[0])

    except KeyboardInterrupt:
        print("Program terminated by user.")
        break

print(loss_data)