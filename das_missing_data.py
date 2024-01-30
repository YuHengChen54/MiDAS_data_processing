import os
import subprocess 
from tqdm import tqdm

num = 1
filename = os.listdir('/home/yuheng5454/MiDAS_test/data/DAS_loss')


for i in tqdm(filename[num-1::]):
    file = '/home/yuheng5454/MiDAS_test/data/DAS_loss/'+i
    print(i)
    s = f'r {file} \n'
    s += 'ppk \n'
    s += 'q \n'
    process = subprocess.Popen(['sac'], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate(s.encode())