from tqdm import tqdm
import subprocess
import os
import shutil
import pandas as pd

num = 1000
filename = os.listdir('/home/yuheng5454/MiDAS_test/data/event_cut_waveform_1257')

not_ok_list = []
for i in tqdm(filename[num-1::]):
    try:
        das = '/home/yuheng5454/MiDAS_test/data/event_cut_waveform_1257/'+i
        GL1 = '/home/yuheng5454/MiDAS_test/data/seis_event_cut_waveform_GL1/'+i[0:4]+'GL1_'+i[4:]
        GL2 = '/home/yuheng5454/MiDAS_test/data/seis_event_cut_waveform_GL2/'+i[0:4]+'GL2_'+i[4:]
        GLZ = '/home/yuheng5454/MiDAS_test/data/seis_event_cut_waveform_GLZ/'+i[0:4]+'GLZ_'+i[4:]
        s = f'r {das} {GLZ} {GL1} {GL2} \n'
        s += f'title {i} \n'
        s += 'ppk \n'
        s += 'q \n'
        process = subprocess.Popen(['sac'], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = process.communicate(s.encode())

        if process.returncode != 0:
            # 如果 sac 命令返回非零退出碼，表示有錯誤
            print(f"Error processing {i}: {stderr.decode()}")
            not_ok_list.append(i)
            continue

        print('if DAS data and seismometer is ok,  PRESS [Enter] if not PRESS [1]')
        check = input()
        if check=="1":
            not_ok_list.append(i)
        else:
            shutil.copy(das, '/home/yuheng5454/MiDAS_test/data/checked/1257/'+i)
            shutil.copy(GL1, f'/home/yuheng5454/MiDAS_test/data/checked/GL1/'+i[0:4]+'GL1_'+i[4:])
            shutil.copy(GL2, f'/home/yuheng5454/MiDAS_test/data/checked/GL2/'+i[0:4]+'GL2_'+i[4:])
            shutil.copy(GLZ, f'/home/yuheng5454/MiDAS_test/data/checked/GLZ/'+i[0:4]+'GLZ_'+i[4:])
    except KeyboardInterrupt:
            print("Program terminated by user.")
            break

print(not_ok_list)
############# 把DAS的資料跟地震儀的三分輛資料一起畫圖，然後檢查可以的複製到checked #############