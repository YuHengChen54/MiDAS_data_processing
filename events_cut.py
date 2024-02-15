import numpy as numpy
import pandas as pd 
import subprocess
from datetime import datetime, timedelta
from tqdm import tqdm

catalog = pd.read_csv("/home/yuheng5454/MiDAS_test/events_230311-231024_allML.csv")

################################ DAS ################################

for i in tqdm(range(catalog.shape[0])):
    date = catalog['date'][i]
    time = catalog['time'][i]
    ML = catalog['ML'][i]
    date_object = datetime(int(date[0:4]), int(date[5:7]), int(date[8:10]))
    day_of_year = date_object.timetuple().tm_yday
    day_of_year = str(day_of_year)
    if len(day_of_year) == 2:
        day_of_year = f'0{str(day_of_year)}'
    time_difference = timedelta(hours=int(time[0:2]), minutes=int(time[3:5]), seconds=int(time[6:8]))
    total_seconds = time_difference.total_seconds()
    das_file = f'/home/yuheng5454/MiDAS_test/data/MiDAS-A_1257/{date[0:4]}{date[5:7]}{date[8:10]}/mseed-dt100/TW.01257..HSF.D.2023.{day_of_year}.sac'
    
    s = f'r {das_file} \n'
    s += 'lh kztime \n'
    s += 'q \n'
    process = subprocess.Popen(['sac'], stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate(s.encode())
    output, error = process
    output = output.decode()
    if output.split('.')[-1][0:3] == '\n':
        kztime = 0
    else:
        kztime = int(output.split('.')[-1][0:3])/1000

    #調用SAC
    s = f'cut {total_seconds-kztime} {total_seconds-kztime+120} \n'
    s += f'r {das_file} \n'
    s += f'ch b 0 \n'
    s += f"TITLE ML={ML} Location Bottom Size Large \n"
    s += f'w /home/yuheng5454/MiDAS_test/data/event_cut_waveform_1257/new_{date}_{time}.sac \n'
    s += 'q \n'
    subprocess.Popen(['sac'], stdin=subprocess.PIPE).communicate(s.encode())

################################ seismometer ################################

# def convert_to_utc_plus_0(date_str, time_str):
#         # 將日期和時間合併成一個字串
#         datetime_str = f"{date_str} {time_str}"
        
#         # 解析成 datetime 物件
#         datetime_utc_plus_8 = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")

#         # 將時間減去8小時
#         datetime_utc_plus_0 = datetime_utc_plus_8 - timedelta(hours=8)

#         # 提取日期和時間的部分
#         result_date_str = datetime_utc_plus_0.strftime("%Y-%m-%d")
#         result_time_str = datetime_utc_plus_0.strftime("%H:%M:%S.%f")[:-3]

#         return result_date_str, result_time_str

# failed_list = []


# for i in tqdm(range(catalog.shape[0])):
#     date = catalog['date'][i]
#     time = catalog['time'][i]
#     ML = catalog['ML'][i]

#     date_object = datetime(int(date[0:4]), int(date[5:7]), int(date[8:10]))
#     day_of_year = date_object.timetuple().tm_yday
#     day_of_year = str(day_of_year)
#     if len(day_of_year) == 2:
#         day_of_year = f'0{str(day_of_year)}'
#     elif len(day_of_year) == 1:
#         day_of_year = f'00{str(day_of_year)}'
#     time_difference = timedelta(hours=int(time[0:2]), minutes=int(time[3:5]), seconds=int(time[6:8]))
#     total_seconds = time_difference.total_seconds()
    
#     seis_z_file = f'/home/yuheng5454/MiDAS_test/data/MiDASA1/2023/GLZ/{day_of_year}/TW.MDSA1..GLZ.R.2023.{day_of_year}.SAC'
#     seis_1_file = f'/home/yuheng5454/MiDAS_test/data/MiDASA1/2023/GL1/{day_of_year}/TW.MDSA1..GL1.R.2023.{day_of_year}.SAC'
#     seis_2_file = f'/home/yuheng5454/MiDAS_test/data/MiDASA1/2023/GL2/{day_of_year}/TW.MDSA1..GL2.R.2023.{day_of_year}.SAC'
    
#     for i, j in zip(['Z', '1', '2'], [seis_z_file, seis_1_file, seis_2_file]):
#         #調用SAC
#         s = f'cut {total_seconds} {total_seconds+120} \n'
#         s += f'r {j} \n'
#         s += f'ch b 0 \n'
#         s += f"TITLE ML={ML} Location Bottom Size Large \n"
#         s += f'w /home/yuheng5454/MiDAS_test/data/seis_event_cut_waveform_GL{i}/new_GL{i}_{date}_{time}.sac \n'
#         s += 'q \n'
#         subprocess.Popen(['sac'], stdin=subprocess.PIPE).communicate(s.encode())
