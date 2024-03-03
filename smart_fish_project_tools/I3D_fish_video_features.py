# python module
import os

# pip installed module
import numpy as np

import torch


from torchvision.io.video import read_video
# pip install av  # If pyav is missing

# My-Me-made module
import sys
sys.path.append('.')
from smart_fish_project_tools.i3d_utility import get_i3d_video_feature

# options
num_segment_cut_from_video = 32


# configurations
dataset_root = '/workspace/heavy-load-dataset-and-saves/datasets'
dataset_name = 'smart-fish-farm-dataset-hungry-disease/labeled'
dataset_savename = f'smart-fish-farm-dataset-hungry-disease/fast_slow_i3d_1024_tencrop_{num_segment_cut_from_video}seg'
dataset_path = os.path.join(dataset_root, dataset_name)

# hungry_file_name = 'hungry_full_list.txt'
# hungry_file_path = os.path.join(dataset_path, hungry_file_name)
# with open(hungry_file_path, 'r') as f:
#     hungry_path = f.readlines()

# hungry_list = []
# for hungry in hungry_path:
#     temp11 = hungry.strip('\n')
#     one_path = os.path.join(dataset_path, temp11)
#     hungry_list.append(one_path)
  
# save_path = os.path.join(dataset_root, dataset_savename)
# os.makedirs(save_path, exist_ok=True)

# # hungry to fast

# class_name = 'fast'
# fast_path = os.path.join(save_path, 'fast')
# os.makedirs(fast_path, exist_ok=True)
# import logging
# log_path = os.path.join(save_path, 'fast.log')
# logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w')
# logging.info("starting")
# count = 1
# for hungry in hungry_list:
#     original_name = hungry.split('/')[-1]
#     try:
#         rgb, audio, info = read_video(hungry, pts_unit='sec')
#         duration = rgb.size(0)
#         if duration > 100:
#             output = get_i3d_video_feature(rgb,n_segments=num_segment_cut_from_video)

#             file_name = f'{class_name}_{count:08d}.npy'
#             file_path = os.path.join(fast_path, file_name)
#             np.save(file_path, output)
            
#             msg = f'OK: {original_name} -> {file_name}'
#             count = count + 1
#             # print('end')
#         else:
#             msg = f'BAD: {original_name} number of frames: {duration}'
#             print(msg)
#     except:
#         msg = f'ERROR: {original_name} Broken'
#         print(msg)
    
#     logging.info(msg)

#     # print('end')


stuffed_file_name = 'stuffed_full_list.txt'
stuffed_file_path = os.path.join(dataset_path, stuffed_file_name)
with open(stuffed_file_path, 'r') as f:
    stuffed_path = f.readlines()

stuffed_list = []
for stuffed in stuffed_path:
    temp11 = stuffed.strip('\n')
    one_path = os.path.join(dataset_path, temp11)
    stuffed_list.append(one_path)
    # print('end')


save_path = os.path.join(dataset_root, dataset_savename)
os.makedirs(save_path, exist_ok=True)

# hungry to fast

class_name = 'slow'
slow_path = os.path.join(save_path, 'slow')
os.makedirs(slow_path, exist_ok=True)
import logging
log_path = os.path.join(save_path, 'slow.log')
logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w')
logging.info("starting")
count = 1
for stuffed in stuffed_list:
    original_name = stuffed.split('/')[-1]
    try:
        rgb, audio, info = read_video(stuffed, pts_unit='sec')
        duration = rgb.size(0)
        if duration > 100:
            output = get_i3d_video_feature(rgb,n_segments=num_segment_cut_from_video)

            file_name = f'{class_name}_{count:08d}.npy'
            file_path = os.path.join(slow_path, file_name)
            np.save(file_path, output)
            
            msg = f'OK: {original_name} -> {file_name}'
            count = count + 1
            # print('end')
        else:
            msg = f'BAD: {original_name} number of frames: {duration}'
            print(msg)
    except:
        msg = f'ERROR: {original_name} Broken'
        print(msg)
    
    logging.info(msg)

    # print('end')

print('end')