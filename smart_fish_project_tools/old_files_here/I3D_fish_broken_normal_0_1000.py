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
num_segment_cut_from_video = 12


# configurations
dataset_root = '/workspace/datasets'
dataset_name = 'smart-fish-farm/broken_video'
dataset_savename = f'smart-fish-farm/broken_video_i3d_1024_tencrop_{num_segment_cut_from_video}seg'
dataset_path = os.path.join(dataset_root, dataset_name)

class_name = 'normal'
one_dataset_path = os.path.join(dataset_path, class_name)

target_files = os.listdir(one_dataset_path)

save_folder_path = os.path.join(dataset_root, dataset_savename)
os.makedirs(save_folder_path, exist_ok=True)

one_folder_path = os.path.join(save_folder_path, class_name)
os.makedirs(one_folder_path, exist_ok=True)

sort_list = sorted(target_files)
mission_list = sort_list[:1000]


for target in mission_list:
    
    original_num = target[-12:-4]
    target_path = os.path.join(one_dataset_path, target)
    rgb, audio, info = read_video(target_path, pts_unit='sec')        
    output = get_i3d_video_feature(rgb,n_segments=num_segment_cut_from_video)

    file_name = f'{class_name}_{original_num}.npy'
    file_path = os.path.join(one_folder_path, file_name)
    np.save(file_path, output)      
            
    