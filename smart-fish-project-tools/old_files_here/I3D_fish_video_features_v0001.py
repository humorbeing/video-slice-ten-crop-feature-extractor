import os

from torchvision.transforms import Resize
resize = Resize(256)

from torchvision.transforms import TenCrop
tencrop = TenCrop([224,224])

import torch
import sys
sys.path.append('.')
from models.i3d_v0001.i3d_rgb_model import I3D_rgb_model as model
import numpy as np  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.eval()
model.to(device)

from torchvision.io.video import read_video
# pip install av  # If pyav is missing


dataset_root = '/workspace/heavy-load-dataset-and-saves/datasets'
dataset_name = 'smart-fish-farm-dataset-hungry-disease/labeled'

dataset_path = os.path.join(dataset_root, dataset_name)

hungry_file_name = 'hungry_full_list.txt'
hungry_file_path = os.path.join(dataset_path, hungry_file_name)
with open(hungry_file_path, 'r') as f:
    hungry_path = f.readlines()

hungry_list = []
for hungry in hungry_path:
    temp11 = hungry.strip('\n')
    one_path = os.path.join(dataset_path, temp11)
    hungry_list.append(one_path)
    # print('end')


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

  

def get_slice(video_duration, target_number_segment=32, one_segment_size=16):
    
    end_point = video_duration - one_segment_size
    idx_float_list = np.arange(
        0, end_point, 
        step=end_point/(target_number_segment-1),
        dtype=float)
    idx_floor = np.floor(idx_float_list)
    idx_int = idx_floor.astype(np.int32)
    slice_start = np.append(idx_int, [end_point])
    return slice_start

def get_video_feature(model, rgb, duration, device):

    # above first, RGB, not BGR (when cv2, make sure)


    # THWC to TCHW. TCHW is pytorch prefered dimension order
    # T here is regard as Batch-size. for example, normalization
    rgb_permute_TCHW = torch.permute(rgb, [0,3,1,2])


    # resize to 256 for cropping.
    # resize here, to save computation
    
    rgb_resize = resize.forward(rgb_permute_TCHW)

    # 255 to -1 to 1
    rgb_dev255 = (rgb_resize * 2 / 255) - 1.

    # 10 crop    
    rgb_temp11 = tencrop.forward(rgb_dev255)
    rgb_tencrop = torch.stack(rgb_temp11)

    # Ncrop T C H W   to   Ncrop C T H W
    rgb_ready_ten_NcCTHW = torch.permute(rgb_tencrop, [0,2,1,3,4])


    

    slice_start_idx = get_slice(
        video_duration=duration,
        target_number_segment=32)

    output_list = []
    for startp in slice_start_idx:
        rgb_oneslice = rgb_ready_ten_NcCTHW[:,:,startp:startp+16,:,:]
        
        input_features = rgb_oneslice.to(device)
        with torch.no_grad():
            output_tencrop = model.forward(input_features, features=True)

        output_list.append(output_tencrop)

    outputs = torch.stack(output_list)
    numpy_outputs = outputs.cpu().numpy()
    return numpy_outputs


hungry_error = []
hungrys = []
dataset_savename = 'smart-fish-farm-dataset-hungry-disease/fast_slow_i3d_1024_tencrop_32seg'

save_path = os.path.join(dataset_root, dataset_savename)
os.makedirs(save_path, exist_ok=True)

# hungry to fast

class_name = 'fast'
fast_path = os.path.join(save_path, 'fast')
os.makedirs(fast_path, exist_ok=True)
import logging
log_path = os.path.join(save_path, 'fast.log')
logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w')
logging.info("starting")
count = 1
for hungry in hungry_list:
    original_name = hungry.split('/')[-1]
    try:
        rgb, audio, info = read_video(hungry, pts_unit='sec')
        duration = rgb.size(0)
        if duration > 100:
            output = get_video_feature(model, rgb, duration, device)

            file_name = f'{class_name}_{count:08d}.npy'
            file_path = os.path.join(fast_path, file_name)
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