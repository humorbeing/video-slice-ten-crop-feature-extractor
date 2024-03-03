import torch

import sys
sys.path.append('.')
from models.i3d_v0001.i3d_rgb_model import I3D_rgb_model as model
from smart_fish_project_tools.utility import get_slice


from torchvision.transforms import Resize
resize = Resize(256)

from torchvision.transforms import TenCrop
tencrop = TenCrop([224,224])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.eval()
model.to(device)

def get_i3d_video_feature(rgb, n_segments=32, one_seg_size=16):
    duration = rgb.size(0)
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
        target_number_segment=n_segments,
        one_segment_size=one_seg_size)

    output_list = []
    for startp in slice_start_idx:
        rgb_oneslice = rgb_ready_ten_NcCTHW[:,:,startp:startp+one_seg_size,:,:]
        
        input_features = rgb_oneslice.to(device)
        with torch.no_grad():
            output_tencrop = model.forward(input_features, features=True)

        output_list.append(output_tencrop)

    outputs = torch.stack(output_list)
    numpy_outputs = outputs.cpu().numpy()
    return numpy_outputs

