import numpy as np
import torch

from torchvision.transforms import Resize
resize = Resize(256)

from torchvision.transforms import TenCrop
tencrop = TenCrop([224,224])


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

def get_video_feature(model, rgb, duration, n_segments=32, one_seg_size=16):
    device = next(model.parameters()).device
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

