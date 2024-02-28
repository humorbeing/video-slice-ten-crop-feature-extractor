import torch

from models.i3d_v0001.i3d_rgb_model import I3D_rgb_model as model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.eval()
model.to(device)    

from torchvision.io.video import read_video
# pip install av  # If pyav is missing
video_path = './videos/fast.mp4'
video_path = './videos/slow.mp4'
rgb, audio, info = read_video(video_path, pts_unit='sec')

video_info = {
    'fps': info['video_fps'],
    'duration': rgb.size(0),
    'height': rgb.size(1),
    'width': rgb.size(2),
}

rgb_dev255 = rgb / 255.
rgb_permute_TCHW = torch.permute(rgb_dev255, [0,3,1,2])



from torchvision.transforms import Resize
resize = Resize(256)
rgb_resize = resize.forward(rgb_permute_TCHW)

from torchvision.transforms import TenCrop
tencrop = TenCrop([224,224])
rgb_temp11 = tencrop.forward(rgb_resize)
rgb_tencrop = torch.stack(rgb_temp11)

rgb_ready_ten_NcCTHW = torch.permute(rgb_tencrop, [0,2,1,3,4])


def get_slice(video_duration, target_number_segment=32, one_segment_size=16):
    import numpy as np    
    end_point = video_duration - one_segment_size
    idx_float_list = np.arange(
        0, end_point, 
        step=end_point/(target_number_segment-1),
        dtype=float)
    idx_floor = np.floor(idx_float_list)
    idx_int = idx_floor.astype(np.int32)
    slice_start = np.append(idx_int, [end_point])
    return slice_start

slice_start_idx = get_slice(
    video_duration=video_info['duration'],
    target_number_segment=12)

output_list = []
for startp in slice_start_idx:
    rgb_oneslice = rgb_ready_ten_NcCTHW[:,:,startp:startp+16,:,:]
    
    input_features = rgb_oneslice.to(device)
    with torch.no_grad():
        output_tencrop = model.forward(input_features, features=True)

    output_list.append(output_tencrop)

outputs = torch.stack(output_list)
print('end')