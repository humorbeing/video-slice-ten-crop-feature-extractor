import torchvision.models as models
import torch
import numpy as np

r21d_model_cfgs = {
    'r2plus1d_18_16_kinetics': {
        'repo': None,
        'stack_size': 16, 'step_size': 16, 'num_classes': 400, 'dataset': 'kinetics'
    },
    'r2plus1d_34_32_ig65m_ft_kinetics': {
        'repo': 'moabitcoin/ig65m-pytorch', 'model_name_in_repo': 'r2plus1d_34_32_kinetics',
        'stack_size': 32, 'step_size': 32, 'num_classes': 400, 'dataset': 'kinetics'
    },
    'r2plus1d_34_8_ig65m_ft_kinetics': {
        'repo': 'moabitcoin/ig65m-pytorch', 'model_name_in_repo': 'r2plus1d_34_8_kinetics',
        'stack_size': 8, 'step_size': 8, 'num_classes': 400, 'dataset': 'kinetics'
    },
}

weights_key = 'DEFAULT'
model = models.get_model('r2plus1d_18', weights=weights_key)

model = torch.hub.load(
    'moabitcoin/ig65m-pytorch',
    'r2plus1d_34_32_kinetics',
    400,
    pretrained=True,
)

model = torch.hub.load(
    'moabitcoin/ig65m-pytorch',
    'r2plus1d_34_8_kinetics',
    400,
    pretrained=True,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)

input_features = torch.randn(1,3,16,112,112)
input_features = input_features.to(device)

outputs = model(input_features)


from torchvision.io.video import read_video
# pip install av
video_path = '/workspace/datasets/smart-fish-farm/labeled/hungry/2020-07-10/feeding_time_2020-07-07T18"06"11_2020-07-07T18"08"00-4-of-16.mp4'
rgb, audio, info = read_video(video_path, pts_unit='sec')

video_info = {
    'fps': info['video_fps'],
    'duration': rgb.size(0),
    'height': rgb.size(1),
    'width': rgb.size(2),
}
from torchvision.transforms import TenCrop
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
normalize = Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
tencrop = TenCrop([112,112])
resize = Resize(128)


rgb_dev255 = rgb / 255.
rgb_permute_fornorm = torch.permute(rgb_dev255, [0,3,1,2])

rgb_resize = resize.forward(rgb_permute_fornorm)

rgb_normalize = normalize.forward(rgb_resize)
rgb_permute = torch.permute(rgb_normalize, [1,0,2,3])
rgb_temp11 = tencrop.forward(rgb_permute)

rgb_tencrop = torch.stack(rgb_temp11)

target_number_segment = 32
duration = video_info['duration']
one_stack_size = 16
end_point = duration - one_stack_size
idx_float_list = np.arange(
    0, end_point, 
    step=end_point/(target_number_segment-1),
    dtype=float)
idx_floor = np.floor(idx_float_list)
idx_int = idx_floor.astype(np.int32)
slice_start = np.append(idx_int, [end_point])

output_list = []
for startp in slice_start:
    rgb_oneslice = rgb_tencrop[:,:,startp:startp+16,:,:]
    
    input_features = rgb_oneslice.to(device)
    with torch.no_grad():
        output_tencrop = model.forward(input_features)

    output_list.append(output_tencrop)

outputs = torch.stack(output_list)


from torchvision.models.feature_extraction import get_graph_node_names
train_nodes, eval_nodes = get_graph_node_names(model)

for i in eval_nodes:
    print(i)

return_nodes = {
    # node_name: user-specified key for output dict
    'flatten': 'layer1',
}

from torchvision.models.feature_extraction import create_feature_extractor
import torch

extractor = create_feature_extractor(
    model, return_nodes=return_nodes)

model.fc = torch.nn.Identity()

output_list = []
for startp in slice_start:
    rgb_oneslice = rgb_tencrop[:,:,startp:startp+16,:,:]
    
    input_features = rgb_oneslice.to(device)
    with torch.no_grad():
        # output_tencrop = extractor(input_features)['layer1']
        output_tencrop = model(input_features)


    output_list.append(output_tencrop)

outputs = torch.stack(output_list)


print("end")