import torch

from models.i3d_v0001.i3d_rgb_model import I3D_rgb_model as model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.eval()
model.to(device)    

from torchvision.io.video import read_video
# pip install av
video_path = './videos/fast.mp4'
video_path = './videos/slow.mp4'
rgb, audio, info = read_video(video_path, pts_unit='sec')

print('end')