import torch

from models.i3d_v0001.i3d_rgb_model import I3D_rgb_model as model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.eval()
model.to(device)    

input_features = torch.randn(1,3,16,224,224)
input_features = input_features.to(device)

outputs = model(input_features, features=True)

print('end')