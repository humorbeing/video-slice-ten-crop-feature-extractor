import torchvision.models as models
import torch


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
model_1 = models.get_model('r2plus1d_18', weights=weights_key)

model_2 = torch.hub.load(
    'moabitcoin/ig65m-pytorch',
    'r2plus1d_34_32_kinetics',
    400,
    pretrained=True,
)

model_3 = torch.hub.load(
    'moabitcoin/ig65m-pytorch',
    'r2plus1d_34_8_kinetics',
    400,
    pretrained=True,
)

model = model_1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)

input_features = torch.randn(1,3,16,112,112)
input_features = input_features.to(device)

outputs = model(input_features)

model.fc = torch.nn.Identity()

with torch.no_grad():        
    output_tencrop1 = model(input_features)

from torchvision.models.feature_extraction import get_graph_node_names
train_nodes, eval_nodes = get_graph_node_names(model)

for i in eval_nodes:
    print(i)

return_nodes = {
    # node_name: user-specified key for output dict
    'flatten': 'layer1',
}

from torchvision.models.feature_extraction import create_feature_extractor

extractor = create_feature_extractor(
    model, return_nodes=return_nodes)

with torch.no_grad():
    output_tencrop2 = extractor(input_features)['layer1']

print("end")