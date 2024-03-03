import os

dataset_root = '/workspace/heavy-load-dataset-and-saves/datasets'
dataset_name = 'smart-fish-farm-dataset-hungry-disease/labeled'

dataset_path = os.path.join(dataset_root, dataset_name)

class_name = ['hungry', 'stuffed']

hungry_path = os.path.join(dataset_path, 'hungry')
stuffed_path = os.path.join(dataset_path, 'stuffed')


hungry_list = []

folders = os.listdir(hungry_path)

for folder in folders:
    folder_path = os.path.join(hungry_path, folder)
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join('hungry', folder, file)
        hungry_list.append(file_path)

hungry_file_name = 'hungry_full_list.txt'
hungry_file_path = os.path.join(dataset_path, hungry_file_name)
with open(hungry_file_path,'w') as f:
    for hungry in hungry_list:
        f.writelines(hungry + '\n')


stuffed_list = []

folders = os.listdir(stuffed_path)

for folder in folders:
    folder_path = os.path.join(stuffed_path, folder)
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join('stuffed', folder, file)
        stuffed_list.append(file_path)

stuffed_file_name = 'stuffed_full_list.txt'
stuffed_file_path = os.path.join(dataset_path, stuffed_file_name)
with open(stuffed_file_path,'w') as f:
    for stuffed in stuffed_list:
        f.writelines(stuffed + '\n')


print('end')