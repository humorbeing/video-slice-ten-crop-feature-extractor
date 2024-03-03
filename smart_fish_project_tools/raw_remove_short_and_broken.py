import os

dataset_root = '/workspace/datasets'
dataset_name = 'smart-fish-farm/hungry-stuffed-video_v0001'

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


stuffed_list = []

folders = os.listdir(stuffed_path)

for folder in folders:
    folder_path = os.path.join(stuffed_path, folder)
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join('stuffed', folder, file)
        stuffed_list.append(file_path)


import cv2


remove_list = []
for temp11 in hungry_list:    
    temp12 = os.path.join(dataset_path, temp11)

    cap = cv2.VideoCapture(temp12)
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 100:
            msg = f'{frame_count}: {temp12}'
            print(msg)
            remove_list.append(msg)
            
            if os.path.exists(temp12):
                os.remove(temp12)
            else:
                print("The file does not exist")
    else:
        msg = f'video error: {temp12}'
        print(msg)
        remove_list.append(msg)
        if os.path.exists(temp12):
            os.remove(temp12)
        else:
            print("The file does not exist")


for temp11 in stuffed_list:    
    temp12 = os.path.join(dataset_path, temp11)

    cap = cv2.VideoCapture(temp12)
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 100:
            msg = f'{frame_count}: {temp12}'
            print(msg)
            remove_list.append(msg)
            
            if os.path.exists(temp12):
                os.remove(temp12)
            else:
                print("The file does not exist")
    else:
        msg = f'video error: {temp12}'
        print(msg)
        remove_list.append(msg)
        if os.path.exists(temp12):
            os.remove(temp12)
        else:
            print("The file does not exist")

cap.release()
# cv2.destroyAllWindows()

with open('remove_list.txt', 'w') as f:
    for l in remove_list:
        f.writelines(l+'\n')

print('end')