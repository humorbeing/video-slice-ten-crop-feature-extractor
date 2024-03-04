import os

dataset_root = '/workspace/datasets'
dataset_name = 'smart-fish-farm/broken_fine_video'

dataset_path = os.path.join(dataset_root, dataset_name)

# class_name = ['hungry', 'stuffed']



class_name = 'broken'
dataset_one_path = os.path.join(dataset_path, class_name)
# stuffed_path = os.path.join(dataset_path, 'stuffed')


file_list = []

broken_file_list = os.listdir(dataset_one_path)
import random
random.seed(2024)

random.shuffle(broken_file_list)
num_test = 80
atest_list = broken_file_list[:num_test]
atrain_list = broken_file_list[num_test:]

class_name = 'normal'
dataset_one_path = os.path.join(dataset_path, class_name)
# stuffed_path = os.path.join(dataset_path, 'stuffed')


file_list = []

normal_file_list = os.listdir(dataset_one_path)

random.shuffle(normal_file_list)
num_test = 100
ntest_list = normal_file_list[:num_test]
ntrain_list = normal_file_list[num_test:]


# with open('broken_test.txt', 'w') as f:
#     for l in test_list:
#         f.writelines(l + '\n')

test_list = []
test_list.extend(ntest_list)
test_list.extend(atest_list)


with open('test_list.txt', 'w') as f:
    for l in test_list:
        f.writelines(l + '\n')

print('end')
# import shutil
# count = 0
# rename_list = []
# for file_name in broken_file_list:
#     count = count + 1
#     before_name_path = os.path.join(dataset_one_path, file_name)
#     new_name = f'{class_name}_{count:08d}.mp4'
#     after_name_path = os.path.join(dataset_path, 'new_normal', new_name)
#     shutil.copyfile(before_name_path, after_name_path)
#     msg = f'{before_name_path} --> {after_name_path}'
#     rename_list.append(msg)


# with open('rename_list_normal.txt', 'w') as f:
#     for l in rename_list:
#         f.writelines(l + '\n')
# print('end')
class_name = 'normal'
dataset_one_path = os.path.join(dataset_path, class_name)
# stuffed_path = os.path.join(dataset_path, 'stuffed')


file_list = []

normal_file_list = os.listdir(dataset_one_path)


print('end')
# for folder in folders:
#     folder_path = os.path.join(dataset_one_path, folder)
#     files = os.listdir(folder_path)
#     for file in files:
#         file_path = os.path.join(class_name, folder, file)
#         file_list.append(file_path)
# import shutil
# for temp11 in file_list:
#     # temp11 = file_list[0]
#     this_file_name = temp11.split('/')[-1]
#     from_path = os.path.join(dataset_path, temp11)
#     to_folder = os.path.join(dataset_path, 'normal')
#     to_path = os.path.join(to_folder, this_file_name)    
#     try:
#         shutil.copyfile(from_path, to_path)
#     except:
#         pass

dataset_one_path = os.path.join(dataset_path, 'fine_a')
# stuffed_path = os.path.join(dataset_path, 'stuffed')


file_list = []

folders = os.listdir(dataset_one_path)

for folder in folders:
    folder_path = os.path.join(dataset_one_path, folder)
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join('hungry', folder, file)
        file_list.append(file_path)
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