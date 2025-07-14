from imports import *


transform = transforms.Compose([
    transforms.Resize((256, 256)), # Resize to a fixed size
    #transforms.ColorJitter(brightness=0.2, contrast=0.2), # Augmentation, DUNNO ABOUT THIS, MAYBE LATER
    transforms.ToTensor(), # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Normalize the image to have mean 0.5 and std 0.5
    #transforms.Grayscale(num_output_channels=1), #allows the model to focus on plate numbers without color distraction
    #transforms.RandomRotation(degrees=3), # small tilt to simulate real-world scenarios, already present, commenting for now
    #transforms.RandomPerspective(distortion_scale=0.2, p=0.5) #this as well, to simulate real world random perspective distortions. have to check other dataset folders, maybe already present
])





transform_yolo = transforms.Compose([
    transforms.Resize((640, 640)), # Resize to a fixed size
    ])




#saving fields of the licence plate as global variables, i'm gonna use them later on
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


MY_DICTIONARY = provinces + [c for c in alphabets if c not in provinces] + [c for c in ads if c not in provinces and c not in alphabets]    #basically not to add three times "O"
MY_DICTIONARY = list(dict.fromkeys(MY_DICTIONARY))  # Remove duplicates while preserving order
# Create character to index and index to character mappings
char2idx = {c: i for i, c in enumerate(MY_DICTIONARY)}
idx2char = {i: c for i, c in enumerate(MY_DICTIONARY)}
BLANK_IDX = len(MY_DICTIONARY)  # CTC needs +1 for "blank" , so keep the len