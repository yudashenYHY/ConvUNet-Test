import os

import numpy as np
from torchvision import transforms, utils, datasets
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch
import matplotlib.pyplot as plt



class BrainDataset(Dataset):
    def __init__(self, train: bool, transforms=None):
        super(BrainDataset, self).__init__()
        # self.flag = "Training" if train else "Testing"
        # data_root = os.path.join(root, self.flag) ##./archive/Training
        # assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        # labels = os.listdir(data_root)##类别列表
        labels = os.listdir(os.path.join(os.getcwd(), 'archive', 'Training'))  ##类别列表

        img_names = []
        # cls = open(os.path.join(os.getcwd(), 'archive', 'Training', 'train') + '.txt', 'a+')
        # print(os.path.join(os.getcwd(), 'archive', 'Training', 'train') + '.txt')
        for file in labels:
            # img_names = [i for i in os.listdir(os.path.join(os.getcwd(), file))]
            # print(os.listdir(os.path.join(os.getcwd(), 'archive', 'Training', file)))
            for i, k in enumerate(os.listdir(os.path.join(os.getcwd(), 'archive', 'Training', file))):
                print(os.path.join(os.getcwd(), file, k))
                # cls.write(os.path.join(os.getcwd(), file, k) + ' ' + file + '\n')

        # cls.close()

        # print(img_names)



    def __getitem__(self, item):
        file_list = os.listdir(os.path.join(os.getcwd(), 'archive'))[:4]

        image = self.img_list[item]
        label = self.label
        image = Image.open(image).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
       return 1



get_transform = transforms.Compose([
        transforms.RandomResizedCrop(480),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
# path = 'D:\Python\pythonProject1\ConvUNeXt-text\\archive'
train_dataset = BrainDataset(train=True, transforms=None)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=4,
                                           shuffle=True)

for img in train_loader:
    print('----')
    print(img)

