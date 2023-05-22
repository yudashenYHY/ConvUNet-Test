import os

import numpy as np
from torchvision import transforms, utils, datasets
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch
import matplotlib.pyplot as plt



class BrainDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(BrainDataset, self).__init__()
        self.flag = "Training" if train else "Testing"
        data_root = os.path.join(root, self.flag) ##./archive/Training
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms

        labels = os.listdir(data_root)##类别列表
        # print(labels)
        # print(labels[0])##glioma
        if not os.path.exists(os.path.join(root, self.flag)+'.txt'):
            cls = open(os.path.join(root,self.flag +'.txt'), 'a+')
        for file in labels:
            #img_names = [i for i in os.listdir(os.path.join(data_root, file))]
            #print(img_names)
            #print('1')
            for k in os.listdir(os.path.join(data_root, file)):
                # print(os.path.getsize(os.path.join(data_root, self.flag)+'.txt'))
                ##if not os.path.getsize(os.path.join(root, self.flag+'.txt')):
                    cls.write(os.path.join(data_root, file, k) + ' ' + file + '\n')
        self.img_list = []
        with open(os.path.join(root,self.flag +'.txt'), "r") as f:
            for line in f.readlines():
                line = line.strip("\n")
                line = line.split()
                self.img_list.append(line)


    def __getitem__(self, item):
        image = self.img_list[item][0]
        label = self.img_list[item][1]
        image = Image.open(image).convert('RGB')
        if self.transforms is not None:
            image= self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.img_list)

    def my_collate(batch):
        inputs = [data[0].tolist() for data in batch]
        inputs = torch.tensor(inputs)
        target = [data[1].tolist() for data in batch]
        target = torch.tensor(target)
        return [inputs, target]




get_transform = transforms.Compose([
        transforms.RandomResizedCrop(480),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
# path = ".\\archive"
# train_dataset = BrainDataset(path, train=True, transforms=get_transform)
#
# train_loader = torch.utils.data.DataLoader(train_dataset,
#                                            batch_size=1,
#                                            shuffle=True,
#                                            )
# for img,label in train_loader:
#     print(img.shape,label)



