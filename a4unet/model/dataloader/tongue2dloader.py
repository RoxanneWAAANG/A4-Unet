import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate

class Tongue2D_Dataset(Dataset):
    def __init__(self, data_path , transform = None, mode = 'Test'):

        if mode == "Training":
            df = pd.read_csv('/home/fyp1/ML/MedSegDiff-master/data/tongue_2d/tongue_train.csv', encoding='gbk')
        else:
            print("read gt file")
            df = pd.read_csv('/home/fyp1/ML/MedSegDiff-master/data/tongue_2d/tongue_test.csv', encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        mask_name = self.label_list[index]

        img_path = os.path.join(str(self.data_path) + '/train_img/' + str(name) if self.mode == 'Training' else str(self.data_path) + '/test_img/' + str(name))
        msk_path = os.path.join(str(self.data_path) + '/train_gt/' + str(mask_name) if self.mode == 'Trainig' else str(self.data_path) + '/test_gt/' + str(mask_name))

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)


        return (img, mask, name)


