import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


ROOT_PATH = './cmr/misc/CUB_200_2011/'


class CUB200(Dataset):

    def __init__(self, setname):
        lines = open(ROOT_PATH+'images.txt','r').readlines()
        idx_name={}
        for line in lines:
            idx, file_name =line.split(' ')
            file_name=file_name.split('\n')[0]
            idx_name[idx]=file_name
            
        lines = open(ROOT_PATH+'classes.txt','r').readlines()
        idx_cls={}
        for line in lines:
            idx, cls_name =line.split(' ')
            cls_name=cls_name.split('\n')[0]
            idx_cls[cls_name]=idx
            

        data=[]
        label=[]
        lines = open(ROOT_PATH+'train_test_split.txt','r').readlines()
        for line in lines:
            idx, is_train = line.split(' ')
            cls_name, file=idx_name[idx].split('/')
            file=file.split('\n')[0]
            is_train=is_train.split('\n')[0]
            if setname=='train' and is_train=='1':
                data.append(os.path.join(ROOT_PATH,'train/',idx_cls[cls_name], file))
                label.append(int(idx_cls[cls_name])-1)
            elif setname=='val' and is_train=='0':
                data.append(os.path.join(ROOT_PATH, 'test/',idx_cls[cls_name], file))
                label.append(int(idx_cls[cls_name])-1)
               
 
        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image=Image.open(path).convert('RGB')
        image = self.transform(image)
        
        return image, label

