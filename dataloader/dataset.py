import os
import torch
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dataloader.transform import OctTransform, SloTransform
class AierDataset(Dataset):
    def __init__(self, cfgs, mode = "train"):
        super(AierDataset, self).__init__()
        self.cfgs = cfgs
        self.root_dir = self.cfgs['base_cfg']['root']
        self.mode = mode
        self.item_list = self.__parseDataset__()
        self.octTransform = OctTransform(cfgs, mode = mode)
        self.sloTransform = SloTransform(cfgs, mode = mode)

    def __getitem__(self, index):
        row = self.item_list.iloc[index]
        missModalTag = [1, 1, 1] # patientMsg、OCT、SLO
        if(pd.isna(row.OCT)):
            missModalTag[1] = 0
        if (pd.isna(row.SLO)):
            missModalTag[2] = 0
        OctImage = self.loadImage((os.path.join(self.root_dir, row.base)), row.OCT)
        SloImage = self.loadImage((os.path.join(self.root_dir, row.base)), row.SLO)
        OctImage, OTSU = self.octTransform(OctImage)
        ROI = torch.load(os.path.join(os.path.join(self.root_dir, row.base), "ROI.pt"), map_location = 'cpu')
        SloImage = self.sloTransform(SloImage, missModalTag[2])
        label = torch.tensor(row.postLogMAR)
        patientMessage = torch.load(os.path.join(os.path.join(self.root_dir, row.base), "patient.pt"), map_location = 'cpu')
        diagOct = torch.load(os.path.join(os.path.join(self.root_dir, row.base), "diagOct.pt"), map_location = 'cpu')
        diagSlo = torch.load(os.path.join(os.path.join(self.root_dir, row.base), "diagSlo.pt"), map_location = 'cpu')
        return OctImage, SloImage, label, patientMessage, diagOct, diagSlo, missModalTag, OTSU, ROI
    def __len__(self):
        return len(self.item_list)
    def __parseDataset__(self):
        self.txt = self.cfgs['base_cfg']['txt']
        dataList = pd.read_csv(os.path.join(self.root_dir, self.txt), encoding = "utf-8")
        seed = self.cfgs['base_cfg']['seed']
        dataList = dataList.sample(frac = 1, random_state = seed, replace = False)
        if self.mode == "train":
            data = dataList[0: int(len(dataList) * 0.8)]
        elif self.mode == "val":
            data = dataList[int(len(dataList) * 0.8): int(len(dataList) * 0.9)]
        else:
            data = dataList[int(len(dataList) * 0.9): int(len(dataList))]
        return data

    def loadImage(self, imageBasePath, imageName):
        if(pd.isna(imageName)):
            img = np.zeros((512, 512, 3), dtype = np.uint8)
            img = Image.fromarray(img)
        else:
            img = Image.open(os.path.join(imageBasePath, imageName)).convert("RGB")
        return img

def dataloader(dataset, cfgs):
    return DataLoader(dataset = dataset,
                    batch_size = cfgs['train_cfg']['Batch_Size'],
                    shuffle = True,
                    pin_memory = True,
                    num_workers = 8)
