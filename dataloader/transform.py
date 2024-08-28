import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np

def OTSU_ROI(image):
    OctImage = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR) # PIL->CV2
    gray = cv2.cvtColor(OctImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    OTSU = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    OTSU = Image.fromarray(OTSU[..., ::-1])  # BGR -> RGB
    return OTSU

class OctTransform(object):
    def __init__(self, cfgs, img_size = 256, crop_size = 224, mode = "train"):
        self.image_size = (crop_size, crop_size)
        self.mode = mode
        self.cfgs = cfgs
        mean = cfgs['train_cfg']['oct']['IMG_MEAN']
        std = cfgs['train_cfg']['oct']['IMG_STD']
        self.transform1 = transforms.Compose([
                                              transforms.Resize((img_size, img_size)),
                                              transforms.CenterCrop(crop_size),
                                              ])
        self.transform2 = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean = mean, std = std)])
        self.transform3 = transforms.ToTensor()

    def __call__(self, x):
        x = self.transform1(x)
        OTSU = OTSU_ROI(x)
        x = self.transform2(x)
        OTSU = self.transform3(OTSU)
        return x.type(torch.FloatTensor), OTSU

class SloTransform(object):
    def __init__(self, cfgs, img_size = 224, mode = "train"):
        mean = cfgs['train_cfg']['slo']['IMG_MEAN']
        std = cfgs['train_cfg']['slo']['IMG_STD']
        self.divsor = 255.0
        self.mode = mode
        self.transform1 = transforms.Resize((img_size,img_size))
        self.transform2 = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean = mean, std = std)])
    def __call__(self, x, m_label):
        x = self.transform1(x)
        x = np.array(x) / self.divsor
        x = self.transform2(x)
        return x.type(torch.FloatTensor)

