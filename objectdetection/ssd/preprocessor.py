import torch
import numpy as np
import random

# XMLをファイルやテキストから読み込んだり、加工したり、保存したりするためのライブラリ
import xml.etree.ElementTree as ET

import cv2

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import torch.utils.data as data
import torchvision
from PIL import Image

class Pipeline():
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, boxes=None, labels=None):
        for transform in self.transforms:
            # 必ず引数が3つ必要
            # 予測時はboxes, labels共にNoneなので、それを前提に作成数するひつつ用あり
            img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels

class Cutout:
    def __init__(self, n_holes=5, length=10):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img, box=None, labels=None):
        w, h = img.size
        img = np.array(img)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.

        for i in range(3):
            img[:, :, i] = img[:, :, i] * mask

        return Image.fromarray(np.uint8(img)), box, labels
    
class Resize:
    def __init__(self, size):
        self.resize = torchvision.transforms.Resize((size, size))
        
    def __call__(self, img, boxes=None, labels=None):
        img = self.resize(img)
        return img, boxes, labels

class ToTensor:
    def __call__(self, img, box=None,labels=None):
        img = torchvision.transforms.ToTensor()(img)
        if not box is None:
            box = torch.tensor(box)
            labels = torch.tensor(labels)
        
        return img, box, labels


class SubtraceMeans:
    def __call__(self, im, boxes=None, labels=None):
        _im = np.array(im)
        _im = _im - _im.mean((0,1))
        return Image.fromarray(np.uint8(_im)), boxes, labels



class Xml2ListConverter():
    def __init__(self, classes):
        self.classes = classes
        
    def __call__(self, xml_path, width, height):
        """
        [[xmin, ymin, xmax, ymax, label]...]
        """
        anno_list = []
        
        xml = ET.parse(xml_path).getroot()
        
        # 難しいもの削除
        for obj in xml.iter("object"):
            
            if int(obj.find('difficult').text) == 1:
                continue
                
            bnd_box = []
            
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')  
            
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            
            for pt in pts:
                #VOCの原点移動
                size = int(bbox.find(pt).text) - 1
                size = size/ height if pt.find("y") > -1 else size / width
                
                bnd_box.append(size)
            
            label_idx = self.classes.index(name)
            bnd_box.append(label_idx)
            anno_list.append(bnd_box)
   
        return np.array(anno_list)