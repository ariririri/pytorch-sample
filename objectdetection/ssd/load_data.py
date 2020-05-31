from preprocessor import Xml2ListConverter, Resize, SubtraceMeans, ToTensor, Cutout, Pipeline
from pathlib import Path
import torch.utils.data as data
from PIL import Image

import torch

voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

def load_data(root_dir):
    train_img_list, train_ann_list, val_img_list, val_ann_list = get_datapath_list(root_dir)
    xml_converter = Xml2ListConverter(voc_classes)

    input_size = 300
    train_pipline = Pipeline([
        Cutout(),
        Resize(input_size),
        SubtraceMeans(),
        ToTensor()
        ])
    
    valid_pipline = Pipeline([
        Resize(input_size),
        SubtraceMeans(),
        ToTensor()
    ])

    train_dataset = VOCDataset(train_img_list, train_ann_list, transform=train_pipline, transform_ann=xml_converter)
    val_dataset = VOCDataset(val_img_list, val_ann_list, transform=valid_pipline, transform_ann=xml_converter)

    return train_dataset, val_dataset


def get_datapath_list(root_dir):
    
    root_dir = Path(root_dir)
    img_dir = root_dir / "JPEGImages"
    ann_dir = root_dir / "Annotations"
    
    train_id_name_txt = root_dir / "ImageSets" / "Main" / "train.txt"
    valid_id_name_txt = root_dir / "ImageSets" / "Main" / "val.txt"
    
    train_img_list = []
    train_ann_list = []
    
    for line in open(train_id_name_txt):
        file_id = line.strip()
        train_img_list.append(img_dir / f"{file_id}.jpg")
        train_ann_list.append(ann_dir / f"{file_id}.xml")
    
    valid_img_list = []
    valid_ann_list = []
    
    for line in open(valid_id_name_txt):
        file_id = line.strip()
        valid_img_list.append(img_dir / f"{file_id}.jpg")
        valid_ann_list.append(ann_dir / f"{file_id}.xml")

    return train_img_list, train_ann_list, valid_img_list, valid_ann_list

class VOCDataset(data.Dataset):
    def __init__(self, img_list, ann_list, transform, transform_ann):
        self.img_list = img_list
        self.ann_list = ann_list
        self.transform = transform # trainとvalidで異なる
        self.transform_ann = transform_ann
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        
        img = Image.open(self.img_list[index])
        h, w = img.size
        
        ann_path = self.ann_list[index]
        ann_list = self.transform_ann(ann_path, h, w)
        # 4はlabel name
        
        img, boxes, labels = self.transform(img, ann_list[:, :4], ann_list[:, 4])
        
        ans = torch.cat((boxes, labels.view(-1, 1)), axis=1)
        
        return img, ans 