import numbers
import os
import queue as Queue
import threading
import json
import cv2
import random


import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance

from augment.blur import DefocusBlur, MotionBlur


def adjust_brightness_contrast(original):
    # Randomly generate brightness and contrast values
    brightness = random.randint(95, 350)
    contrast = random.randint(160, 180)
    
    # Convert brightness and contrast values to appropriate ranges
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
    
    # Adjust brightness
    img = original.copy()
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max = 255
        else:
            shadow = 0
            max = 255 + brightness
        al_pha = (max - shadow) / 255
        ga_mma = shadow
        img = cv2.addWeighted(img, al_pha, img, 0, ga_mma)
    
    # Adjust contrast
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
        img = cv2.addWeighted(img, Alpha, img, 0, Gamma)
    
    return img


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch



class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)

def random_crop(image, ratio = 0.07):
    pct_focusx = random.uniform(0, ratio)
    pct_focusy = random.uniform(0, ratio)
    x, y = image.size
    image = image.crop((x*pct_focusx, y*pct_focusy, x*(1-pct_focusx), y*(1-pct_focusy)))
    
    return image

class FaceDataset(Dataset):
    def __init__(self, root_dir, json_path):
        super(FaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5,saturation=0.3, hue=0.2 ),
            transforms.RandomResizedCrop(size=(112,112),
                                scale=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        
        self.root_dir = root_dir
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __getitem__(self, index):
        data_item = self.data[str(index)]
        img = cv2.imread(os.path.join(self.root_dir, data_item['path']))
        choice = random.uniform(0,1)
        if choice > 0.4:
            img = adjust_brightness_contrast(img)

        # img_ = Image.fromarray(img)
        
        # # add Augment
        # img_ = random_crop(img_)

        # # Random sharpen
        # choice = random.uniform(0,1)
        # if choice > 0.7:
        #     img3 = ImageEnhance.Sharpness(img_)
        #     img_ = img3.enhance(random.randint(-2,2))

        # # defocus blur random
        # choice = random.uniform(0,1)
        # choice_mag = random.randint(0,4)
        # if choice > 0.7:
        #     img_ = DefocusBlur()(img_, mag=choice_mag)
        
        # choice = random.uniform(0,1)
        # choice_mag = random.randint(0,3)
        # if choice > 0.7 :
        #     img_ = MotionBlur()(img_, mag=choice_mag)

        # img = np.asarray(img_)

        # end augment
        sample = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        label = int(data_item['labels'])
        label = torch.tensor(label, dtype=torch.long)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.data)

class FaceDataset_2class(Dataset):
    def __init__(self, root_dir, json_path):
        super(FaceDataset_2class, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,saturation=0.3, hue=0.2 ),
            transforms.RandomResizedCrop(size=(112,112),
                                scale=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        
        self.root_dir = root_dir
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __getitem__(self, index):
        data_item = self.data[str(index)]
        img = cv2.imread(os.path.join(self.root_dir, data_item['path']))

        # end augment

        label = int(data_item['labels'])
        
        if label == 0:
            
            img_ = Image.fromarray(img)
        
            # add Augment
            img_ = random_crop(img_)

            # defocus blur random
            choice = random.uniform(0,1)
            choice_mag = random.randint(0,3)
            if choice > 0.7:
                img_ = DefocusBlur()(img_, mag=choice_mag)
            
            choice = random.uniform(0,1)
            choice_mag = random.randint(0,3)
            if choice > 0.7 :
                img_ = MotionBlur()(img_, mag=choice_mag)
            img = np.asarray(img_)
 

        sample = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.data)
