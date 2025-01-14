import cv2
import numpy as np
import torch
import random
import os
import hashlib
import sys
import glob
import json

from isp.unprocess_np import unprocess_wo_mosaic
from util import AsyncTaskManager

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataset_old import LoadImagesAndLabelsRAW, LoadImagesAndLabelsNormalize, \
    LoadImagesAndLabelsNormalizeHR, LoadImagesAndLabelsRAWHR, LoadImagesAndLabelsRAWReplay, LoadImagesAndLabelsNormalizeReplay

class LoadImagesAndLabelsRAWReplay_target(Dataset):
    '''
    rewrite self.indices, get_item
    do we need data augmentation? in low level vision?
    - do exposure have data aug -- have but not used
    '''
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1,
                 add_noise=False,
                 brightness_range=None,
                 noise_level=None,
                 use_linear=False,
                 highres_eval=False,
                 ):
        self.data_dir = path
        self.img_size = img_size
        print(f'[ dataset ] before loading img pairs')
        if highres_eval:
            self.image_pairs = self._load_image_pairs()
        else:
            self.image_pairs = self._load_image_pairs_from_file()
        print(f'[ dataset ] after loading img pairs')
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Resize images to img_size x img_size
            transforms.ToTensor(),  # Convert images to PyTorch tensors (CHW format)
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
        ])
        self.synchronous = False
        self.batch_size = batch_size
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.image_pairs)
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.brightness_range = brightness_range
        self.use_linear = use_linear
        self.highres_eval = highres_eval

        self.indices = range(self.num_images)

        print(f'[ dataset ] LoadImagesAndLabelsRAWReplay_target load from :', self.data_dir)
        print(f'[ dataset ] [===== loaded {self.num_images} images ====] :')

    def _load_image_pairs(self):
        image_pairs = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith("-Input.jpg") or filename.endswith("-Input.png") or filename.endswith("-Input.tif"):
                input_path = os.path.join(self.data_dir, filename)
                label_filename = filename.replace("-Input", "-Target")
                label_path = os.path.join(self.data_dir, label_filename)
                label_path_png = label_path.replace(".tif", ".png")
                if os.path.exists(label_path):
                    image_pairs.append((input_path, label_path))
                elif os.path.exists(label_path_png):
                    image_pairs.append((input_path, label_path_png))
        return image_pairs

    def _load_image_pairs_(self):
        image_pairs = []
        # Find all input files with specified suffixes in the directory
        for input_pattern in ["*-Input.jpg"]:
            for input_path in glob.iglob(os.path.join(self.data_dir, input_pattern)):
                print(".", end="")
                base_filename = os.path.basename(input_path)
                label_filename = base_filename.replace("-Input", "-Target")
                label_path = os.path.join(self.data_dir, label_filename)
                if os.path.exists(label_path):
                    image_pairs.append((input_path, label_path))
        return image_pairs

    def _load_image_pairs__(self):
        image_pairs = []
        # Desired suffixes for input files
        input_suffixes = ('-Input.jpg', '-Input.png', '-Input.tif')

        # Use scandir to iterate through files in the directory
        print(self.data_dir)
        with os.scandir(self.data_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith('-Input.jpg'):
                    print(".", end="")
                    input_path = os.path.join(self.data_dir, entry.name)
                    # Construct the corresponding label path
                    label_filename = entry.name.replace("-Input", "-Target")
                    label_path = os.path.join(self.data_dir, label_filename)
                    # Check if the corresponding label file exists
                    if os.path.exists(label_path):
                        image_pairs.append((input_path, label_path))

        return image_pairs

    def _load_image_pairs_from_file(self):
        with open(os.path.join(self.data_dir, "image_pairs"), 'r') as f:
            image_pairs = json.load(f)
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        input_path, label_path = self.image_pairs[idx]
        input_image = Image.open(input_path).convert("RGB")
        label_image = Image.open(label_path).convert("RGB")

        # Apply the defined transformations
        if self.highres_eval:
            H, W = input_image.size
            transform = transforms.Compose([
                transforms.Resize((W // 4, H // 4)),  # Resize images to img_size x img_size
                transforms.ToTensor(),  # Convert images to PyTorch tensors (CHW format)
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
            ])
            input_image = transform(input_image)
            label_image = transform(label_image)
        else:
            input_image = self.transform(input_image)
            label_image = self.transform(label_image)

        # TODO: data aug - optional

        return input_image, label_image, input_path, input_image.shape  # Returns (input_image, label_image, input_path, image_shape)

    def get_next_batch_(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            s = min(len(self.indices), batch_size - len(batch))
            batch += self.indices[:s]
            self.indices = self.indices[s:]
            if len(self.indices) == 0:
                self.indices = list(range(self.num_images))
                random.shuffle(self.indices)
        im_list = []
        label_list = []
        path_list = []
        shapes_list = []
        for i in range(len(batch)):
            im, label, path, shapes = self.__getitem__(batch[i])
            im_list.append(im)
            label_list.append(label)
            path_list.append(path)
            shapes_list.append(shapes)
        # return self.collate_fn_raw([im_list, label_list, path_list, shapes_list])
        return im_list, label_list, path_list, shapes_list

    def get_next_batch(self, batch_size):
        if self.synchronous or (self.async_task and batch_size != self.default_batch_size):
            return self.get_next_batch_(batch_size)
        else:
            if self.async_task is None:
                self.async_task = AsyncTaskManager(target=self.get_next_batch_, args=(self.default_batch_size,))
            if batch_size != self.default_batch_size:
                ret = self.get_next_batch_(batch_size)
            else:
                ret = self.async_task.get_next()
            return ret

    def get_next_batch_det(self, batch_size):
        """
        deterministic batch sampler, used for evaluation set loading
        keep is_last
        todo: it seems to stuck with one image at the end
        """
        batch = []
        is_last = False
        while len(batch) < batch_size:
            s = min(len(self.indices), batch_size - len(batch))
            if len(self.indices) - batch_size < batch_size:
                is_last = True
            batch += self.indices[:s]
            self.indices = self.indices[s:]
            if is_last or len(self.indices) == 0:
                self.indices = list(range(self.num_images))
        im_list = []
        label_list = []
        path_list = []
        shapes_list = []
        for i in range(len(batch)):
            im, label, path, shapes = self.__getitem__(batch[i])
            im_list.append(im)
            label_list.append(label)
            path_list.append(path)
            shapes_list.append(shapes)
        return im_list, label_list, path_list, shapes_list, is_last


class LoadImagesRetouchTargets(Dataset):
    '''
    rewrite self.indices, get_item
    do we need data augmentation? in low level vision?
    - do exposure have data aug -- have but not used
    '''
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16):
        self.data_dir = path
        self.img_size = img_size
        self.image_pairs = self._load_image_pairs()
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Resize images to img_size x img_size
            transforms.ToTensor(),  # Convert images to PyTorch tensors (CHW format)
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
        ])
        self.synchronous = False
        self.batch_size = batch_size
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.image_pairs)

        self.indices = range(self.num_images)

        print(f'[ dataset ] LoadImagesRetouchTarget load from :', self.data_dir)
        print(f'[ dataset ] [===== loaded {self.num_images} images ====] :')

    def _load_image_pairs(self):
        image_pairs = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith("-Input.jpg") or filename.endswith("-Input.png") or filename.endswith("-Input.tif"):
                input_path = os.path.join(self.data_dir, filename)
                label_filename = filename.replace("-Input", "-Target")
                label_path = os.path.join(self.data_dir, label_filename)
                retouch_filename = filename.replace("-Input", "-Render")
                retouch_path = os.path.join(os.path.abspath(os.path.join(self.data_dir, "../source")), retouch_filename)
                if os.path.exists(label_path) and os.path.exists(retouch_path):
                    image_pairs.append((input_path, label_path, retouch_path))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        input_path, label_path, retouch_path = self.image_pairs[idx]
        input_image = Image.open(input_path).convert("RGB")
        label_image = Image.open(label_path).convert("RGB")
        retouch_image = Image.open(retouch_path).convert("RGB")

        # Apply the defined transformations
        input_image = self.transform(input_image)
        label_image = self.transform(label_image)
        retouch_image = self.transform(retouch_image)

        # TODO: data aug - optional

        return input_image, label_image, retouch_image, input_path  # Returns (input_image, label_image, input_path, image_shape)



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def show(x, title="a", format="HWC", is_last=True):
        if format == 'CHW':
            x = np.transpose(x, (1, 2, 0))
        plt.figure()
        plt.cla()
        plt.title(title)
        plt.imshow(x)
        if is_last:
            plt.show()
    data_dict = {'path': '/home/PJLAB/wangyujin/HDD/projects/isp/datasets/COCO/coco2017',
     'train': '/home/PJLAB/wangyujin/HDD/projects/isp/datasets/COCO/coco2017/train2017.txt',
     'val': '/home/PJLAB/wangyujin/HDD/projects/isp/datasets/COCO/coco2017/val2017.txt',
     'test': '/home/PJLAB/wangyujin/HDD/projects/isp/datasets/COCO/coco2017/test-dev2017.txt',
     'names': {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
               8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
               14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
               22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
               29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
               35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
               40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
               48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
               55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
               62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
               69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
               76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'},
     'download': "from utils.general import download, Path\n\n\n# Download labels\nsegments = False  # segment or box labels\ndir = Path(yaml['path'])  # dataset root dir\nurl = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'\nurls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels\ndownload(urls, dir=dir.parent)\n\n# Download data\nurls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images\n        'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images\n        'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)\ndownload(urls, dir=dir / 'images', threads=3)\n",
     'nc': 80}

    batch_size = 4
    imgsz = 512
    val_path = '/mnt/data/sail_3090/wujiarui/data/distilled_jpg_pair/val'
    dataset = LoadImagesAndLabelsRAWReplay_target(
        val_path,
        imgsz,
        batch_size,
        augment=False,  # augmentation
        limit=1000
    )
    print(len(dataset))
    print(dataset.get_next_batch_(batch_size)[1][0].shape)
    exit()

    # from PIL import Image
    # img = Image.open("/home/PJLAB/wangyujin/HDD/projects/isp/datasets/COCO/coco2017/images/val2017/000000000139.jpg")
    # img.show()
    # for x in dataset:
    #     print(len(x))
    #     print(x[0].shape, x[1].shape, x[2], x[3])
    #     print(x[0].numpy().shape)
    #     # show(x[0].numpy(), format="CHW", is_last=False)
    #     # show(x[1].numpy(), format="CHW", is_last=True)
    #     # torch.from_numpy(img), labels_out, self.im_files[index], shapes
    #     break

    # from torch.utils.data import DataLoader
    # loader = DataLoader(dataset, batch_size=batch_size, collate_fn=LoadImagesAndLabelsRAW.collate_fn)
    # for x in loader:
    #     print(x)
    #     break

    for i in range(10):
        print(dataset.get_next_batch(batch_size))
        if i > 3:
            break


