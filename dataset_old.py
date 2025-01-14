import cv2
import numpy as np
import torch
import random
import os
import hashlib

import sys
sys.path.append("yolov3")

from yolov3.utils.dataloaders import LoadImagesAndLabels
from yolov3.utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                        letterbox, mixup, random_perspective)
from yolov3.utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, check_dataset, check_requirements,
                                  check_yaml, clean_str, cv2, is_colab, is_kaggle, segments2boxes, unzip_file, xyn2xy,
                                  xywh2xyxy, xywhn2xyxy, xyxy2xywhn)

from isp.unprocess_np import unprocess_wo_mosaic
from util import AsyncTaskManager


class LoadImagesAndLabelsRAW(LoadImagesAndLabels):
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
                 use_linear=False):
        super(LoadImagesAndLabelsRAW, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                     cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)
        self.add_noise = add_noise
        self.brightness_range = brightness_range
        self.noise_level = noise_level
        self.use_linear = use_linear
        self.train = True if 'train' in prefix else False

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp

        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)
        img = img[..., ::-1]  # BGR to RGB uint8
        img = img / 255.0  # uint8 to float

        # rgb_img = img
        # rgb_img = rgb_img.transpose((2, 0, 1))  # HWC to CHW
        # TODO add unprocess
        if not self.train:
            seed = int(os.path.splitext(os.path.split(self.im_files[index])[1])[0])
            np.random.seed(seed)
        img, _ = unprocess_wo_mosaic(img, self.add_noise, self.brightness_range, self.noise_level, self.use_linear)  # RGB to linear RGB

        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, color=(0, 0, 0), auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            img, labels = random_perspective(img,
                                             labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)

        # raw img: 0-1
        # labels_out = [n, 6]
        # file path
        # shapes: (h0, w0), ((h / h0, w / w0), pad)
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes = batch  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes


class LoadImagesAndLabelsRAWHR(LoadImagesAndLabels):
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
                 use_linear=False):
        super(LoadImagesAndLabelsRAWHR, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                     cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)
        self.add_noise = add_noise
        self.brightness_range = brightness_range
        self.noise_level = noise_level
        self.use_linear = use_linear
        self.train = True if 'train' in prefix else False

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp

        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)
        img = img[..., ::-1]  # BGR to RGB uint8
        img = img / 255.0  # uint8 to float

        # rgb_img = img
        # rgb_img = rgb_img.transpose((2, 0, 1))  # HWC to CHW
        # TODO add unprocess
        if not self.train:
            seed = int(os.path.splitext(os.path.split(self.im_files[index])[1])[0])
            np.random.seed(seed)
        img, _ = unprocess_wo_mosaic(img, self.add_noise, self.brightness_range, self.noise_level, self.use_linear)  # RGB to linear RGB
        img_hr = img.copy()

        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, color=(0, 0, 0), auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            img, labels = random_perspective(img,
                                             labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)

        img_hr = img_hr.transpose((2, 0, 1))  # HWC to CHW
        img_hr = np.ascontiguousarray(img_hr)

        # raw img: 0-1
        # labels_out = [n, 6]
        # file path
        # shapes: (h0, w0), ((h / h0, w / w0), pad)
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes, torch.from_numpy(img_hr)

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes, img_hr = batch  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes, torch.stack(img_hr, 0)
    
    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes, img_hr = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes, torch.stack(img_hr, 0)

    # use original image, but OOM
    # def load_image(self, i):
    #     # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
    #     im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
    #     if im is None:  # not cached in RAM
    #         if fn.exists():  # load npy
    #             im = np.load(fn)
    #         else:  # read image
    #             im = cv2.imread(f)  # BGR
    #             assert im is not None, f'Image Not Found {f}'
    #         h0, w0 = im.shape[:2]  # orig hw
    #         # r = self.img_size / max(h0, w0)  # ratio
    #         # if r != 1:  # if sizes are not equal
    #         #     interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
    #         #     im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
    #         return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    #     return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized


class LoadImagesAndLabelsRAWReplay(LoadImagesAndLabels):
    '''
    rewrite self.indices, get_item

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
                 use_linear=False):
        super(LoadImagesAndLabelsRAWReplay, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                           cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.brightness_range = brightness_range
        self.use_linear = use_linear

    def __getitem__(self, index):
        # TODO must comment this, just use input index
        # index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp

        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)
        img = img[..., ::-1]  # BGR to RGB uint8
        img = img / 255.0  # uint8 to float

        # rgb_img = img
        # rgb_img = rgb_img.transpose((2, 0, 1))  # HWC to CHW
        # TODO add unprocess
        img, _ = unprocess_wo_mosaic(img, self.add_noise, self.brightness_range, self.noise_level, self.use_linear)  # RGB to linear RGB

        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, color=(0, 0, 0), auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            img, labels = random_perspective(img,
                                             labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)

        # raw img: 0-1
        # labels_out = [n, 6]
        # file path
        # shapes: (h0, w0), ((h / h0, w / w0), pad)
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes = batch  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

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

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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
                 use_linear=False):
        self.data_dir = path
        self.img_size = img_size
        self.image_pairs = self._load_image_pairs()
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Resize images to img_size x img_size
            transforms.ToTensor(),  # Convert images to PyTorch tensors (CHW format)
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
        ])
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.image_pairs)
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.brightness_range = brightness_range
        self.use_linear = use_linear

        self.indices = range(self.num_images)

    def _load_image_pairs(self):
        image_pairs = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith("-Input.jpg") or filename.endswith("-Input.png") or filename.endswith("-Input.tif"):
                input_path = os.path.join(self.data_dir, filename)
                label_filename = filename.replace("-Input", "-Target")
                label_path = os.path.join(self.data_dir, label_filename)
                if os.path.exists(label_path):
                    image_pairs.append((input_path, label_path))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        input_path, label_path = self.image_pairs[idx]
        input_image = Image.open(input_path).convert("RGB")
        label_image = Image.open(label_path).convert("RGB")

        # Apply the defined transformations
        input_image = self.transform(input_image)
        label_image = self.transform(label_image)

        # TODO: data aug - optional

        return input_image, label_image, input_path, input_image.shape  # Returns (input_image, label_image, input_path, image_shape)

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes = batch  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

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


class LoadImagesAndLabelsNormalize(LoadImagesAndLabels):
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
                 limit=-1):
        super(LoadImagesAndLabelsNormalize, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                           cache_images, single_cls, stride, pad, min_items, prefix, limit)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, color=(0, 0, 0))
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) / 255.

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes


class LoadImagesAndLabelsNormalizeHR(LoadImagesAndLabels):
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
                 limit=-1):
        super(LoadImagesAndLabelsNormalizeHR, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                           cache_images, single_cls, stride, pad, min_items, prefix, limit)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)
            img_hr = img.copy()

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, color=(0, 0, 0))
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) / 255.

        img_hr = img_hr.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_hr = np.ascontiguousarray(img_hr) / 255.

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes, torch.from_numpy(img_hr)
    
    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes, img_hr = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes, torch.stack(img_hr, 0)

    # use original image, but OOM
    # def load_image(self, i):
    #     # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
    #     im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
    #     if im is None:  # not cached in RAM
    #         if fn.exists():  # load npy
    #             im = np.load(fn)
    #         else:  # read image
    #             im = cv2.imread(f)  # BGR
    #             assert im is not None, f'Image Not Found {f}'
    #         h0, w0 = im.shape[:2]  # orig hw
    #         # r = self.img_size / max(h0, w0)  # ratio
    #         # if r != 1:  # if sizes are not equal
    #         #     interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
    #         #     im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
    #         return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    #     return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized


class LoadImagesAndLabelsNormalizeReplay(LoadImagesAndLabels):
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
                 limit=-1):
        super(LoadImagesAndLabelsNormalizeReplay, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                                 cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)

    def __getitem__(self, index):
        # TODO must comment this, just use input index
        # index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, color=(0, 0, 0))
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) / 255.

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes = batch  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

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


def restore_image(image, ori_image):
    ih, iw, _ = image.shape
    if isinstance(ori_image, (tuple, list)):
        h, w, _ = ori_image
    else:
        h, w, _ = ori_image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    dst_img = image[dh:dh + nh, dw:dw + nw, ::]
    # print(dst_img.shape)
    dst_img = cv2.resize(dst_img, (w, h))
    # print(dst_img.shape)
    # print(scale, dw, dh, nw, nh)
    return dst_img


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
    val_path = '/home/PJLAB/wangyujin/HDD/projects/isp/datasets/COCO/coco2017/val2017.txt'
    dataset = LoadImagesAndLabelsRAWReplay(
        val_path,
        imgsz,
        batch_size,
        augment=False,  # augmentation
        limit=1000
    )
    print(len(dataset))
    print(dataset.get_next_batch_(batch_size))
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


