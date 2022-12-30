import cv2
import os.path as osp
import numpy as np
import random
import torchvision
from torch.utils import data


class ISPRSDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(512, 512),
                 mean=(128, 128, 128), scale=False,
                 mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, "train/rgb/%s.tif" % name)
            dsm_file = osp.join(self.root, "train/dsmnpy/%s.npy" % name)
            self.files.append({
                "img": img_file,
                "dsm": dsm_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, dsm):
        f_scale = 0.5 + random.randint(0, 16) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        dsm = cv2.resize(dsm, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        return image, dsm

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        dsm = np.load(datafiles["dsm"])
        dsm = np.reshape(dsm, (512, 512, 1))

        size = image.shape
        name = datafiles["name"]

        if self.scale:
            image, dsm = self.generate_scale_label(image, dsm)

        image = np.asarray(image, np.float32)
        image -= self.mean

        dsm = np.asarray(dsm, np.float32)
        dsm = dsm / 183.17412

        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            dsm_pad = cv2.copyMakeBorder(dsm, 0, pad_h, 0,
                                       pad_w, cv2.BORDER_CONSTANT,
                                       value=(0.0, 0.0, 0.0))
        else:
            img_pad, dsm_pad = image, dsm

        img_h, img_w, _ = img_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        dsm = np.asarray(dsm_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        image = image.transpose((2, 0, 1))
        dsm = dsm.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            dsm = dsm[:, :, ::flip]

        return image.copy(), dsm.copy(), np.array(size), name



class ISPRSDataValSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, mean=(128, 128, 128), scale=False, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "val/rgb/%s.tif" % name)
            dsm_file = osp.join(self.root, "val/dsmnpy/%s.npy" % name)
            self.files.append({
                "img": img_file,
                "dsm": dsm_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        return image

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)

        size = image.shape
        dsm = np.load(datafiles["dsm"])
        dsm = np.reshape(dsm, (size[0], size[1], 1))

        name = datafiles["name"]
        if self.scale:
            image = self.generate_scale_label(image)
        image = np.asarray(image, np.float32)
        image -= self.mean

        dsm = np.asarray(dsm, np.float32)

        image = image.transpose((2, 0, 1))
        dsm = dsm.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            dsm = dsm[:, ::flip]

        return image.copy(), dsm.copy(), np.array(size), name







