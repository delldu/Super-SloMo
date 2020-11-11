"""Data loader."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:54:18 CST
# ***
# ************************************************************************************/
#

import os

import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.utils as utils
from PIL import Image

import pdb

train_dataset_rootdir = "dataset/train/"
test_dataset_rootdir = "dataset/test/"

VIDEO_SEQUENCE_LENGTH = 2 # for Video Slow

def get_transform(train=True):
    """Transform images."""
    ts = []
    # if train:
    #     ts.append(T.RandomHorizontalFlip(0.5))
    mean = [0.429, 0.431, 0.397]
    std = [1, 1, 1]
    normalize = T.Normalize(mean=mean,std=std)
    ts.append(T.ToTensor())

    ts.append(normalize)

    return T.Compose(ts)

def reverse_transform():
    ts = []
    mean = [-0.429, -0.431, -0.397]
    std = [1, 1, 1]
    normalize = T.Normalize(mean=mean, std=std)
    ts.append(normalize)

    ts.append(T.ToPILImage())
    
    return T.Compose(ts)

def multiple_crop(data, mult=32, HWmax=[4096, 4096]):
    # crop image to a multiple
    H, W = data.shape[1:]
    Hnew = min(int(H/mult)*mult, HWmax[0])
    Wnew = min(int(W/mult)*mult, HWmax[1])
    h = (H-Hnew)//2
    w = (W-Wnew)//2
    return data[:, h:h+Hnew, w:w+Wnew]


class Video(data.Dataset):
    """Define Video Frames Class."""

    def __init__(self, seqlen=VIDEO_SEQUENCE_LENGTH, transforms=get_transform()):
        """Init dataset."""
        super(Video, self).__init__()
        self.seqlen = seqlen
        self.transforms = transforms
        self.root = ""
        self.images = []
        self.height = 0
        self.width = 0

    def reset(self, root):
        # print("Video Reset Root: ", root)
        self.root = root
        self.images = list(sorted(os.listdir(root)))

        # Suppose the first image size is video frame size
        if len(self.images) > 0: 
            filename = os.path.join(self.root, self.images[0])
            img = self.transforms(Image.open(filename).convert("RGB"))
            img = multiple_crop(img, mult=32)
            C, H, W = img.size()
            self.height = H
            self.width = W

    def __getitem__(self, idx):
        """Load images."""
        n = len(self.images)
        filelist = []
        delta = (self.seqlen - 1)/2
        for k in range(-int(delta), int(delta + 0.5) + 1):
            if (idx + k < 0):
                filename = self.images[0]
            elif (idx + k >= n):
                filename = self.images[n - 1]
            else:
                filename = self.images[idx + k]
            filelist.append(os.path.join(self.root, filename))
        # print("filelist: ", filelist)
        sequence = []
        for filename in filelist:
            img = Image.open(filename).convert("RGB")
            img = self.transforms(img)
            img = multiple_crop(img)
            C, H, W = img.size()
            img = img.view(1, C, H, W)
            sequence.append(img)
        return torch.cat(sequence, dim=0)

    def __len__(self):
        """Return total numbers of images."""
        return len(self.images)


class VideoSlowDataset(data.Dataset):
    """Define dataset."""

    def __init__(self, root, seqlen=VIDEO_SEQUENCE_LENGTH, transforms=get_transform()):
        """Init dataset."""
        super(VideoSlowDataset, self).__init__()

        self.root = root
        self.seqlen = seqlen
        self.transforms = transforms

        # load all images, sorting for alignment
        self.images = []
        # index start offset
        self.indexs = []
        offset = 0
        ds = list(sorted(os.listdir(root)))
        for d in ds:
            fs = sorted(os.listdir(root + "/" + d))
            for f in fs:
                self.images.append(d + "/" + f)
                self.indexs.append(offset)
            offset += len(fs)
        self.video_cache = Video(seqlen=seqlen, transforms=transforms)

    def __getitem__(self, idx):
        """Load images."""
        # print("dataset index:", idx)
        image_path = os.path.join(self.root, self.images[idx])
        if (self.video_cache.root != os.path.dirname(image_path)):
            self.video_cache.reset(os.path.dirname(image_path))
        return self.video_cache[idx - self.indexs[idx]]

    def __len__(self):
        """Return total numbers of images."""
        return len(self.images)

    def __repr__(self):
        """
        Return printable representation of the dataset object.
        """
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms: '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def train_data(bs):
    """Get data loader for trainning & validating, bs means batch_size."""

    train_ds = VideoSlowDataset(
        train_dataset_rootdir, VIDEO_SEQUENCE_LENGTH, get_transform(train=True))
    print(train_ds)

    # Split train_ds in train and valid set
    valid_len = int(0.2 * len(train_ds))
    indices = [i for i in range(len(train_ds) - valid_len, len(train_ds))]

    valid_ds = data.Subset(train_ds, indices)
    indices = [i for i in range(len(train_ds) - valid_len)]
    train_ds = data.Subset(train_ds, indices)

    # Define training and validation data loaders
    train_dl = data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=4)
    valid_dl = data.DataLoader(
        valid_ds, batch_size=bs, shuffle=False, num_workers=4)

    return train_dl, valid_dl


def test_data(bs):
    """Get data loader for test, bs means batch_size."""

    test_ds = VideoSlowDataset(
        test_dataset_rootdir, VIDEO_SEQUENCE_LENGTH, get_transform(train=False))
    test_dl = data.DataLoader(test_ds, batch_size=bs,
                              shuffle=False, num_workers=4)

    return test_dl


def get_data(trainning=True, bs=4):
    """Get data loader for trainning & validating, bs means batch_size."""

    return train_data(bs) if trainning else test_data(bs)


def VideoSlowDatasetTest():
    """Test dataset ..."""

    ds = VideoSlowDataset(train_dataset_rootdir)
    print(ds)
    vs = Video()
    vs.reset("dataset/predict/input")
    print("Video frame Size: HxW = {:3d} x {:3d}".format(vs.height, vs.width))
    print("First Frame Size:", vs[0].size())

if __name__ == '__main__':
    VideoSlowDatasetTest()
