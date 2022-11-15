import os
import random
import copy

import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed
import glob
from .utils import Subset, filter_images, group_images

# Converting the id to the train_id. Many objects have a train id at
# 255 (unknown / ignored).
# See there for more information:
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
id_to_trainid = {
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 1,   # road
    8: 2,   # sidewalk
    9: 255,
    10: 255,
    11: 3,  # building
    12: 4,  # wall
    13: 5,  # fence
    14: 255,
    15: 255,
    16: 255,
    17: 6,  # pole
    18: 255,
    19: 7,  # traffic light
    20: 8,  # traffic sign
    21: 9,  # vegetation
    22: 10,  # terrain
    23: 11, # sky
    24: 12, # person
    25: 13, # rider
    26: 14, # car
    27: 15, # truck
    28: 16, # bus
    29: 255,
    30: 255,
    31: 17, # train
    32: 18, # motorcycle
    33: 19, # bicycle
    -1: 255
}

class CityscapeSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, train=True, is_aug=True, transform=None):

        self.root = os.path.expanduser(root)
        annotation_folder = os.path.join(self.root, 'gtFine')
        image_folder = os.path.join(self.root, 'leftImg8bit')

        self.train = train
        self.transform = transform
        if self.train:
            self.images = [  # Add train cities
                (
                    path,
                    os.path.join(
                        annotation_folder,
                        "train",
                        path.split("/")[-2],
                        path.split("/")[-1][:-15] + "gtFine_labelIds.png"
                    )
                ) for path in sorted(glob.glob(os.path.join(image_folder, "train/*/*.png")))
            ]
        else:
            self.images = [  # Add validation cities
                (
                    path,
                    os.path.join(
                        annotation_folder,
                        "val",
                        path.split("/")[-2],
                        path.split("/")[-1][:-15] + "gtFine_labelIds.png"
                    )
                ) for path in sorted(glob.glob(os.path.join(image_folder, "val/*/*.png")))
            ]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        #for idx, map_id in id_to_trainid.items():
        #    target[target == idx] = map_id
        
        return img, target

    def viz_getter(self, index):
        try:
            img = Image.open(self.images[index][0]).convert('RGB')
            target = Image.open(self.images[index][1])
        except Exception as e:
            raise Exception(f"Index: {index}, len: {len(self)}, message: {str(e)}")

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class CityscapeSegmentationIncremental(data.Dataset):

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        labels=None,
        labels_old=None,
        idxs_path=None,
        masking=True,
        overlap=True,
        data_masking="current",
        test_on_val=False,
        **kwargs
    ):

        full_data = CityscapeSegmentation(root, train)

        self.labels = []
        self.labels_old = []
        
        # filter images
        cls2img = [[] for i in range(20)]
        for i in range(len(full_data)):
            cls = np.unique(np.array(full_data[i][1]))
            for k in range(cls.shape[0]):
                cls[k] = id_to_trainid[cls[k]]
                if cls[k] < 20:
                    cls2img[cls[k]].append(i)
        
        cls_order = [17, 16, 15, 18, 4, 13, 5, 7, 9, 1, 2, 3, 6, 8, 10, 11, 12, 14, 19]
        mp_used = [0 for i in range(len(full_data))]
        per_cls_idx = [[] for i in range(20)]
        
        print('start select!')
        for cls_num in cls_order:
            selected = 0
            for kk in cls2img[cls_num]:
                if mp_used[kk] == 1:
                    continue
                selected += 1
                mp_used[kk] = 1
                per_cls_idx[cls_num].append(kk)                
                if cls_num != 19 and selected >= 150:
                    break
        
        print('select done!')

        if labels is not None:
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(
                l in labels_old for l in labels
            ), "labels and labels_old must be disjoint sets"

            self.labels = [0] + labels
            self.labels_old = [0] + labels_old

            self.order = [0] + labels_old + labels

            # take index of images with at least one class in labels and all classes in labels+labels_old+[0,255]
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = []
                for cls in labels:
                    idxs = idxs + per_cls_idx[cls]
                    
                if idxs_path is not None and distributed.get_rank() == 0:
                    np.save(idxs_path, np.array(idxs, dtype=int))

            if test_on_val:
                rnd = np.random.RandomState(1)
                rnd.shuffle(idxs)
                train_len = int(0.8 * len(idxs))
                if train:
                    idxs = idxs[:train_len]
                else:
                    idxs = idxs[train_len:]

            #if train:
            #    masking_value = 0
            #else:
            #    masking_value = 255

            #self.inverted_order = {label: self.order.index(label) for label in self.order}
            #self.inverted_order[255] = masking_value

            masking_value = 0  # Future classes will be considered as background.
            self.inverted_order = {label: self.order.index(label) for label in self.order}
            self.inverted_order[255] = 255
            self.inverted_order[-1] = 255

            reorder_transform = tv.transforms.Lambda(
                lambda t: t.apply_(
                    lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value
                )
            )

            if masking:
                if data_masking == "current":
                    tmp_labels = self.labels + [255]
                elif data_masking == "current+old":
                    tmp_labels = labels_old + self.labels + [255]
                elif data_masking == "all":
                    raise NotImplementedError(
                        f"data_masking={data_masking} not yet implemented sorry not sorry."
                    )
                elif data_masking == "new":
                    tmp_labels = self.labels
                    masking_value = 255

                target_transform = tv.transforms.Lambda(
                    lambda t: t.
                    apply_(lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value)
                )
            else:
                assert False
                target_transform = reorder_transform

            # make the subset of the dataset
            self.dataset = Subset(full_data, idxs, transform, target_transform, id_to_trainid)
        else:
            self.dataset = full_cityscape

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    def viz_getter(self, index):
        return self.dataset.viz_getter(index)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)
