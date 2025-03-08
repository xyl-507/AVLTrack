import os
# from .base_video_dataset import BaseVideoDataset
from .base_image_dataset import BaseImageDataset
from lib.train.data.image_loader import jpeg4py_loader
import torch
import random
from collections import OrderedDict
from lib.train.admin.environment import env_settings
import json
import cv2
from PIL import Image
import numpy as np

class VisualGenome(BaseImageDataset):

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, split="train", version="2014"):

        root = env_settings().coco_dir if root is None else root
        super().__init__('VisualGenome', root, image_loader)

        self.img_pth = os.path.join(root, 'images', 'VG_100K')
        self.anno_path = os.path.join(root, 'region_descriptions.json')

        with open(self.anno_path, 'r') as openfile:
            # Reading from json file
            self.anno = json.load(openfile)

        self.image_list, self.anno_list, self.phrase_list = self._get_sequence_list()

        pass


    def _get_sequence_list(self):
        image_list = []
        anno_list = []
        phrase_list = []
        for i in range(len(self.anno)):
        # for i in range(1):
            regions = self.anno[i]['regions']
            for region in regions:
                bbox = [region['x'],
                        region['y'],
                        region['width'],
                        region['height']
                        ]    # x_left_top, y_left_top, w, h

                image_list.append(region['image_id'])
                anno_list.append(bbox)
                phrase_list.append(region['phrase'])

        return image_list, anno_list, phrase_list


    def is_video_sequence(self):
        return False

    def get_num_classes(self):
        return len(self.class_list)

    def get_name(self):
        return 'coco'

    def has_class_info(self):
        return True

    def get_class_list(self):
        class_list = []
        for cat_id in self.cats.keys():
            class_list.append(self.cats[cat_id]['name'])
        return class_list

    def has_segmentation_info(self):
        return True

    # def get_num_sequences(self):
    #     return len(self.sequence_list)
    def get_num_sequences(self):
        return len(self.image_list)

    def get_sequence_info(self, seq_id):
        anno = self._get_anno(seq_id)
        bbox = np.array([anno] + [anno])
        bbox = torch.tensor(bbox)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_anno(self, im_id):
        anno = self.anno_list[im_id]

        return anno

    def _get_frames(self, im_id):
        path = str(self.image_list[im_id])+".jpg"
        img = self.image_loader(os.path.join(self.img_pth, path))
        return img

    def get_meta_info(self, seq_id):
        language = self.phrase_list[seq_id].replace(".", "").replace("\n", "").lower()
        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None,
                                   'language': language
                                   })
        return object_meta

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        # Visual Genome is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in range(2)]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...] for _ in frame_ids]

        object_meta = self.get_meta_info(seq_id)

        return frame_list, anno_frames, object_meta
