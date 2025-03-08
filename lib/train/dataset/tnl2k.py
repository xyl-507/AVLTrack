import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data.image_loader import default_image_loader
from lib.train.admin.environment import env_settings
from lib.test.utils.load_text import load_text
import cv2
from PIL import Image

class TNL2K(BaseVideoDataset):

    def __init__(self, root=None, image_loader=default_image_loader, split=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
        """
        self.root = env_settings().tnl2k_dir if root is None else root
        super().__init__('TNL2K', root, image_loader)

        self.sequence_list = self._build_sequence_list(split)

    def _build_sequence_list(self, split=None):
        if split is not None:
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = self.root
            else:
                raise ValueError('Unknown split name.')
            sequence_list = os.listdir(file_path)
            random.shuffle(sequence_list)
        else:
            raise ValueError('Set split_name.')

        return sequence_list

    def get_name(self):
        return 'TNL2K'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_anno(self, seq_path):
        anno_file = os.path.join(self.root, seq_path, "groundtruth.txt")
        # gt = pandas.read_csv(anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        # NOTE: OTB has some weird annos which panda cannot handle
        gt = load_text(str(anno_file), delimiter=(',', None), dtype=np.float64, backend='numpy')
        return torch.tensor(gt)

    # def _read_target_visible(self, seq_path, anno):
    #     # Read full occlusion and out_of_view
    #     occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
    #     out_of_view_file = os.path.join(seq_path, "out_of_view.txt")
    #
    #     with open(occlusion_file, 'r', newline='') as f:
    #         occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
    #     with open(out_of_view_file, 'r') as f:
    #         out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
    #
    #     target_visible = ~occlusion & ~out_of_view & (anno[:,2]>0) & (anno[:,3]>0)
    #
    #     return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        # class_name = seq_name.split('-')[0]

        return os.path.join(self.root, seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        images_path = sorted(os.listdir(os.path.join(seq_path, 'imgs')))
        image_path = os.path.join(seq_path, 'imgs', images_path[frame_id])
        return image_path

    def _get_frame(self, seq_path, frame_id):
        img = Image.open(self._get_frame_path(seq_path, frame_id))
        img = np.array(img)
        if len(img.shape)<3:   # gray to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def _get_class(self, seq_path):
        obj_class = seq_path.split('/')[-1]
        return obj_class

    ###############################################################
    def _read_language(self, seq_path):
        try:
            language_file = os.path.join(seq_path, "language.txt")
            with open(language_file, 'r') as file:
                language = file.readline().replace(".", "").replace("\n", "").lower()
            return language
        except:
            return None
        # language = None
        # return language
    ###############################################################

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        language = self._read_language(seq_path)  # e.g., baskball on a boy's hand
        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self._read_anno(seq_path)

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...] for _ in frame_ids]

        #########################################################
        ## Add Language
        object_meta = OrderedDict({'object_class': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None,
                                   'language': language})

        return frame_list, anno_frames, object_meta



