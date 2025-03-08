import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os

class OTB99LANGDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.otb99lang_path   ###############
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            # cls, _ = self.sequence_list[i].split('-')
            cls = self.sequence_list[i]
            clean_lst.append(cls)
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name
        # anno_path = '{}/{}/{}/groundtruth_rect.txt'.format(self.base_path, class_name, sequence_name)
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        # ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        # occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(self.base_path, class_name, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        # full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        # out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        # out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        # target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        # frames_path = '{}/{}/{}/img'.format(self.base_path, class_name, sequence_name)
        frames_path = '{}/{}/img'.format(self.base_path, sequence_name)

        #########################################
        ## some images do not start from 0001.jpg
        video_name = sequence_name
        if "BlurCar1" == video_name:
            start = 247-1
        elif "BlurCar3" == video_name:
            start = 3-1
        elif "BlurCar4" == video_name:
            start = 18-1
        elif "David" == video_name:
            start = 300-1
        else:
            start = 0

        #########################################
        # frames_list = ['{}/{:04d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]
        if  "Board" == video_name:
            frames_list = ['{}/{:05d}.jpg'.format(frames_path, frame_number) for frame_number in
                           range(1 + start, ground_truth_rect.shape[0] + 1 + start)]
        else:
            frames_list = ['{}/{:04d}.jpg'.format(frames_path, frame_number) for frame_number in
                           range(1 + start, ground_truth_rect.shape[0] + 1 + start)]

        target_class = class_name

        ############################################################################
        ## Language
        # language = None
        try:
            language_file = os.path.join(self.base_path, class_name, sequence_name, "nlp.txt")
            with open(language_file, 'r') as file:
                language = file.readline().replace(".", "").replace("\n", "").lower()
        except:
            language = None
        #############################################################################
        # return Sequence(sequence_name, frames_list, 'otb99lang', ground_truth_rect.reshape(-1, 4),
        #                 object_class=target_class)
        return Sequence(sequence_name, frames_list, 'otb99lang', ground_truth_rect.reshape(-1, 4),
                        language=language, object_class=target_class)


    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = os.listdir(self.base_path)

        return sequence_list
