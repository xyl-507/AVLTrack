import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text, load_str
import os

class TNL2KDataset(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.tnl2k_path   ###############
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
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        # occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(self.base_path, class_name, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        # full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        # out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        # out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        # target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        # frames_path = '{}/{}/{}/img'.format(self.base_path, class_name, sequence_name)
        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)
        # /Data/Datasets/TNL2K/test/test_015_Sord_video_Q01_done/imgs/00000492.png, break (we use 491)

        # frames_list = ['{}/{:05d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]
        ######################################################################
        ## some images do not start from 00001.jpg
        images_list = sorted(os.listdir(frames_path))
        frames_list = ['{}/{}'.format(frames_path, images_list[frame_number]) for frame_number in
                       range(0, ground_truth_rect.shape[0])]
        target_class = class_name

        ############################################################################
        ## Language
        try:
            language_file = os.path.join(self.base_path, class_name, sequence_name, "language.txt")
            with open(language_file, 'r') as file:
                language = file.readline().replace(".", "").replace("\n", "").lower()
        except:
            language = None
        #############################################################################
        # return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4),
        #                 object_class=target_class)
        return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4),
                        language=language, object_class=target_class)


    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = os.listdir(self.base_path)

        return sequence_list
