import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os

class UAVTrack112LDataset(BaseDataset):

    def __init__(self):
        super().__init__()
        # self.base_path = "/home/liuyao/Datasets/smallobject/"
        # self.base_path = '/DATA/lcl/A-data-tiny'
        self.base_path = '/home/xyl/newdrive/0.test_datasets/UAVTrack112-L'  # 更改数据集时需要修改的地方
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info["path"]
        imgs = os.listdir(os.path.join(self.base_path, sequence_path))  # 更改数据集时需要修改的地方
        frames = list()
        for img in sorted(imgs):
            imgpath = os.path.join(sequence_path, img)
            # print(imgpath)
            frames.append(os.path.join(self.base_path, imgpath))  # 更改数据集时需要修改的地方
        
        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']
        anno_path = '{}/../UAVTrack112-anno/anno_l/{}'.format(self.base_path, sequence_info['anno_path'])  # 更改数据集时需要修改的地方
        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'uavtrack112l', ground_truth_rect[init_omit:, :],  # 更改数据集时需要修改的地方
                        object_class=sequence_info['object_class'])

    """def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'otb', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])"""

    def __len__(self):
        return len(self.sequence_info_list)


    def _get_sequence_info_list(self):
        
        sequence_dir = []
        sequence_path = []
        # for root, dirs, files in os.walk('/home/liuyao/Datasets/smallobject'):
        # for root, dirs, files in os.walk('/DATA/lcl/A-data-tiny'):
        for root, dirs, files in os.walk('/home/xyl/newdrive/0.test_datasets/UAVTrack112-L'):  # 更改数据集时需要修改的地方
            for dir in dirs:
                if dir == 'img':   # 更改数据集时需要修改的地方
                # if dir == 'img': # filter the img direction
                    continue
                sequence_dir.append(dir) # restore sequence direction
                sequence_path.append(os.path.join(root, dir)) # restore sequence dircetion path
        sequence_dir = sorted(sequence_dir)
        sequence_path = sorted(sequence_path)
        img_path = []
        end_frame = []
        # print('seq: ',sequence_path)
        for path in sequence_path:

            for root, dirs, files in os.walk(path):
                # for dir in dirs:
                #     img_path.append(os.path.join(root, dir))
                for file in files:
                    img_path.append(os.path.join(root, file))  # xyl20230525 没有img子文件夹

        # for img_p in img_path:
        for img_p in sequence_path:
            # print('img: ',img_p)
            # for root, dirs, files in os.walk(img_p):
            files = os.listdir(img_p)
            end_frame.append(len(files))
        sequence_info_list = []
        for i in range(len(sequence_dir)):
            sequence_info = {}
            sequence_info["name"] = sequence_dir[i] 
            sequence_info["path"] = sequence_dir[i]
            sequence_info["startFrame"] = int('1')
            sequence_info["endFrame"] = end_frame[i]
                
            sequence_info["nz"] = int('6')
            sequence_info["ext"] = 'jpg'  # 更改数据集时需要修改的地方
            sequence_info["anno_path"] = os.path.join(sequence_info["name"]+'.txt')  # 更改数据集时需要修改的地方，标注文件为 air conditioning box1.txt
            sequence_info["object_class"] = 'vehicle'
            sequence_info_list.append(sequence_info)    
        return sequence_info_list
