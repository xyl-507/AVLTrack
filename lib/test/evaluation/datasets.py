from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    webuav3m=DatasetInfo(module=pt % "webuav3m", class_name="WebUAV3MDataset", kwargs=dict()),
    otb99lang=DatasetInfo(module=pt % "otb99lang", class_name="OTB99LANGDataset", kwargs=dict()),
    tnl2k=DatasetInfo(module=pt % "tnl2k", class_name="TNL2KDataset", kwargs=dict()),
    lasotext=DatasetInfo(module=pt % "lasotext", class_name="LaSOTEXTDataset", kwargs=dict()),
    otb=DatasetInfo(module=pt % "otb", class_name="OTBDataset", kwargs=dict()),
    nfs=DatasetInfo(module=pt % "nfs", class_name="NFSDataset", kwargs=dict()),
    uav=DatasetInfo(module=pt % "uav", class_name="UAVDataset", kwargs=dict()),
    tc128=DatasetInfo(module=pt % "tc128", class_name="TC128Dataset", kwargs=dict()),
    tc128ce=DatasetInfo(module=pt % "tc128ce", class_name="TC128CEDataset", kwargs=dict()),
    trackingnet=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict()),
    got10k_test=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='test')),
    got10k_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val')),
    got10k_ltrval=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='ltrval')),
    lasot=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict()),
    lasot_lmdb=DatasetInfo(module=pt % "lasot_lmdb", class_name="LaSOTlmdbDataset", kwargs=dict()),
    vot18=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict()),
    vot22=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict(year=22)),
    itb=DatasetInfo(module=pt % "itb", class_name="ITBDataset", kwargs=dict()),
    # tnl2k=DatasetInfo(module=pt % "tnl2k", class_name="TNL2kDataset", kwargs=dict()),
    # lasot_extension_subset=DatasetInfo(module=pt % "lasotextensionsubset", class_name="LaSOTExtensionSubsetDataset",
    #                                    kwargs=dict()),
    uavtrack112=DatasetInfo(module=pt % "uavtrack112", class_name="UAVTrack112Dataset", kwargs=dict()),  # UAVTrack112 数据集
    uavtrack112l=DatasetInfo(module=pt % "uavtrack112l", class_name="UAVTrack112LDataset", kwargs=dict()),  # UAVTrack112L 数据集
    uavdt=DatasetInfo(module=pt % "uavdt", class_name="UAVDTDataset", kwargs=dict()),  # UAVDT数据集
    dtb=DatasetInfo(module=pt % "dtb", class_name="DTBDataset", kwargs=dict()),  # DTB70 数据集
    visdrone=DatasetInfo(module=pt % "visdrone", class_name="VisDroneDataset", kwargs=dict()),  # VisDrone2019-SOT-test-dev 数据集
    uav20l=DatasetInfo(module=pt % "uav20l", class_name="UAV20LDataset", kwargs=dict()),  # UAV20L 数据集
    dtbnlp=DatasetInfo(module=pt % "dtbnlp", class_name="DTBNLPDataset", kwargs=dict()),  # DTBNLP 数据集
    uavdtnlp=DatasetInfo(module=pt % "uavdtnlp", class_name="UAVDTNLPDataset", kwargs=dict()),  # UAVDTNLP 数据集
    visdronenlp=DatasetInfo(module=pt % "visdronenlp", class_name="VisDroneNLPDataset", kwargs=dict()),  # VisDroneNLP 数据集
    uav20lnlp=DatasetInfo(module=pt % "uav20lnlp", class_name="UAV20LNLPDataset", kwargs=dict()),  # UAV20LNLP 数据集
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset