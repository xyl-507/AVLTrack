import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist
import argparse

trackers = []

'''
cd /home/dell/WorkSpace/Tracking_by_NL/All-in-One/tracking/
conda activate python38   
python analysis_results.py
'''

#dataset_name = 'lasot'
#dataset_name = 'lasotext'
#dataset_name = 'tnl2k'
# dataset_name = 'otb99lang'
# dataset_name = 'webuav3m'

parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
parser.add_argument('--tracker_name', type=str, default='ostrack', help='Name of tracking method.')
parser.add_argument('--tracker_param', type=str, default='abavit_patch16_224_ep300', help='Name of config file.')
parser.add_argument('--dataset_name', type=str, default='webuav', help='Name of config file.')
parser.add_argument('--save_file', type=str, default=None)

args = parser.parse_args()

"""ostrack"""

dataset_name = args.dataset_name
# parameter_name = 'abavit_patch16_224_trans_tksa_ep297'
parameter_name = args.tracker_param

# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='levit_256_32x4_ep3', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256-levit'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='abavit_patch16_224_ep296', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256-abavit'))
# trackers.extend(trackerlist(name='ostrack', parameter_name=parameter_name, dataset_name=dataset_name,
#                             run_ids=None, display_name=parameter_name))

trackers.extend(trackerlist(name='ostrack', parameter_name=parameter_name, dataset_name=dataset_name,
                            run_ids=None, display_name=parameter_name))

# trackers.extend(trackerlist(name=args.tracker_name, parameter_name=args.tracker_param, dataset_name=args.dataset_name,  # copy from UVLTrack
#                             run_ids=None, display_name=args.tracker_name))

dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'), force_evaluation=True)
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))







