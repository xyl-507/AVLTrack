from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/public/datasets/got10k_lmdb'
    settings.got10k_path = '/public/datasets/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/public/datasets/itb'
    settings.lasot_lmdb_path = '/public/datasets/lasot_lmdb'
    settings.lasot_path = '/public/datasets/lasot'
    settings.network_path = '/home/xyl/newdrive/xyl-code2/All-in-One-ACMMM2023/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/public/datasets/nfs'
    settings.otb_path = '/public/datasets/otb'
    settings.prj_dir = '/home/xyl/newdrive/xyl-code2/All-in-One-ACMMM2023'
    settings.result_plot_path = '/home/xyl/newdrive/xyl-code2/All-in-One-ACMMM2023/output/test/result_plots'
    settings.results_path = '/home/xyl/newdrive/xyl-code2/All-in-One-ACMMM2023/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/xyl/newdrive/xyl-code2/All-in-One-ACMMM2023/output'
    settings.segmentation_path = '/home/xyl/newdrive/xyl-code2/All-in-One-ACMMM2023/output/test/segmentation_results'
    settings.tc128_path = '/public/datasets/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/public/datasets/trackingnet'
    settings.uav_path = '/public/datasets/uav'
    settings.vot18_path = '/public/datasets/vot2018'
    settings.vot22_path = '/public/datasets/vot2022'
    settings.vot_path = '/public/datasets/VOT2019'
    settings.youtubevos_dir = ''
    settings.tnl2k_path = '/public/datasets/TNL2K/test'
    settings.otb99lang_path = '/public/datasets/OTB_sentences/OTB_videos'
    settings.webuav3m_path = '/home/xyl/pysot-master/testing_dataset/WebUAV'

    return settings

