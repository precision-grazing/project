import argparse


def parse_args():
    parser = argparse.ArgumentParser("TimeSeries")

    parser.add_argument("--exp_name", type=str, default='1D_15L_0.4Dr_No3D_32')
    parser.add_argument("--lr_search", default=False, action='store_true')
    parser.add_argument("--train_network", default=True, action='store_true')
    parser.add_argument("--training_mode", type=str, default="train_final", choices=['test', 'train_init', 'train_final'])
    parser.add_argument("--model", type=str, default="3D", choices=["3D", "3d"])

    # Training
    # Device Dependent
    parser.add_argument("--chunk_size", type=int, default=3072)
    parser.add_argument("--save_size", type=int, default=128)
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument("--select_gpus", default=True, action='store_true')
    parser.add_argument("--device_ids", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--multi_gpu", default=True, action='store_true')
    parser.add_argument("--data_proc_workers", type=int, default=24)
    parser.add_argument("--data_loader_workers", type=int, default=36)
    
    # Training Parameters
    parser.add_argument("--lr", type=float, default=3.67E-05,
                        help='1.94E-05 with features 1d, 4.65E-05 without features')
    parser.add_argument("--lr_decay", type=float, default=1.0E-06)
    parser.add_argument("--epoch_load", type=int, default=25)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=3072)

    # Data size & scale
    parser.add_argument("--multiple_data", default=True, action='store_true')
    parser.add_argument("--train_shuffle", default=True, action='store_true')
    parser.add_argument("--xdim", type=int, default=100)
    parser.add_argument("--ydim", type=int, default=100)
    parser.add_argument("--num_features", type=int, default=10000)    
    parser.add_argument("--in_seq_len", type=int, default=15, help='Input prediction steps')
    parser.add_argument("--out_seq_len", type=int, default=15, help='Output prediciton steps')
    parser.add_argument("--seq_stride", type=int, default=1, help='Days between each input and output sequence')
    
    # Data Post Processing
    # Cropping based tests
    parser.add_argument("--v_crop_scale", type=int, default=4)
    parser.add_argument("--h_crop_scale", type=int, default=4, help='Amount of scaling to be done')
    parser.add_argument("--crop_frame", default=True, action='store_true', help='Crop frames during in')
    parser.add_argument("--overlap_frame", default=True, action='store_true')
    parser.add_argument("--fix_boundary", default=True, action='store_true', help='remove pixels from the boundary')
    parser.add_argument("--fix_boundary_len", type=int, default=2, help='Used for cropping, number of pixels from each boundary to be removed')
    # Overlapping Based tests
    parser.add_argument("--subdivisions", type=int, default=2, help='parameter to control how much overlapping is performed')
    parser.add_argument("--window_size", type=int, default=32, help='Effective input size after overlapping')
    # Ablation Studies
    parser.add_argument("--impt_stride", type=int, default=2, help='Interpolation Stride Between Missing Intervals')
    parser.add_argument("--imputation", default=True, action='store_true')

    """Architectures"""
    # 2D ConvLSTM Architecture
    parser.add_argument("--final_residue", default=True, action='store_true')
    parser.add_argument("--use_conv3d_preenc", default=False, action='store_true')

    # Bayesian Uncertainity Parameters
    parser.add_argument("--use_bayes_inf", default=True, action='store_true')
    parser.add_argument("--mcmcdrop", default=True, action='store_true')
    parser.add_argument("--enc_droprate", type=float, default=0.4)
    parser.add_argument("--dec_droprate", type=float, default=0.4)
    parser.add_argument("--n_samples", type=int, default=500)

    # Data pre-processing methods
    parser.add_argument("--no_cylical_dates", default=False, action='store_true')
    parser.add_argument("--height_correction", type=str, default='log', choices=['log', 'autocorrelation'])
    parser.add_argument("--testing_start_date", type=str, default='2008-01-01')
    parser.add_argument("--drop_dates", type=list, default=['2010-01-01'])
    parser.add_argument("--testing_end_date", type=str, default='2009-12-31')
    parser.add_argument("--validation_start_date", type=str, default='2008-01-01')
    # parser.add_argument("--predict_date", type=str, default='2009-04-01')

    # Data Processing Requirements
    #parser.add_argument("--load_data", default=False, action='store_true')
    #parser.add_argument("--sequence_data", default=False, action='store_true')
    #parser.add_argument("--sequence_to_np", default=False, action='store_true')
    #parser.add_argument("--preprocess_only", default=False, action='store_true')
    # Don't use
    parser.add_argument("--compress_data", default=False, action='store_true')

    # Data Processing Requirements
    parser.add_argument("--load_data", default=True, action='store_true')
    parser.add_argument("--sequence_data", default=True, action='store_true')
    parser.add_argument("--sequence_to_np", default=True, action='store_true')
    parser.add_argument("--preprocess_only", default=False, action='store_true')

    # File Paths
    parser.add_argument("--data_folder", type=str, default='/home/tago/data/')
    # parser.add_argument("--plan_data_folder", type=str, default='/mnt/nfs/common/20210927/')
    parser.add_argument("--plan_data_folder", type=str, default='/home/tago/data/20210927/')
    parser.add_argument("--plan_dataset", type=str, default='.csv')
    parser.add_argument("--process_folder", type=str, default='processed_data/')
    parser.add_argument("--preseq_folder", type=str, default='sequence_data/')
    parser.add_argument("--seq_np_folder", type=str, default='sequence_data/numpy/')
    parser.add_argument("--predict_folder", type=str, default='prediction_data/')
    parser.add_argument("--plot_folder", type=str, default='prediction_data/plots/')
    parser.add_argument("--show_plot", default=False, action='store_true')

    # Features
    parser.add_argument("--use_log_h", default=False, action='store_true')
    parser.add_argument("--use_add_features", default=False, action='store_true')
    parser.add_argument("--use_yr_corr", default=False, action='store_true')
    parser.add_argument("--use_skip_conn", default=True, action='store_true')
    parser.add_argument("--twolayer_convlstm", default=False, action='store_true')
    parser.add_argument("--skip_layers", type=list, default=[0, 2])

    # Data Postprocessing
    parser.add_argument("--predict_mode", default=False, action='store_true')
    parser.add_argument("--gen_seq_np_predict", default=False, action='store_true')
    parser.add_argument("--predict_selected", default=False, action='store_true')
    parser.add_argument("--scale_outputs", default=True, action='store_true')
    parser.add_argument("--predict_run", type=str, default='0')
    parser.add_argument("--plot", default=True, action='store_true')
    parser.add_argument("--plot_interval", type=int, default=10)
    parser.add_argument('--predict_dates', nargs='+', default=['2009-03-31', '2009-08-01'],
                        help='Need a list of prediction dates, use --predict_selected with this for predictions')
    parser.add_argument("--plot_date_file", nargs='+', default=['2009-03-31', 0])
    parser.add_argument("--post_process", default=True, action='store_true')
    parser.add_argument("--run_inference", default=False, action='store_true')

    return parser.parse_args()
