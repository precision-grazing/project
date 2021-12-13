import torch
import torch.nn as nn

import warnings
import numpy as np
import matplotlib
import pandas as pd
import scipy.io
import pickle
import multiprocessing as mp

from os import listdir

from torchviz import make_dot

from data_utils.data_preprocess import process_testing_data

warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.figsize'] = (12.0, 12.0)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from config_args import parse_args
from data_utils.crop_utils import prep_overlap, predict_tiles, undo_overlap, predict_batch_tiles
from data_utils.data_postprocess import plot_surface, scatter_plot, get_scale_3d

from trainer_utils.trainer import TorchTrainer
from networks.encoderdecoder3d import EncoderDecoderWrapper3d

torch.manual_seed(420)
np.random.seed(420)


def predict_batch():
    print("Starting")
    # Parse arguments and load data
    args = parse_args()
    #with mp.Pool(args.data_proc_workers) as pool:
    #        result = pool.map(process_testing_data, [args])[0]
    
    # Loading all data in to numpy arrays
    scaled_data = pd.read_pickle(args.data_folder + args.process_folder + args.model + '_test_predictions_processed_data' + '.pkl')
    height_list = ["h" + str(i + 1) for i in range(args.num_features)]  # This is already scaled
    h_aggr_list = np.array([np.array(scaled_data[h]) for h in height_list])
    h_aggr_list = np.swapaxes(h_aggr_list, 1, 0)
    h_aggr_list = np.reshape(h_aggr_list, (-1, args.xdim, args.ydim))
    h_aggr_list = h_aggr_list[np.newaxis]
    # Add mirror padding to the images
    h_aggr_list_target = h_aggr_list
    with mp.Pool(args.data_proc_workers) as pool:
        h_aggr_list = pool.map(prep_overlap, [(args, h_aggr_list)])[0]

    # h_aggr_list = prep_overlap(args, h_aggr_list) # h_aggr_list: (1, len, h+p, w+p)
    print(f"Shape of overlap: {h_aggr_list[0].shape}")
    if not args.mcmcdrop:
        args.n_samples = 1
    """
    Defining the Model
    """
    feature_list = ['h_in']
    c = 1
    t = 1
    h = args.window_size
    w = args.window_size
    x_features = (c, t, h, w)
    model = EncoderDecoderWrapper3d(args, None, None, feature_list, x_features)
    print(f'GPUs used: {torch.cuda.device_count()}')
    model = nn.DataParallel(model)  # , device_ids=[0], output_device=[0])
    model.to(args.device)
    loss_fn = torch.nn.MSELoss()
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    optimizers = [model_optimizer]
    schedulers = []
    trainer = TorchTrainer(
        args.exp_name,
        model,
        optimizers,
        loss_fn,
        schedulers,
        args.device,
        scheduler_batch_step=True,
        pass_y=False,
        args=args
    )
    
    # print(repr(model))
    trainer._load_checkpoint(only_model=True, epoch=args.epoch_load)

    # Start sequencing and predict in batches
    args.test_batch_size = 731
    in_len_b = int(args.in_seq_len * args.seq_stride)  + args.test_batch_size
    in_len_i = int(args.in_seq_len * args.seq_stride)
    out_len_i = int(args.out_seq_len * args.seq_stride)
    
    print("Starting tile prediction")
    break_loop = None
    for i in range(0, h_aggr_list[0].shape[1], args.test_batch_size):
        if i + args.test_batch_size + in_len_i + out_len_i >= h_aggr_list[0].shape[1]:
            args.test_batch_size = h_aggr_list[0].shape[1] - in_len_i - out_len_i - 1
            if args.test_batch_size == 0:
                break
            break_loop = True

        h_aggr_list_b = [h_aggr_list[0][:, i+j: i+j+in_len_i: args.seq_stride] 
                                        for j in range(args.test_batch_size)]
        h_aggr_list_b = [np.concatenate(h_aggr_list_b, axis=0)] #[b, seq, h+p, w+p]

        h_aggr_list_out_b = [h_aggr_list_target[:, i+j+in_len_i: i+j+in_len_i+out_len_i: args.seq_stride] 
                                        for j in range(args.test_batch_size)]
        h_aggr_list_out_b = np.concatenate(h_aggr_list_out_b, axis=0) #[b, seq, h+p, w+p]

        
        h_pred_b = predict_batch_tiles(h_aggr_list_b, [h_aggr_list_out_b], args, trainer)
        h_pred_mean_b, h_pred_std_b = h_pred_b
        
        with mp.Pool(args.data_proc_workers) as pool:
            h_pred_mean_b = pool.map(undo_overlap, [(args, h_pred_mean_b)])[0]
        with mp.Pool(args.data_proc_workers) as pool:
            h_pred_std_b = pool.map(undo_overlap, [(args, h_pred_std_b)])[0]
        h_error_b = h_aggr_list_out_b - h_pred_mean_b

        print(f'Mean: {h_pred_mean_b.shape}, Std: {h_pred_std_b.shape}, Target: {h_aggr_list_out_b.shape}, Error: {h_error_b.shape}')

        if i == 0:
            h_pred_mean = h_pred_mean_b
            h_pred_std = h_pred_std_b
            h_error = h_error_b
            h_target = h_aggr_list_out_b
        else:
            h_pred_mean = np.concatenate([h_pred_mean, h_pred_mean_b], axis=0)
            h_pred_std = np.concatenate([h_pred_std, h_pred_std_b], axis=0)
            h_error = np.concatenate([h_error, h_error_b], axis=0)
            h_target = np.concatenate([h_target, h_aggr_list_out_b], axis=0)
        
        if break_loop:
            break
        

    def scale_outs(value_str, scale, scale_std=False):
        if scale_std:
            value_str = np.multiply(value_str, scale[1] - scale[0])
        else:
            value_str = np.multiply(value_str, scale[1] - scale[0]) + scale[0]
        
        return value_str

    scale = get_scale_3d(args, file='Testing')
    h_pred_mean = scale_outs(h_pred_mean, scale)
    h_pred_std = scale_outs(h_pred_std, scale, True)
    h_target = scale_outs(h_target, scale)
    h_error = scale_outs(h_error, scale, True)

    y_mdic = {'y_predict_mean': h_pred_mean, 'y_predict_std': h_pred_std, 'y_predict_err': h_error,
              'y_target': h_target}

    scipy.io.savemat(
        args.data_folder + args.predict_folder + args.model + '_predict_data_' + args.predict_run + '_' + args.exp_name + '_testing_set.mat', mdict=y_mdic, oned_as='row')
    
    return

if __name__ == '__main__':
    predict_batch()
