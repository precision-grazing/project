import torch
import torch.nn as nn

import warnings
import numpy as np
import matplotlib
import pandas as pd
import scipy.io

warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.figsize'] = (12.0, 12.0)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from config_args import parse_args
from data_utils.crop_utils import prep_overlap, predict_tiles, undo_overlap, predict_batch_tiles
from data_utils.data_postprocess import plot_surface, scatter_plot, plot_contour

from trainer_utils.trainer import TorchTrainer
from networks.encoderdecoder3d import EncoderDecoderWrapper3d

torch.manual_seed(420)
np.random.seed(420)

def main():
    print("Starting")
    # Parse arguments and load data
    args = parse_args()
    df = pd.read_csv(args.data_folder + 'processed_lidar_data.csv')    
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

    predict_single(df, args, trainer, split=True, plot=True)


def predict_single(df, args, trainer, split=True, plot=False):
    # print(df.head())
    height_list = ["h" + str(i + 1) for i in range(args.num_features)]
    # In: batch, seq, dim, dim
    scale_map_test = {}
    scaled_data_test = pd.DataFrame()
    scaled_data_test = pd.concat([scaled_data_test, df], ignore_index=True)
    for h in height_list:
        scaled_data_test[h] = (scaled_data_test[h] - df[h].min()) / (df[h].max() - df[h].min())
        scale_map_test[h] = {'min_test': df[h].min(), 'max_test': df[h].max()}

    h_aggr_list = np.array([np.array(scaled_data_test[h]) for h in height_list])
    h_aggr_list = np.swapaxes(h_aggr_list, 1, 0)
    h_aggr_list = np.reshape(h_aggr_list, (-1, args.xdim, args.ydim))
    h_aggr_list = h_aggr_list[np.newaxis]
    print(f"Shape of the given data: {h_aggr_list.shape}")
    h_out = h_aggr_list
    seq_len = h_aggr_list.shape[1]
    h_aggr_list = prep_overlap((args, h_aggr_list))
    print(f"Total Len of overlap: {len(h_aggr_list)} and shape: {h_aggr_list[0].shape}")
    if split:
        print("Os it coming?")
        seq_len = int(seq_len/2)
        print(f'Splitting across time: {seq_len}')
        h_in = [h[:, :seq_len] for h in h_aggr_list]
        h_out = h_out[:, seq_len:]
    else:
        h_in = h_aggr_list
        
    print(f"Shape of the input: {h_in[0].shape}, Output: {h_out.shape}")

    """

    Defining the Model
    """

    # x = ([torch.randn(size=(10, 5, 32, 32))], [])
    #
    # vis_graph = make_dot(model(x), params=dict(model.named_parameters()))
    # vis_graph.render("attached", format="png")
    #
    # return

    """
    Running Predictions
    """
    # h_in = ([torch.as_tensor(h_in, dtype=torch.float32)], [])
    # h_out = [torch.as_tensor(h_out, dtype=torch.float32)]
    print("Starting tile prediction")
    h_pred = predict_batch_tiles(h_in, h_out, args, trainer)
    h_pred_mean, h_pred_std = h_pred
    print("Startin Overlap Undo")
    print(f"Undo Overlap: {len(h_pred_mean)}, {h_pred_mean[0].shape}")
    h_pred_mean = undo_overlap((args, h_pred_mean))
    print(f"Undo Overlap: {len(h_pred_std)}, {h_pred_std[0].shape}")
    h_pred_std = undo_overlap((args, h_pred_std))
    h_target = h_out[0]
    h_error = h_target - h_pred_mean

    print(f'Mean: {h_pred_mean.shape}, Std: {h_pred_std.shape}, Target: {h_target.shape}')

    # Scaling
    min_test_scale = []
    max_test_scale = []
    for i in range(args.xdim * args.ydim):
        min_test_scale.append(scale_map_test['h' + str(i + 1)]['min_test'])
        max_test_scale.append(scale_map_test['h' + str(i + 1)]['max_test'])
    min_test_scale = np.asarray(min_test_scale).reshape((args.xdim, args.ydim))
    max_test_scale = np.asarray(max_test_scale).reshape((args.xdim, args.ydim))

    h_pred_mean = np.multiply(h_pred_mean, max_test_scale - min_test_scale) + min_test_scale
    h_pred_std = np.multiply(h_pred_std, max_test_scale - min_test_scale)
    
    if plot:
        h_error = np.multiply(h_error, max_test_scale - min_test_scale)
        h_target = np.multiply(np.expand_dims(h_target, 0), max_test_scale - min_test_scale) + min_test_scale
        for i in range(seq_len):
            predict_mean = h_pred_mean[0][i]
            predict_std = h_pred_std[0][i]
            predict_err = h_error[0][i]
            target_values = h_target[0][i]
            plot_contour(args, predict_mean, title=f"3D Mean: Time: {i}")
            plot_contour(args, predict_std, title=f"3D Std: Time: {i}")
            plot_contour(args, predict_err, title=f"3D Error: Time: {i}")
            plot_contour(args, target_values, title=f'3D Target: Time: {i}')

            scatter_plot(args, h_error, h_pred_std, title="Error vs Std. Deviation")

            y_mdic = {'y_predict_mean': h_pred_mean[0], 'y_predict_std': h_pred_std[0], 'y_predict_err': h_error[0],
                    'y_target': h_target[0]}

            scipy.io.savemat(
                args.data_folder + args.predict_folder + args.model + '_predict_data_' + args.predict_run + '_' + args.exp_name + '.mat', mdict=y_mdic, oned_as='row')
    else:
        return h_pred_mean, h_pred_std


if __name__ == '__main__':
    main()
