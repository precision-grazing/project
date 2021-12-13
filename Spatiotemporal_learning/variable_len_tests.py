import os
import json
from pathlib import Path
from os import listdir

import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import scipy.io
from cnn_calc import main

from config_args import parse_args

from trainer_utils.trainer import TorchTrainer
from networks.encoderdecoder3d import EncoderDecoderWrapper3d
from single_overlap_test import predict_single

torch.manual_seed(420)
np.random.seed(420)

def predict_planners(args, trainer, imputation_seq):     
    predict_dir = Path(args.plan_data_folder + 'prediction')
    if not predict_dir.exists():
        os.makedirs(predict_dir)
    
    filepaths = [f for f in sorted(listdir(args.plan_data_folder)) if f.endswith(args.plan_dataset)]

    for i, f in enumerate(filepaths):
        print(f"Reading {f}")
        
        df = pd.read_csv(args.plan_data_folder + f)
        # print(f)
        df.drop(['date', 'year'], inplace=True, axis=1)
        
        df = interpolate_missing(df)
        day_of_year = df['day_of_year'].values
        seq_len = len(day_of_year)
        day_of_year_out = day_of_year + seq_len
        
        if imputation_seq == 1:
            # 1 Stride Sequence
            args.in_seq_len = seq_len
            args.out_seq_len = seq_len
            h_pred_mean, h_pred_std = predict_single(df, args, trainer, split=False, plot=False)
        else:
            df_i = df[::imputation_seq]
            day_of_year_out = day_of_year_out[::imputation_seq]
            seq_len = len(day_of_year_out)
            args.in_seq_len = seq_len
            args.out_seq_len = seq_len
            h_pred_mean, h_pred_std = predict_single(df_i, args, trainer, split=False, plot=False)

        y_mdic = {'day_of_year': day_of_year_out, 'y_predict_mean': h_pred_mean[0], 'y_predict_std': h_pred_std[0]}
        scipy.io.savemat(
            str(predict_dir) + '/' + str(imputation_seq) + '_stride_' + f[:-4] + '.mat', mdict=y_mdic, oned_as='row')

def interpolate_missing(df):
    df.set_index('day_of_year', inplace=True)
    print(f'Min: {df.index.min()}, Max: {df.index.max()}')
    # Insert missing nans here to interpolate the data
    for t in range(df.index.min(), df.index.max()):
        if any(df.index == t):
            continue
        else:
            df1 = pd.Series(name=t, dtype=float)
            df = df.append(df1)
    df.sort_index(inplace=True)
    df.reset_index(inplace=True)
    df.interpolate(method='linear', inplace=True, axis=0)

    return df

def load_models(args):
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

    return trainer

def main():
    args = parse_args()
    #models = ['1D_15L_0.4Dr_No3D_32', '2D_15L_0.4Dr_No3D_32', '4D_15L_0.4Dr_No3D_32']
    models = ['4D_15L_0.4Dr_No3D_32']
    epoch = [18]
    #epoch = [34, 28, 18]
    impt_seq = [4]
    #impt_seq = [1, 2, 4]
    for i, t in enumerate(impt_seq):
        args.exp_name = models[i]
        args.epoch_load = epoch[i]
        trainer = load_models(args)
        print(f'Exp: {args.exp_name}, Epoch: {args.epoch_load}, Stride: {t}')
        predict_planners(args, trainer, t)




if __name__ == '__main__':
    main()


    