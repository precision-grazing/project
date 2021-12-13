import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import gc
import warnings
import os
import sys
from pathlib import Path
import multiprocessing as mp

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle

warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.figsize'] = (12.0, 12.0)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from config_args import parse_args
from data_utils.data_preprocess import load_data
from data_utils.sequence_builder import seq_builder
from data_utils.seq_loader import load_seq_as_np, load_seq_as_np_predict
from data_utils.data_loader import ItemDataset
from data_utils.data_postprocess import plot_3d, np_to_mat_3d, save_raw_predictions, plot_1d, np_to_mat_1d, scale_1d

from trainer_utils.trainer import TorchTrainer
from networks.encoderdecoder3d import EncoderDecoderWrapper3d
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(420)
np.random.seed(420) 

def dir_checks(args):
    process_dir = Path(args.data_folder + args.process_folder)
    if not process_dir.exists():
        os.makedirs(process_dir)

    seq_dir = Path(args.data_folder + args.preseq_folder)
    if not seq_dir.exists():
        os.makedirs(seq_dir)

    pred_dir = Path(args.data_folder + args.predict_folder)
    if not pred_dir.exists():
        os.makedirs(pred_dir)

    num_dir = Path(args.data_folder + args.seq_np_folder)
    if not num_dir.exists():
        os.makedirs(num_dir)

    train_file = Path(args.data_folder + args.seq_np_folder + args.model + '_train_seq_data_' + str(args.in_seq_len) + '_' +
                      str(args.out_seq_len) + '.h5')
    test_file = Path(args.data_folder + args.seq_np_folder + args.model + '_test_seq_data_' + str(args.in_seq_len) + '_' +
                     str(args.out_seq_len) + '.h5')

    if train_file.exists():
        print("Train File Seq Exists")
        sys.exit()

    if test_file.exists():
        print("Test File Seq Exists")
        sys.exit()

    return


def train():
    # Parse arguments and load data
    args = parse_args()
    if args.select_gpus:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device_ids
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_train_data = False
    if args.lr_search or args.train_network:
        load_train_data = True
    if args.load_data or args.sequence_data or args.sequence_to_np:
        dir_checks(args)
    if args.predict_mode:
        predict_file = Path(args.data_folder + args.predict_folder + args.model + '_predict_seq_data_' + args.predict_run + '_' + str(
            args.in_seq_len) + '_' +
                            str(args.out_seq_len) + '.h5')
        if predict_file.exists() and args.gen_seq_np_predict:
            print('Predict File Exists, please change predict run')
            sys.exit()

    # If new dataset is to be loaded and processed with scaling/norms etc, then
    # Create batches of input sequence and output sequence that needs to be predicted
    # feature_list = ['h_in']
    feature_list = ['h_in']

    if args.load_data:
        with mp.Pool(args.data_proc_workers) as pool:
            result = pool.map(load_data, [args])[0]
    if args.sequence_data:
        with mp.Pool(args.data_proc_workers) as pool:
            result = pool.map(seq_builder, [(args, feature_list, 'train')])[0]
        with mp.Pool(args.data_proc_workers) as pool:
            result = pool.map(seq_builder, [(args, feature_list, 'test')])[0]
    if args.sequence_to_np:
        pool = mp.Pool(args.data_proc_workers)
        for f in range(2 * len(feature_list)):
            print(f"Working with {f}")
            result = pool.map(load_seq_as_np, [(args, feature_list, f)])[0]
        pool.close()
        del result
    if args.preprocess_only:
        return
    # Create separate dataset for prediction, as searching for indexes requires loading
    # the whole dataset in memory which is inefficient for large datasets like the test data.
    if args.predict_mode and args.gen_seq_np_predict:
        # Augment the test data with sequences of dates, it's not needed for train dataset
        # Running it separately to be efficient on RAM usage
        pool = mp.Pool(args.data_proc_workers)
        for f in range(2 * len(feature_list)):
            print(f"Working with {f}")
            result = pool.map(load_seq_as_np_predict, [(args, feature_list, f, False)])[0]
        result = pool.map(load_seq_as_np_predict, [(args, feature_list, 0, True)])[0] # Add date index
        pool.close()
        del result

    gc.collect()

    if load_train_data:
        print("Create Training DataLoader")
        # Load into memory
        train_dataset = ItemDataset(args, feature_list, 'train')
        train_dataset.load_sequence_data(args.data_folder + args.seq_np_folder + args.model + '_train_seq_data_'
                                         + str(args.in_seq_len) + '_' + str(args.out_seq_len) + '.h5')
        # Do not shuffle here. Data is preshuffled for training to improve sequential read performance
        # This is due to the data stored/read in chunks equivalent to batches.
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=False,
                                      num_workers=args.data_loader_workers, pin_memory=True)
        X_enc, y = train_dataset.__getitem__(0)

        print("Create Testing DataLoader")
        test_dataset = ItemDataset(args, feature_list, 'test')
        test_dataset.load_sequence_data(args.data_folder + args.seq_np_folder + args.model + '_test_seq_data_'
                                        + str(args.in_seq_len) + '_' + str(args.out_seq_len) + '.h5')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                     num_workers=args.data_loader_workers, pin_memory=True)

    if args.predict_mode:
        print("Create Prediction DataLoader")
        args.chunk_size = args.batch_size
        predict_dataset = ItemDataset(args, feature_list, 'predict')
        predict_dataset.load_sequence_data(args.data_folder + args.predict_folder + args.model + '_predict_seq_data_' + args.predict_run
                                           + '_' + str(args.in_seq_len) + '_' + str(args.out_seq_len) + '.h5')
        predict_dataloader = DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                        num_workers=args.data_loader_workers, pin_memory=True)
        X_enc, y, _ = predict_dataset.__getitem__(0)
        print(f'Dataset Length: {predict_dataset.length}')

    # Load all features to initialize models
    # Encoder Features
    # ['h_in']
    c = 1
    t = 1

    h, w = X_enc[0, :, :].shape
    print(f'Shape of input: {X_enc[0, :, :].shape}')
    x_enc_features_3d = (c, t, h, w)
    
    model = EncoderDecoderWrapper3d(args, None, None, feature_list, x_enc_features_3d)
    print("loading model to device")
    if args.multi_gpu:
        print(f'GPUs used: {torch.cuda.device_count()}')
        model = nn.DataParallel(model)#, device_ids=args.device_ids, output_device=[0]).to(args.device)
    
    model.to(args.device)
    print(repr(model))

    n_epochs = args.epoch
    if args.train_network:
        steps_epoch = len(train_dataloader)
    else:
        steps_epoch = 10

    # Training Loop
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()

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

    if args.lr_search and load_train_data:
        print("Starting Learning Rate Finder")
        trainer.lr_find(train_dataloader, model_optimizer, start_lr=1e-5, end_lr=1e-2, num_iter=500)

    if args.train_network and load_train_data:
        print("Starting Epochs")
        trainer.train(n_epochs, train_dataloader, test_dataloader, resume_only_model=True, resume=True)

    """
    Prediction Part 
    """
    if args.predict_mode:
        if args.run_inference:
            print("Starting Prediction")
            trainer._load_checkpoint(only_model=True)
            (predict_values_mean, predict_values_std), target_values = trainer.predict(predict_dataloader, args,
                                                                                        n_samples=args.n_samples,
                                                                                        plot_phase=True)
            print("Finished Predicting")
            # Save raw predicted values
            save_raw_predictions(args, predict_values_mean, predict_values_std, target_values)
            del predict_values_std
            del predict_values_mean
            del target_values
            gc.collect()

        if args.post_process:
            print("Starting post process and saving to MATLAB")
            with mp.Pool(args.data_proc_workers) as pool:
                result = pool.map(np_to_mat_3d, [(args, 'y_predict_mean')])[0]
            with mp.Pool(args.data_proc_workers) as pool:
                result = pool.map(np_to_mat_3d, [(args, 'y_predict_std')])[0]
            with mp.Pool(args.data_proc_workers) as pool:
                result = pool.map(np_to_mat_3d, [(args, 'y_predict_err')])[0]
            with mp.Pool(args.data_proc_workers) as pool:
                result = pool.map(np_to_mat_3d, [(args, 'y_target')])[0]
            # np_to_mat_3d(args, 'y_predict_mean')
            # np_to_mat_3d(args, 'y_predict_std')
            # np_to_mat_3d(args, 'y_predict_err')
            # np_to_mat_3d(args, 'y_target')

        if args.plot:
            print("Plotting Graphs")
            with mp.Pool(args.data_proc_workers) as pool:
                result = pool.map(plot_3d, [args])[0]

        return


if __name__ == '__main__':
    train()
