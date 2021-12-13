import pandas as pd
import numpy as np
import gc
import h5py
from sklearn.utils import shuffle
import math

from data_utils.crop_utils import crop_frames, crop_overlap_frames


def load_seq_as_np(argv):
    print("Loading data frame for sequencing as numpy")
    args, feature_list, f_idx_in = argv
    train_hf = h5py.File(
        args.data_folder + args.seq_np_folder + args.model + '_train_seq_data_' + str(args.in_seq_len) + '_' +
        str(args.out_seq_len) + '.h5', 'a')
    test_hf = h5py.File(
        args.data_folder + args.seq_np_folder + args.model + '_test_seq_data_' + str(args.in_seq_len) + '_' +
        str(args.out_seq_len) + '.h5', 'a')
    f_idx = f_idx_in
    y_feature = False
    if f_idx >= len(feature_list):
        f_idx -= len(feature_list)
        y_feature = True

    if args.training_mode in ['train_init', 'train_final']:
        train_df = pd.read_pickle(args.data_folder + args.preseq_folder + args.model + '_train_seq_data_' + str(args.in_seq_len) + '_' +
                      str(args.out_seq_len) + '.pkl')
        print("Train data loaded")

    test_df = pd.read_pickle(args.data_folder + args.preseq_folder + args.model + '_test_seq_data_' + str(args.in_seq_len) + '_' +
                              str(args.out_seq_len) + '.pkl')
    print("Test data loaded")

    # print(sequence_data.head())
    if args.training_mode == 'train_init':
        train_sequence_data = train_df[train_df['date'] < args.validation_start_date]
        test_sequence_data = train_df[(train_df['date'] >= args.validation_start_date) & (train_df['date'] <
                                                                                            args.testing_start_date)]
    elif args.training_mode == 'train_final':
        train_sequence_data = train_df[train_df['date'] < args.testing_start_date]
        test_sequence_data = test_df[(test_df['date'] >= args.testing_start_date) & (test_df['date'] <
                                                                                            args.testing_end_date)]

    else:
        test_sequence_data = test_df[(test_df['date'] >= args.testing_start_date) & (test_df['date'] <
                                                                                            args.testing_end_date)]

    # We need to sequence it in parts to reduce memory consumption
    # as numpy operations take upwards of 250GB+ RAM/Swap
    print(f"Starting train sequencing: {f_idx_in}")
    if args.train_shuffle:
        train_idx = train_sequence_data.index
        train_sequence_data = shuffle(train_sequence_data)
        train_sequence_data.index = train_idx
    train_sequence_chunk = split_dataframe(train_sequence_data, args.batch_size)

    for i, chunk in enumerate(train_sequence_chunk):
        train_f_chunk = pd_to_numpy(chunk, feature_list, f_idx, y_feature)
        train_f_idx = numpy_obj_to_arr(train_f_chunk)
        if args.overlap_frame:
            train_f_idx = crop_overlap_frames(args, train_f_idx)
        elif args.crop_frame and args.model in ['3d', '3D']:
            train_f_idx = crop_frames(args, train_f_idx)
        if i == 0:
            max_shape = get_shape(train_f_idx)
            train_hf.create_dataset('train' + str(f_idx_in), data=np.asarray(train_f_idx, dtype=np.float32), chunks=True, maxshape=max_shape)
        else:
            train_hf = append_hf(train_hf, 'train', f_idx_in, np.asarray(train_f_idx, dtype=np.float32))
    # train_f_idx = pd_to_numpy(train_sequence_data, feature_list, f_idx, y_feature)
    print(f"Finished train sequencing and saved to file: {f_idx_in}")
    del train_sequence_data
    del train_df
    
    print(f"Starting test sequencing: {f_idx_in}")
    test_sequence_chunk = split_dataframe(test_sequence_data, args.batch_size)
    for i, chunk in enumerate(test_sequence_chunk):
        test_f_chunk = pd_to_numpy(chunk, feature_list, f_idx, y_feature)
        test_f_idx = numpy_obj_to_arr(test_f_chunk)
        if args.overlap_frame:
            test_f_idx = crop_overlap_frames(args, test_f_idx)
        elif args.crop_frame:
            test_f_idx = crop_frames(args, test_f_idx)
        if i == 0:
            max_shape = get_shape(test_f_idx)
            test_hf.create_dataset('test' + str(f_idx_in), data=np.asarray(test_f_idx, dtype=np.float32), chunks=True, maxshape=max_shape)
        else:
            test_hf = append_hf(test_hf, 'test', f_idx_in, np.asarray(test_f_idx, dtype=np.float32))
    print(f"Finished test sequencing and saved to file: {f_idx_in}")

    del test_sequence_data
    del test_df
    gc.collect()
    train_hf.close()
    test_hf.close()

    return


def load_seq_as_np_predict(argv):
    args, feature_list, f_idx_in, add_date_idx = argv
    predict_hf = h5py.File(
        args.data_folder + args.predict_folder + args.model + '_predict_seq_data_' + args.predict_run + '_' + str(args.in_seq_len) + '_' +
        str(args.out_seq_len) + '.h5', 'a')

    print("Loading pickle data frame for sequencing for prediction.")

    predict_df = pd.read_pickle(args.data_folder + args.preseq_folder + args.model + '_test_seq_data_' + str(args.in_seq_len) + '_' +
                              str(args.out_seq_len) + '.pkl')
    if args.predict_selected:
        print(f"Sequencing dates: {args.predict_dates}")
        predict_sequence_data = predict_df[(predict_df['date'].isin(args.predict_dates))]
    else:
        predict_sequence_data = predict_df

    f_idx = f_idx_in
    y_feature = False
    if f_idx >= len(feature_list):
        f_idx -= len(feature_list)
        y_feature = True

    if add_date_idx:
        print(f"Starting test sequencing: dates & data_index")
        predict_dates = np.stack(predict_sequence_data['date'].dt.strftime('%Y-%m-%d').tolist(), axis=0)
        predict_data_idx = np.stack(predict_sequence_data['data_index'].tolist(), axis=0)
        # print(f'Predict: {predict_data_idx.shape} and {predict_dates.shape}')
        if args.overlap_frame:
            image_size = args.xdim + int(round(args.window_size * (1 - 1.0/args.subdivisions)))
            frame_step = int(args.window_size/args.subdivisions)
            overlap_tiles = int(math.floor((image_size-(args.window_size))/frame_step)+1)**2
            predict_dates = np.tile(predict_dates, overlap_tiles)
            predict_data_idx = np.tile(predict_data_idx, overlap_tiles)
        elif args.crop_frame:
            predict_dates = np.tile(predict_dates, int(args.v_crop_scale * args.h_crop_scale))
            predict_data_idx = np.tile(predict_data_idx, int(args.v_crop_scale * args.h_crop_scale))
            # predict_dates = np.repeat(predict_dates, int(args.v_crop_scale * args.h_crop_scale), axis=0)
            # predict_data_idx = np.repeat(predict_data_idx, int(args.v_crop_scale * args.h_crop_scale), axis=0)
        # print(f'Predict crop: {predict_data_idx.shape} and {predict_dates.shape}')

        predict_hf.create_dataset('date', data=predict_dates.astype('S10'))
        predict_hf.create_dataset('data_index', data=predict_data_idx)
        predict_hf.close()
        print("Finished date & data index entries to prediction dataset")
        return

    print(f"Starting prediction sequencing: {f_idx_in}")
    predict_sequence_chunk = split_dataframe(predict_sequence_data, args.chunk_size)

    for i, chunk in enumerate(predict_sequence_chunk):
        predict_f_chunk = pd_to_numpy(chunk, feature_list, f_idx, y_feature)
        predict_f_idx = numpy_obj_to_arr(predict_f_chunk)
        if args.overlap_frame:
            predict_f_idx = crop_overlap_frames(args, predict_f_idx)
        elif args.crop_frame:
            predict_f_idx = crop_frames(args, predict_f_idx)
        if i == 0:
            max_shape = get_shape(predict_f_idx)
            predict_hf.create_dataset('predict' + str(f_idx_in), data=np.asarray(predict_f_idx, dtype=np.float32), chunks=True, maxshape=max_shape)
        else:
            predict_hf = append_hf(predict_hf, 'predict', f_idx_in, np.asarray(predict_f_idx, dtype=np.float32))
    print(f"Finished test sequencing and saved to file: {f_idx_in}")
    predict_hf.close()


def numpy_obj_to_arr(seq_f_idx):
    batch_data_shape = list(seq_f_idx.shape)
    row_data_shape = list(seq_f_idx[0][0].shape)

    if len(row_data_shape) > 0:
        new_data = np.stack(seq_f_idx.ravel()).reshape(batch_data_shape[0], batch_data_shape[1], row_data_shape[0],
                                                       row_data_shape[1])
        # print(f"3D: {new_data.shape}")
    else:
        new_data = np.stack(seq_f_idx.ravel()).reshape(batch_data_shape[0], batch_data_shape[1], 1)
        # print(f"Scalar: {new_data.shape}")
    return new_data


def pd_to_numpy(seq_data, feature_list, f_idx, y_feature=False):
    if not y_feature:
        seq_np = np.stack(seq_data['x_seq_' + feature_list[f_idx]].tolist(), axis=0)
    else:
        seq_np = np.stack(seq_data['y_seq_' + feature_list[f_idx]].tolist(), axis=0)

    return seq_np


def split_dataframe(df, chunk_size=100):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])

    return chunks


def append_hf(hf, data_type, f_idx, data):
    hf[data_type + str(f_idx)].resize((hf[data_type + str(f_idx)].shape[0] + data.shape[0]), axis=0)
    hf[data_type + str(f_idx)][-data.shape[0]:] = data

    return hf


def get_shape(data):
    max_shape = list(data.shape)
    max_shape[0] = None
    max_shape = tuple(max_shape)

    return max_shape
