import sys
import pickle
import itertools

import numpy as np
import pandas as pd


# Load and Save Sequence to Pickle
def seq_builder(argv):
    args, feature_list, data_type = argv
    # feature_list = ['h_in', 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']
    scaled_data = pd.read_pickle(args.data_folder + args.process_folder + args.model + '_' + data_type + '_processed_data' + '.pkl')
    
    # height_list = ["h" + str(i + 1) for i in range(args.num_features)]
    # height_yearly_corr_list = [h + '_yearly_corr' for h in height_list]
    # log_height_list = ["log_h" + str(i + 1) for i in range(args.num_features)]

    print("Starting Sequencing Inputs")
    height_list = ["h" + str(i + 1) for i in range(args.num_features)]  # This is already scaled
    h_aggr_list = np.array([np.array(scaled_data[h]) for h in height_list])
    # Change to (data_len, num_features) and then move to 3D
    h_aggr_list = np.swapaxes(h_aggr_list, 1, 0)
    h_aggr_list = np.reshape(h_aggr_list, (-1, args.xdim, args.ydim))
    h_aggr_list = list(h_aggr_list)
    scaled_data['h_in'] = h_aggr_list

    drop_features_list = [h for h in list(scaled_data.columns)
                        if h not in feature_list + ['date', 'data_index']]

    scaled_data.drop(drop_features_list, axis=1, inplace=True)
    print(f'Scaled and Reshaped Features: \n {scaled_data.head()} \n \n')

    # Create the sliding windows
    print(f"Starting Sliding Windows for {data_type} data.")
    X_seq, y_seq = create_sliding_win(args, scaled_data, feature_list, data_type)
    print(f"Finished Sliding Windows for {data_type} data.")

    print(f"Saving sequenced data to {data_type} file")
    sequence_data = pd.DataFrame()
    for i, f in enumerate(feature_list):
        sequence_data['x_seq_' + f] = X_seq[i]
    # sequence_data['x_seq'] = X_seq
        sequence_data['y_seq_' + f] = y_seq[i]
    sequence_data['date'] = scaled_data['date']
    sequence_data['data_index'] = scaled_data['data_index']
    print(f'Sequenced Features for {data_type}: \n {sequence_data.head()} \n {sequence_data.tail()} \n \n')

    sequence_data.to_pickle(args.data_folder + args.preseq_folder + args.model + '_' + data_type + '_seq_data_' + str(args.in_seq_len) + '_' +
                  str(args.out_seq_len) + '.pkl')


# Make sure h_in is the first
def create_sliding_win(args, data, feature_list, data_type, stride=1):
    X_list = [[] for _ in range(len(feature_list))]
    y_list = [[] for _ in range(len(feature_list))]
    # Calculate the number of steps across the complete dataset
    steps = list(range(0, len(data), stride))

    len_data_file = "_n_train_days"
    if data_type is "test":
        len_data_file = "_n_test_days"
    with open(args.data_folder + args.process_folder + args.model + len_data_file + '.pkl', 'rb') as fp:
        len_year = pickle.load(fp)

    # feature_list = ['h_in', 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']
    len_year_cycle = itertools.cycle(len_year)
    cumm_year_days = next(len_year_cycle)

    in_seq_len = args.in_seq_len
    out_seq_len = args.out_seq_len

    if args.seq_stride > 1:
        in_seq_len = int(args.seq_stride * args.in_seq_len)
        out_seq_len = int(args.seq_stride * args.out_seq_len)

    for i in steps:
        # find the end of this pattern
        end_ix = i + in_seq_len
        out_end_ix = end_ix + out_seq_len
        # check if we are beyond the current dataset
        if out_end_ix > cumm_year_days:
            if end_ix == cumm_year_days - 1:
                cumm_year_days += next(len_year_cycle)
            continue
        # check if we are beyond the complete merged dataset
        if out_end_ix > len(data):
            break
        # [rows/steps, #features]
        for j, f in enumerate(feature_list):
            X_list[j].append(data.iloc[i:end_ix:args.seq_stride][f].values)
            y_list[j].append(data.iloc[end_ix:out_end_ix:args.seq_stride][f].values)

    return X_list, y_list
