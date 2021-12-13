import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import scipy.io
import numpy as np
import h5py
import pickle

import data_utils.crop_utils
from data_utils.crop_utils import construct_frames
from data_utils.data_loader import PostProcessDataset
from data_utils.seq_loader import get_shape, append_hf

from pathlib import Path
import os


def plot_surface(args, z_data, title):
    z = z_data
    # print(z.shape)
    sh_0, sh_1 = z.shape
    x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title=title, autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))

    plot_dir = Path(args.data_folder + args.plot_folder + args.exp_name + '/')
    if not plot_dir.exists():
        os.makedirs(plot_dir)

    fig.write_html(args.data_folder + args.plot_folder + args.exp_name + '/' + title + '.html')
    if args.show_plot:
        fig.show()

def plot_contour(args, z_data, title):
    # Valid color strings are CSS colors, rgb or hex strings
    colorscale = [[0, 'lightsalmon'], [0.5, 'mediumturquoise'], [1, 'green']]

    fig = go.Figure(data =
        go.Contour(
            z=z_data,
            colorscale=colorscale)
        )
    colorbar=dict(
        title='mm', # title here
        titleside='right',
        titlefont=dict(
        size=14,
        family='Arial, sans-serif')
    )

    fig.update_layout(legend=dict(
        orientation="h")
        )

    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
        )

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    plot_dir = Path(args.data_folder + args.plot_folder + args.exp_name + '/')
    if not plot_dir.exists():
        os.makedirs(plot_dir)
    fig.write_image(args.data_folder + args.plot_folder + args.exp_name + '/' + title + '.pdf')
    #fig.write_html(args.data_folder + args.plot_folder + args.exp_name + '/' + title + '.html')
    if args.show_plot:
        fig.show()

def scatter_plot(args, x, y, title):
    x = x.flatten()
    y = y.flatten()
    fig = px.scatter(x=x, y=y, title=title, labels={'x':'Error (mm)', 'y':'Standard Deviation'})
    plot_dir = Path(args.data_folder + args.plot_folder)
    if not plot_dir.exists():
        os.makedirs(plot_dir)
    fig.update_layout(
        showlegend=False,
        font_family="Times New Roman",
        font_color="black",
        title_font_family="Times New Roman",
        title_font_color="black",
        legend_title_font_color="black",
        hovermode="x"
        )
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
        )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.write_image(args.data_folder + args.plot_folder + args.exp_name + '/' + title + '.pdf')

    #fig.write_html(args.data_folder + args.plot_folder + args.exp_name + '/' + title + '.html')

    if args.show_plot:
        fig.show()


# 1. Load the dataset indexing for all mean, std, err, target
# 2. Load the scaled data and save it to the indexing class
# 3.
# Reshape to feed to Matlab
def np_to_mat_3d(argv):
    args, data_name = argv
    y_date, data_idx = get_date_idx(args)
    chunk_size = y_date.shape[0] // args.save_size + 1

    # Input Dataset: Raw values
    value_file_set = PostProcessDataset(args, data_name)
    value_file = value_file_set.load_sequence_data(args.data_folder + args.predict_folder + args.model + '_predict_val_data_' + args.predict_run + '_' + str(args.in_seq_len) + '_' +
        str(args.out_seq_len) + '.h5')

    filepath = Path(args.predict_folder + '/' + data_name + '/')
    if not filepath.exists():
        os.makedirs(filepath)

    # Output Dataset for Plotting Purpose
    value_final_hf = h5py.File(
        args.data_folder + args.predict_folder + args.model + '_predict_data_' + data_name + '_' + args.predict_run + '.h5', 'a')
    value_final_hf.create_dataset('date', data=y_date.astype('S10'))
    value_final_hf.create_dataset('data_index', data=data_idx)

    if args.crop_frame:
        print(f"Starting crop frames: {data_name}")
        value = construct_frames(args, value_file, data_name, value_file_set.__len__())
        print(f'Stitched {data_name}: {value.shape}')
    elif args.overlap_frames:
        print(f"Starting to undo overlapping frames: {data_name}")
    else:
        value = value_file.get(data_name)

    if args.scale_outputs:
        print("Starting Scaling")
        scale = get_scale_3d(args)

    for i in range(chunk_size):
        # idx_strt = i * args.save_size
        if i == chunk_size - 1:
            value_str = value#[idx_strt:]
            y_date_str = y_date#[idx_strt:]
            data_idx_str = data_idx#[idx_strt:]
        else:
            value_str = value[:args.save_size] #[idx_strt:idx_strt + args.save_size]
            y_date_str = y_date[:args.save_size]#[idx_strt:idx_strt + args.save_size]
            data_idx_str = data_idx[:args.save_size]#[idx_strt:idx_strt + args.save_size]

        value = value[args.save_size:]
        y_date = y_date[args.save_size:]
        data_idx = data_idx[args.save_size:]

        if args.scale_outputs:
            if data_name == "y_predict_std" or data_name == "y_predict_err":
                value_str = np.multiply(value_str, scale[1] - scale[0])
            else:
                value_str = np.multiply(value_str, scale[1] - scale[0]) + scale[0]

        y_mdic = {'date': y_date_str, 'data_index': data_idx_str,
                  data_name: value_str}

        if i == 0:
            max_shape = get_shape(value_str)
            value_final_hf.create_dataset(data_name, data=np.asarray(value_str, dtype=np.float32),
                                      chunks=True, maxshape=max_shape)
        else:
            value_final_hf[data_name].resize((value_final_hf[data_name].shape[0] + value_str.shape[0]), axis=0)
            value_final_hf[data_name][-value_str.shape[0]:] = value_str

        scipy.io.savemat(args.data_folder + args.predict_folder + '/' + data_name + '/' + args.model + '_predict_data_' + data_name + '_' + args.predict_run + '_f'
                         + str(i) + '.mat', mdict=y_mdic, oned_as='row')

    value_final_hf.close()


def plot_3d(args):
    mean_value_file_set = PostProcessDataset(args, 'y_predict_mean')
    mean_value_file = mean_value_file_set.load_sequence_data(
        args.data_folder + args.predict_folder + args.model + '_predict_data_' + 'y_predict_mean' + '_' + args.predict_run + '.h5')

    std_value_file_set = PostProcessDataset(args, 'y_predict_std')
    std_value_file = std_value_file_set.load_sequence_data(
        args.data_folder + args.predict_folder + args.model + '_predict_data_' + 'y_predict_std' + '_' + args.predict_run + '.h5')

    err_value_file_set = PostProcessDataset(args, 'y_predict_err')
    err_value_file = err_value_file_set.load_sequence_data(
        args.data_folder + args.predict_folder + args.model + '_predict_data_' + 'y_predict_err' + '_' + args.predict_run + '.h5')

    target_value_file_set = PostProcessDataset(args, 'y_target')
    target_value_file = target_value_file_set.load_sequence_data(
        args.data_folder + args.predict_folder + args.model + '_predict_data_' + 'y_target' + '_' + args.predict_run + '.h5')

    print(f'Plotting date: {args.plot_date_file[0]} from File: {args.plot_date_file[1]}')
    for i in range(mean_value_file_set.__len__()):
        if mean_value_file['date'][i][0].decode("utf-8") == args.plot_date_file[0] and mean_value_file['data_index'][i] == args.plot_date_file[1]:
            predict_mean = mean_value_file['y_predict_mean'][i]
            predict_std = std_value_file['y_predict_std'][i]
            predict_err = err_value_file['y_predict_err'][i]
            target_values = target_value_file['y_target'][i]
            print(predict_mean.shape)
            date = mean_value_file['date'][i]
            data_idx = mean_value_file['data_index']
            seq_len = predict_mean.shape[0]
            for t in range(seq_len):
                if t % args.plot_interval == 0 or t + 1 == seq_len:
                    plot_surface(predict_mean[t], title=f"3D Mean: t{i}, {date}, File: {data_idx}")
                    plot_surface(predict_std[t], title=f"3D Std: t{i}, {date}, File: {data_idx}")
                    plot_surface(predict_err[t], title=f"3D Error: t{i}, {date}, File: {data_idx}")
                    plot_surface(target_values[t], title=f'3D Target: t{i}, {date}, File: {data_idx}')


def get_scale_3d(args, file=None):
    if file is None:
        scale_map_test = pickle.load(open(args.process_folder + args.model + '_scale_map_test.pkl', 'rb'))
    else:
        scale_map_test = pickle.load(open(args.data_folder + args.process_folder + args.model + '_scale_map_test_predictions.pkl', 'rb'))
    # Now convert the dictionary {h1: {'min_test": value, 'max_test': value}}
    min_test_scale = []
    max_test_scale = []
    for i in range(args.xdim * args.ydim):
        min_test_scale.append(scale_map_test['h' + str(i + 1)]['min_test'])
        max_test_scale.append(scale_map_test['h' + str(i + 1)]['max_test'])

    min_test_scale = np.asarray(min_test_scale).reshape((args.xdim, args.ydim))
    max_test_scale = np.asarray(max_test_scale).reshape((args.xdim, args.ydim))
    print(f'Scale Shape: {max_test_scale.shape}, {min_test_scale.shape}')
    scale = (min_test_scale, max_test_scale)

    return scale


"""
1D Utilities
"""


def np_to_mat_1d(args, y_predict_mean, y_predict_std, y_predict_err, y_target):
    y_date, data_idx = get_date_idx(args)
    y_mdic = {'date': y_date, 'data_index': data_idx,
              'y_predict_mean': y_predict_mean, 'y_predict_std': y_predict_std,
              'y_predict_err': y_predict_err, 'y_target': y_target}

    scipy.io.savemat(args.predict_folder + args.model + '_predict_data_' + args.predict_run + '.mat', mdict=y_mdic, oned_as='row')


def plot_1d(batch_idx, predict_mean, predict_std, predict_err, target_values):
    plot_graph_1d(predict_mean[batch_idx], predict_std[batch_idx], predict_err[batch_idx], target_values[batch_idx])


def scale_1d(args, y_predict_mean, y_predict_std, y_target):
    scaled_data_test = pickle.load(open(args.data_folder + args.process_folder + args.model + '_scale_map_test.pkl', 'rb'))
    min_test_scale = scaled_data_test['h1']['min_test']
    max_test_scale = scaled_data_test['h1']['max_test']
    scale = (min_test_scale, max_test_scale)

    y_predict_mean = np.multiply(y_predict_mean, scale[1] - scale[0]) + scale[0]
    y_target = np.multiply(y_target, scale[1] - scale[0]) + scale[0]
    y_predict_std = np.multiply(y_predict_std, scale[1] - scale[0])

    return y_predict_mean, y_predict_std, y_target


def plot_graph_1d(z_pred_mean, z_pred_std, z_pred_err, z_target):
    sh_0, sh_1 = z_target.shape
    x_lin = np.linspace(0, 1, sh_0)
    z_target = np.squeeze(z_target)
    z_pred_mean = np.squeeze(z_pred_mean)
    z_pred_std = np.squeeze(z_pred_std)
    z_pred_err = np.squeeze(z_pred_err)
    fig = go.Figure([
    go.Scatter(
        name='Prediction',
        x=x_lin,
        y=z_pred_mean,
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
    ),
    go.Scatter(
        name='Upper Bound',
        x=x_lin,
        y=z_pred_mean+z_pred_std,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Lower Bound',
        x=x_lin,
        y=z_pred_mean-z_pred_std,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    ),
    go.Scatter(
        name='Target',
        x=x_lin,
        y=z_target,
        mode='lines',
        line=dict(color='rgb(148, 103, 189)'),
    ),
    go.Scatter(
        name='Error',
        x=x_lin,
        y=z_pred_err,
        mode='lines',
        line=dict(color='rgb(148, 103, 189)'),
    )
    ])


    fig.update_layout(
        yaxis_title='Height Values',
        title='1D Prediction with variance',
        hovermode="x"
    )
    fig.show()


"""
Common Utilities
"""


def get_date_idx(args):
    # Get index and dates
    predict_hf = h5py.File(args.data_folder + args.predict_folder + args.model + '_predict_seq_data_' + args.predict_run
                                           + '_' + str(args.in_seq_len) + '_' + str(args.out_seq_len) + '.h5', 'r')
    # Inverse of np.repeat to store data or plot
    target_dates = predict_hf['date'][:]
    target_dates = np.asarray(target_dates).reshape(-1, 1)
    target_data_idx = predict_hf['data_index'][:]
    target_data_idx = np.asarray(target_data_idx).reshape(-1, 1)
    each_frame = int(args.v_crop_scale * args.h_crop_scale)
    b = int(target_dates.shape[0] / each_frame)
    if args.crop_frame and args.model in ['3d', '3D']:
        target_dates = target_dates[:b, :]
        target_data_idx = target_data_idx[:b, :]

    predict_hf.close()

    target_dates = target_dates.copy(order='C')
    target_data_idx = target_data_idx.copy(order='C')

    return target_dates, target_data_idx


def save_raw_predictions(args, y_mean, y_std, y_target):
    print("Saving raw prediction file")
    predict_val_hf = h5py.File(
        args.data_folder + args.predict_folder + args.model + '_predict_val_data_' + args.predict_run + '_' + str(args.in_seq_len) + '_' +
        str(args.out_seq_len) + '.h5', 'w')
    predict_val_hf.create_dataset('y_predict_std', data=np.asarray(y_std, dtype=np.float32))
    del y_std
    y_err = y_mean - y_target
    predict_val_hf.create_dataset('y_predict_err', data=np.asarray(y_err, dtype=np.float32))
    del y_err
    predict_val_hf.create_dataset('y_predict_mean', data=np.asarray(y_mean, dtype=np.float32))
    del y_mean
    predict_val_hf.create_dataset('y_target', data=np.asarray(y_target, dtype=np.float32))
    del y_target
