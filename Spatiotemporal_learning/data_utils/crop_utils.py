from operator import sub
import sys
import numpy as np
import scipy.signal
from torch.nn.modules.activation import LeakyReLU
from config_args import parse_args
import multiprocessing as mp
import torch
import torch.nn as nn

cached_2d_windows = dict()

def crop_frames(args, data):
    """ Indexing is done this way
    ##################
    #  1  #  2  # 3  #
    #  4  #  5  # 6  #
    #################
    """
    if args.num_features % (args.v_crop_scale * args.h_crop_scale) != 0:
        print("Incorrect Crop Scaling selected. Please ensure original dimension is divisible by crop ratio.")
        sys.exit()
    frames = []
    # Split frames
    # In: batch, seq, dim, dim
    # Out: [batch, seq, dim/scale, dim] * scale
    h_frames = np.split(data, args.h_crop_scale, -2)
    # In: [batch, seq, dim/scale, dim] * scale
    for f in h_frames:
        frames += np.split(f, args.v_crop_scale, -1)
    frames = np.concatenate(frames, axis=0)

    return frames


def fix_boundaries(args, predict_values):
    fix_idx = args.fix_boundary_len
    predict_values_fix = np.empty(shape=predict_values.shape)
    predict_values_fix[:, :, fix_idx:-fix_idx, fix_idx:-fix_idx] = \
        predict_values[:, :, fix_idx:-fix_idx, fix_idx:-fix_idx]

    # Final dim: batch, samples, seq_len, 1
    top_row_fix = predict_values[:, :, fix_idx, fix_idx:-fix_idx]
    top_row_fix = np.concatenate(([top_row_fix[:, :, 0][..., np.newaxis]] * fix_idx + [top_row_fix] + [top_row_fix[:, :, -1][..., np.newaxis]] * fix_idx), axis=-1)
    bot_row_fix = predict_values[:, :, -fix_idx, fix_idx:-fix_idx]
    bot_row_fix = np.concatenate(([bot_row_fix[:, :, 0][..., np.newaxis]] * fix_idx + [bot_row_fix] + [bot_row_fix[:, :, -1][..., np.newaxis]] * fix_idx), axis=-1)

    # Final dim: batch, samples, seq_len, row_size
    left_col_fix = predict_values[:, :, fix_idx:-fix_idx, fix_idx]
    right_col_fix = predict_values[:, :, fix_idx:-fix_idx, -fix_idx]

    for i in range(fix_idx):
        predict_values_fix[:, :, i, :] = top_row_fix
        predict_values_fix[:, :, -(i+1), :] = bot_row_fix
        predict_values_fix[:, :, fix_idx:-fix_idx, i] = left_col_fix
        predict_values_fix[:, :, fix_idx:-fix_idx, -(i+1)] = right_col_fix

    return predict_values_fix


def stitch_frames(args, z, each_frame, chunk_size):
    b = int(z.shape[0] / each_frame)
    y = np.empty((b, z.shape[1], args.xdim, args.ydim))
    crop_frame = int(args.xdim / args.h_crop_scale)
    i = 0
    for r in range(args.h_crop_scale):
        row = int(r * crop_frame)
        for c in range(args.v_crop_scale):
            col = int(c * crop_frame)
            for b in range(chunk_size):
                y[b, :, row:row + crop_frame, col:col + crop_frame] = z[i]
                i += 1

    return y


# This is a bit of hand wavy method since hdf5 is not so great at indexing,
# and that comes at a cost of size and performance, so we do it heuristically
def construct_frames(args, x, data_name, total_len):
    each_frame = int(args.v_crop_scale * args.h_crop_scale)
    chunk_frames = int(args.chunk_size * each_frame)
    chunk_size = args.chunk_size
    y = None
    chunk_idx = 0
    # for chunk_idx in range(0, total_len, args.chunk_size):
    break_next = False
    # print(f'Total Length: {total_len}')
    while True:
        # print(f'Idx: {chunk_idx}')
        if chunk_idx + chunk_frames > total_len - 1:
            # print(f'Idx: {chunk_idx}, Check: {chunk_idx + chunk_frames}')
            chunk_size = int((total_len % chunk_frames) / each_frame)
            # print(f'Chunk Size: {chunk_size}')
            break_next = True
        elif chunk_idx + chunk_frames == total_len - 1:
            break

        z = np.concatenate([x[data_name][chunk_idx + i * chunk_size:chunk_idx + (i+1) * chunk_size] for i in range(each_frame)])
        if args.fix_boundary:
            z = fix_boundaries(args, z)  # chunk, seq_len, dim, dim
        if y is None:
            # print(f'Stitching: {z.shape} and C: {chunk_size}')
            y = stitch_frames(args, z, each_frame, chunk_size)
        else:
            # print(f'Stitching: {z.shape} and C: {chunk_size}')
            y = np.concatenate([y, stitch_frames(args, z, each_frame, chunk_size)])
        chunk_idx += chunk_frames
        if break_next:
            break

    return y

def undo_single_crop(args, h_pred):
    y = np.empty((1, h_pred.shape[1], args.xdim, args.ydim))
    crop_frame = int(args.xdim / args.h_crop_scale)
    i = 0
    for r in range(args.h_crop_scale):
        row = int(r * crop_frame)
        for c in range(args.v_crop_scale):
            col = int(c * crop_frame)
            y[0, :, row:row + crop_frame, col:col + crop_frame] = h_pred[i]
            i += 1

    return y

"""
Overlap + Crop for Training
"""
# In: batch, seq, dim, dim
def crop_overlap_frames(args, data):
    aug = int(round(args.window_size * (1 - 1.0/args.subdivisions)))
    more_borders = ((0, 0), (0, 0), (aug, aug), (aug, aug))
    padded_img = np.pad(data, pad_width=more_borders, mode='reflect')
    step = int(args.window_size/args.subdivisions)
    padx_len = padded_img.shape[-2]
    pady_len = padded_img.shape[-1]

    subdivs = []
    for i in range(0, padx_len-args.window_size+1, step):
        for j in range(0, padx_len-args.window_size+1, step):
            patch = padded_img[:, :, i:i+args.window_size, j:j+args.window_size]
            subdivs.append(patch)

    subdivs = np.concatenate(subdivs, axis=0)

    return subdivs


"""
Overlapping Technique to blend edges Used for Single Prediction
https://github.com/Vooban/Smoothly-Blend-Image-Patches
"""
# Single Based
def predict_tiles(pads, y_data, args, trainer):
    res_mean = []
    res_std = []

    for pad in pads:
        # For every rotation:
        sd_mean, sd_std = _windowed_subdivs(pad, y_data, args.window_size, args.subdivisions, trainer, args)
        # output - sub_div * batch, seq, dim + window_size * (1 - 1 / subdivisions), dim + window_size * (1 - 1 / subdivisions)
        one_padded_result_mean = _recreate_from_subdivs(
            sd_mean, args.window_size, args.subdivisions,
            padded_out_shape=list(pad.shape))
        one_padded_result_std = _recreate_from_subdivs(
            sd_std, args.window_size, args.subdivisions,
            padded_out_shape=list(pad.shape))

        res_mean.append(one_padded_result_mean)
        res_std.append(one_padded_result_std)

    return res_mean, res_std

def _windowed_subdivs(padded_img, y_data, window_size, subdivisions, trainer, args):
    """
    Create tiled overlapping patches.
    Returns:
        5D numpy array of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            nb_output_channels
        )
    Note:
        patches_resolution_along_X == patches_resolution_along_Y == window_size
    """
    WINDOW_SPLINE_2D = np.squeeze(_window_2D(window_size=window_size, power=2))


    step = int(window_size/subdivisions)
    padx_len = padded_img.shape[-2]
    pady_len = padded_img.shape[-1]
    subdivs = []
    # print(f'Input Shape for Subdivision: {padded_img.shape}')
    for i in range(0, padx_len-window_size+1, step):
        subdivs.append([])
        for j in range(0, padx_len-window_size+1, step):
            patch = padded_img[:, :, i:i+window_size, j:j+window_size]
            subdivs[-1].append(patch)

    # Here, `gc.collect()` clears RAM between operations.
    # It should run faster if they are removed, if enough memory is available.
    # gc.collect()
    subdivs = np.array(subdivs)
    # print(f'Sub Division shape: {subdivs.shape}')
    # gc.collect()
    #a: list of sub-divisions (overlaps), b: batch, c: seq len, d: xdim, e: ydim
    a, b, c, d, e, f = subdivs.shape
    # subdivs = np.expand_dims(subdivs.reshape(a * b * c * d, e, f), -1)
    subdivs = subdivs.reshape(a * b * c, d, e, f)
    # print(f'Sub Division Shape Into Network: {subdivs.shape}')
    # print(f'Y Data Into Network: {y_data.shape}')
    # gc.collect()
    # print(subdivs.shape)
    subdivs = torch.as_tensor(subdivs, dtype=torch.float32)
    y_data = torch.as_tensor(y_data, dtype=torch.float32)

    pred = trainer.predict_one(subdivs, y_data, args.n_samples)
    subdivs_mean, subdivs_std = pred
    # print(f"Sub Mean Out N/W: {subdivs_mean.shape}, Std {subdivs_std.shape}")
    # subdivs = pred_func(subdivs)

    # gc.collect()
    # subdivs_mean = [s.reshape(s.shape[-2], s.shape[-1], 1) for s in subdivs_mean]
    # subdivs_std = [s.reshape(s.shape[-2], s.shape[-1]) for s in subdivs_std]

    subdivs_mean = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs_mean])
    subdivs_std = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs_std])
    # gc.collect()

    # Such 5D array:
    subdivs_mean = subdivs_mean.reshape(a, b, c, d, e, f)
    subdivs_std = subdivs_std.reshape(a, b, c, d, e, f)
    # print(f"Sub Mean Out After Window: {subdivs_mean.shape}, Std {subdivs_std.shape}")
    # gc.collect()

    return (subdivs_mean, subdivs_std)

#
# Batch Based
# Bit of a handwavy where we only batch for inference and do a naive way for processing
def predict_batch_tiles(pads, y_data, args, trainer):
    # Save shapes for reconstructing
    pad_shape = np.stack(pads).shape
    pad_b_len = pads[0].shape[0]

   

    # Returns list of subdivisions across all paddings
    # Shape of a * b * c, d, e, f for each pad
    with mp.Pool(args.data_proc_workers) as pool:
        subdvis, subdivs_shape = pool.map(_create_subdvis, [(pads, args.window_size, args.subdivisions)])[0]

    #subdvis, subdivs_shape = _create_subdvis(pads, args.window_size, args.subdivisions)

    # Stack subdivs into a batch and then performs prediction
    
    subdivs_mean, subdivs_std = _predict_batch_subdivs(subdvis, y_data, trainer, args)

    # Reconstruct it as a list as the same shape as input list and
    with mp.Pool(args.data_proc_workers) as pool:
        subdivs_mean = pool.map(_recreate_split_frames, [(subdivs_mean, pads, subdivs_shape, args)])[0]
    with mp.Pool(args.data_proc_workers) as pool:
        subdivs_std = pool.map(_recreate_split_frames, [(subdivs_std, pads, subdivs_shape, args)])[0]
    
    #subdivs_mean = _recreate_split_frames(subdivs_mean, pads, subdivs_shape, args)
    #subdivs_std = _recreate_split_frames(subdivs_std, pads, subdivs_shape, args)

    with mp.Pool(args.data_proc_workers) as pool:
        res_mean, res_std = pool.map(recreate_sub_div_lists, [(subdivs_mean, subdivs_std, args, pads)])[0]

    # for sd_mean in subdivs_mean:
    #     one_padded_result_mean = _recreate_from_subdivs(
    #         sd_mean, args.window_size, args.subdivisions,
    #         padded_out_shape=list(pads[0].shape))

    #     res_mean.append(one_padded_result_mean)

    # for sd_std in subdivs_std:
    #     one_padded_result_std = _recreate_from_subdivs(
    #         sd_std, args.window_size, args.subdivisions,
    #         padded_out_shape=list(pads[0].shape))

    #     res_std.append(one_padded_result_std)

    return res_mean, res_std

def recreate_sub_div_lists(argv):
    subdivs_mean, subdivs_std, args, pads = argv
    res_mean = []
    res_std = []
    for sd_mean in subdivs_mean:
        one_padded_result_mean = _recreate_from_subdivs(
            sd_mean, args.window_size, args.subdivisions,
            padded_out_shape=list(pads[0].shape))

        res_mean.append(one_padded_result_mean)

    for sd_std in subdivs_std:
        one_padded_result_std = _recreate_from_subdivs(
            sd_std, args.window_size, args.subdivisions,
            padded_out_shape=list(pads[0].shape))

        res_std.append(one_padded_result_std)

    return (res_mean, res_std)

def _create_subdvis(argv):
    pads, window_size, subdivisions = argv
    """
    Create tiled overlapping patches.
    Returns:
        5D numpy array of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            nb_output_channels
        )
    Note:
        patches_resolution_along_X == patches_resolution_along_Y == window_size
    """
    subdivs_list = []
    subdivs_shape_list = []
    subdivs_patch_list = []

    for padded_img in pads:

        step = int(window_size/subdivisions)
        padx_len = padded_img.shape[-2]
        pady_len = padded_img.shape[-1]
        subdivs = []
        # print(f'Input Shape for Subdivision: {padded_img.shape}')
        for i in range(0, padx_len-window_size+1, step):
            subdivs.append([])
            for j in range(0, padx_len-window_size+1, step):
                patch = padded_img[:, :, i:i+window_size, j:j+window_size]
                subdivs[-1].append(patch)

        # Here, `gc.collect()` clears RAM between operations.
        # It should run faster if they are removed, if enough memory is available.
        # gc.collect()
        subdivs = np.array(subdivs)
        # print(f'Sub Division shape: {subdivs.shape}')
        # gc.collect()
        #a: list of sub-divisions (overlaps), b: batch, c: seq len, d: xdim, e: ydim
        a, b, c, d, e, f = subdivs.shape
        # subdivs = np.expand_dims(subdivs.reshape(a * b * c * d, e, f), -1)
        subdivs = subdivs.reshape(a * b * c, d, e, f)
        # print(f'Sub Division Shape Into Network: {subdivs.shape}')
        # print(f'Y Data Into Network: {y_data.shape}')
        # gc.collect()
        # print(subdivs.shape)
        subdivs_list.append(subdivs)
        subdivs_shape_list.append((a, b, c, d, e, f))

    return subdivs_list, subdivs_shape_list

def _predict_batch_subdivs(subdivs, y_data, trainer, args):
    subdivs = np.concatenate(subdivs, axis=0)
    if subdivs.shape[0] > args.batch_size:
        subdivs = np.array_split(subdivs, subdivs.shape[0] // args.batch_size + 1)
    else:
        subdivs = [subdivs]
    y_data = torch.as_tensor(y_data, dtype=torch.float32)

    pred_mean_out = []
    pred_std_out = []
    for subdiv in subdivs:
        subdiv = torch.as_tensor(subdiv, dtype=torch.float32)
        pred_mean, pred_std = trainer.predict_one(subdiv, y_data, args.n_samples)
        pred_mean_out.append(pred_mean)
        pred_std_out.append(pred_std)

    return np.concatenate(pred_mean_out), np.concatenate(pred_std_out)

def _recreate_split_frames(argv):
    subdivs_in, pads, subdivs_shape, args = argv
    i = 0
    subdivs = np.split(subdivs_in, len(subdivs_shape))
    # for pad in pads:
    #     subdivs.append(subdivs_in[i:i + len(pad)])
    #     i += len(pad)

    WINDOW_SPLINE_2D = np.squeeze(_window_2D(window_size=args.window_size, power=2))

    out = []
    for i, subdiv in enumerate(subdivs):
        a, b, c, d, e, f = subdivs_shape[i]
        subdiv = np.array([patch * WINDOW_SPLINE_2D for patch in subdiv])
        out.append(subdiv.reshape(a, b, c, d, e, f))

    return out

# Weighting Technique

def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        # print(f'Spline Window Shape: {wind.shape}')
        wind = np.expand_dims(np.expand_dims(wind, 1), 1)
        wind = wind * wind.transpose(1, 0, 2)
        # if PLOT_PROGRESS:
        #     # For demo purpose, let's look once at the window:
        #     plt.imshow(wind[:, :, 0], cmap="viridis")
        #     plt.title("Windowing Function (2D) 32x32 used for blending \n"
        #               " the overlapping patches (Interpolation)")
        #     plt.show()
        cached_2d_windows[key] = wind
    return wind


# Generate Mirror Padded Image First
def _pad_img(img, window_size, subdivisions):
    """
    Add borders to img for a "valid" border pattern according to "window_size" and
    "subdivisions".
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((0, 0), (0, 0), (aug, aug), (aug, aug))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    # gc.collect()

    return ret

def _unpad_img(padded_img, window_size, subdivisions):
    """
    Undo what's done in the `_pad_img` function.
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    ret = padded_img[
        :,
        :,
        aug:-aug,
        aug:-aug
    ]
    # gc.collect()
    return ret


def _rotate_mirror_do(im):
    """
    Duplicate an np array (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(-2, -1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(-2, -1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(-2, -1), k=3))
    im = np.array(im)[:, :, :, ::-1]
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(-2, -1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(-2, -1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(-2, -1), k=3))

    # mirrs = np.concatenate(mirrs, axis=0)

    return mirrs


def _rotate_mirror_undo(im_mirrs):
    """
    merges a list of 8 np arrays (images) of shape (x, y, nb_channels) generated
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(-2, -1), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(-2, -1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(-2, -1), k=1))
    origs.append(np.array(im_mirrs[4])[:, :, :, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[5]), axes=(-2, -1), k=3)[:, :, :, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[6]), axes=(-2, -1), k=2)[:, :, :, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[7]), axes=(-2, -1), k=1)[:, :, :, ::-1])

    return np.mean(origs, axis=0)

def _recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
    """
    Merge tiled overlapping patches smoothly.
    """
    step = int(window_size/subdivisions)
    padx_len = padded_out_shape[-2]
    pady_len = padded_out_shape[-1]

    y = np.zeros(padded_out_shape)
    # print(y.shape)
    # print(subdivs.shape)

    a = 0
    for i in range(0, padx_len-window_size+1, step):
        b = 0
        for j in range(0, padx_len-window_size+1, step):
            windowed_patch = subdivs[a, b]
            # print(windowed_patch.shape)
            y[:, :, i:i+window_size, j:j+window_size] = y[:, :, i:i+window_size, j:j+window_size] + windowed_patch
            b += 1
        a += 1

    return y / (subdivisions ** 2)


def prep_overlap(argv):
    args, data = argv
    pad = _pad_img(data, args.window_size, args.subdivisions)
    #pads = _rotate_mirror_do(pad)
    pads = [pad]

    return pads


def undo_overlap(argv):
    args, data = argv
    #data = _rotate_mirror_undo(data)
    prd = _unpad_img(data[0], args.window_size, args.subdivisions)
    prd = prd[:, :, :prd.shape[-2], :prd.shape[-1]]

    return prd


# We will pass individual data to this to keep things simple
# Then convert it torch tensors and back again
def predict_with_tiling(args, data, pred_func):
    data = np.random.uniform(0, 1, size=(10, 4, 100, 100))
    # Input: batch, seq, dim, dim
    # Mirror Pad images first
    # Adds 1/2 size of window_size pixels across the edges of the input image
    # to remove zero padding effects
    # Mirror Pad: batch, seq, dim + window_size * (1 - 1/subdivisions), dim + window_size * (1 - 1/subdivisions)
    pad = _pad_img(data, args.window_size, args.subdivisions)
    print(f'Pad Shape: {pad.shape}')
    # Rotates each image 4x times + Mirror and rotate for a total of 8x images returns as a list
    # Rotate Pad: 8, batch, seq, dim + window_size * (1 - 1/subdivisions), dim + window_size * (1 - 1/subdivisions)
    pads = _rotate_mirror_do(pad)
    print(f'Rotate Shape: {len(pads)}, and {pad[0].shape}')

    res = []
    for pad in pads:
        # For every rotation:
        sd = _windowed_subdivs(pad, args.window_size, args.subdivisions, pred_func)
        # output - sub_div * batch, seq, dim + window_size * (1 - 1 / subdivisions), dim + window_size * (1 - 1 / subdivisions)
        one_padded_result = _recreate_from_subdivs(
            sd, args.window_size, args.subdivisions,
            padded_out_shape=list(pad.shape))

        res.append(one_padded_result)

    # Merge after rotations:
    padded_results = _rotate_mirror_undo(res)
    print(f'Final padded Shape: {padded_results.shape}')
    prd = _unpad_img(padded_results, args.window_size, args.subdivisions)
    print(f'Final Unpadded Shape: {prd.shape}')
    prd = prd[:, :, :data.shape[-2], :data.shape[-1]]
    print(f'Final Shape: {prd.shape}')

    return prd


#
#
# if __name__ == '__main__':
#     args = parse_args()
#     predict_with_tiling(args, None, None)
