import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.modules.pooling import MaxPool2d, MaxPool3d

from networks.ConvLSTM import ConvLSTMCell, KERNEL_SIZE


class EncoderDecoderWrapper3d(nn.Module):
    def __init__(self, args, encoder, decoder_cell, feature_list, n_features=None):
        super().__init__()
        self.encoder = encoder
        self.decoder_cell = decoder_cell
        self.args = args
        self.device = self.args.device
        self.feature_list = feature_list

        self.c, self.t, self.h, self.w = n_features  # time, channel, height, width

        # Out Channels from preCNN encoders
        self.c1x = 32
        self.c2x = 64
        self.c4x = 128
        self.pc1x = int(self.c1x/2)
        self.pc2x = int(self.c2x/2)
        self.pc4x = int(self.c4x/2)

        """
        Encoding Networks
        """
        # Initial Encoders for feeding ConvLSTM
        # Maintain same channel size across all pre-encoders for each scale
        self.conv2d_enc = nn.ModuleList([
            # 1x Scale Representations
            nn.Conv3d(1, self.pc1x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # HW 64 -> 64; T -> T
            nn.Conv3d(self.pc1x, self.pc1x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # HW 64 -> 64; T -> T
            # 2x Scale Representations
            # Use both above layers to create a smaller representation
            nn.Conv3d(self.c1x, self.pc2x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # HW 64 -> 32; T -> T
            nn.Conv3d(self.pc2x, self.pc2x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # HW 64 -> 32; T -> T

            # 3x Scale Representations
            # Use both above layers to create a larger representation
            nn.Conv3d(self.c2x, self.pc4x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # HW 32 -> 16; T -> T
            nn.Conv3d(self.pc4x, self.pc4x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # HW 32 -> 16; T -> T
        ])

        self.conv2d_maxpool = nn.ModuleList([
            nn.MaxPool3d(kernel_size=(1, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), stride=(1, 2, 2)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), stride=(1, 2, 2))
        ])

        self.bn_conv2d_enc = nn.ModuleList([
            nn.BatchNorm3d(self.pc1x),
            nn.BatchNorm3d(self.pc1x),
            nn.BatchNorm3d(self.pc2x),
            nn.BatchNorm3d(self.pc2x),
            nn.BatchNorm3d(self.pc4x),
            nn.BatchNorm3d(self.pc4x)
        ])

        if self.args.use_conv3d_preenc:
            self.conv3d_enc = nn.ModuleList([
                nn.Conv3d(self.c1x, self.c1x, kernel_size=(3, 3, 3), padding=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # HW 64 -> 64; T -> T
                nn.Conv3d(self.c2x, self.c2x, kernel_size=(3, 3, 3), padding=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # HW 32 -> 32; T -> T
                nn.Conv3d(self.c4x, self.c4x, kernel_size=(3, 3, 3), padding=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate') # HW 16 -> 16; T -> T
            ])

            self.bn_conv3d_enc = nn.ModuleList([
                nn.BatchNorm3d(self.c1x),
                nn.BatchNorm3d(self.c2x),
                nn.BatchNorm3d(self.c4x)
            ])

        # Sequential ConvLSTM Encoders
        self.f_convlstm2d_enc = nn.ModuleList([
            # 16
            ConvLSTMCell(self.args, self.c1x, self.c1x, kernel_size=(3, 3), bias=True),
            ConvLSTMCell(self.args, self.c1x, self.c1x, kernel_size=(3, 3), bias=True),
            # 32
            ConvLSTMCell(self.args, self.c2x, self.c2x, kernel_size=(3, 3), bias=True),
            ConvLSTMCell(self.args, self.c2x, self.c2x, kernel_size=(3, 3), bias=True),
            # 64
            ConvLSTMCell(self.args, self.c4x, self.c4x, kernel_size=(3, 3), bias=True),
            ConvLSTMCell(self.args, self.c4x, self.c4x, kernel_size=(3, 3), bias=True)
            ])

        self.bn_f_convlstm2d_enc = nn.ModuleList([
            nn.BatchNorm2d(self.c1x),
            nn.BatchNorm2d(self.c1x),
            nn.BatchNorm2d(self.c2x),
            nn.BatchNorm2d(self.c2x),
            nn.BatchNorm2d(self.c4x),
            nn.BatchNorm2d(self.c4x)
            ])

        self.b_convlstm2d_enc = nn.ModuleList([
            # 16
            ConvLSTMCell(self.args, self.c1x, self.c1x, kernel_size=(3, 3), bias=True),
            ConvLSTMCell(self.args, self.c1x, self.c1x, kernel_size=(3, 3), bias=True),
            # 32
            ConvLSTMCell(self.args, self.c2x, self.c2x, kernel_size=(3, 3), bias=True),
            ConvLSTMCell(self.args, self.c2x, self.c2x, kernel_size=(3, 3), bias=True),
            # 64
            ConvLSTMCell(self.args, self.c4x, self.c4x, kernel_size=(3, 3), bias=True),
            ConvLSTMCell(self.args, self.c4x, self.c4x, kernel_size=(3, 3), bias=True)
            ])

        self.bn_b_convlstm2d_enc = nn.ModuleList([
            nn.BatchNorm2d(self.c1x),
            nn.BatchNorm2d(self.c1x),
            nn.BatchNorm2d(self.c2x),
            nn.BatchNorm2d(self.c2x),
            nn.BatchNorm2d(self.c4x),
            nn.BatchNorm2d(self.c4x)
            ])

        # Merging hidden states between BiConvLSTM networks
        self.merge_hidden_enc = nn.ModuleList([
            nn.Conv2d(self.c1x + self.c1x, self.c1x, kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1), stride=(1, 1), padding_mode='replicate'), # 1st (1x: 64->64)
            nn.Conv2d(self.c1x + self.c1x, self.c1x, kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1), stride=(1, 1), padding_mode='replicate'), # 1st (1x: 64->64)
            nn.Conv2d(self.c2x + self.c2x, self.c2x, kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1), stride=(1, 1), padding_mode='replicate'), # 1st (1x: 32->32)
            nn.Conv2d(self.c2x + self.c2x, self.c2x, kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1), stride=(1, 1), padding_mode='replicate'), # 1st (1x: 32->32)
            nn.Conv2d(self.c4x + self.c4x, self.c4x, kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1), stride=(1, 1), padding_mode='replicate'), # 1st (1x: 16->16)
            nn.Conv2d(self.c4x + self.c4x, self.c4x, kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1), stride=(1, 1), padding_mode='replicate') # 1st (1x: 16->16)
        ])

        self.bn_merge_hidden_enc = nn.ModuleList([
            nn.BatchNorm2d(self.c1x),
            nn.BatchNorm2d(self.c1x),
            nn.BatchNorm2d(self.c2x),
            nn.BatchNorm2d(self.c2x),
            nn.BatchNorm2d(self.c4x),
            nn.BatchNorm2d(self.c4x)
        ])

        """
        Decoding Networks
        """
        # ConvLSTM Decoders
        self.convlstm2d_dec = nn.ModuleList([
            # 64
            ConvLSTMCell(self.args, self.c1x, self.c1x, kernel_size=(3  , 3), bias=True),
            ConvLSTMCell(self.args, self.c1x, self.c1x, kernel_size=(3, 3), bias=True),
            # 32
            ConvLSTMCell(self.args, self.c2x, self.c2x, kernel_size=(3, 3), bias=True),
            ConvLSTMCell(self.args, self.c2x, self.c2x, kernel_size=(3, 3), bias=True),
            # 16
            ConvLSTMCell(self.args, self.c4x, self.c4x, kernel_size=(3, 3), bias=True),
            ConvLSTMCell(self.args, self.c4x, self.c4x, kernel_size=(3, 3), bias=True)
        ])

        self.bn_convlstm2d_dec = nn.ModuleList([
            nn.BatchNorm2d(self.c1x),
            nn.BatchNorm2d(self.c1x),
            nn.BatchNorm2d(self.c2x),
            nn.BatchNorm2d(self.c2x),
            nn.BatchNorm2d(self.c4x),
            nn.BatchNorm2d(self.c4x)
        ])

        self.conv2d_dec = nn.ModuleList([
            nn.Conv3d(int(2*self.c1x), self.c1x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # HW 64 -> 64; T -> T
            nn.Conv3d(int(2*self.c1x), self.c1x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'),  # HW 64 -> 64; T -> T
            nn.Conv3d(int(2*self.c2x), self.c2x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # HW 32 -> 32; T -> T
            nn.Conv3d(int(2*self.c2x), self.c2x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # HW 32 -> 32; T -> T
            nn.Conv3d(self.c4x, self.c4x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # HW 16 -> 16; T -> T
            nn.Conv3d(int(2*self.c4x), self.c4x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate') # HW 16 -> 16; T -> T
        ])

        self.bn_conv2d_dec = nn.ModuleList([
            nn.BatchNorm3d(self.c1x),
            nn.BatchNorm3d(self.c1x),
            nn.BatchNorm3d(self.c2x),
            nn.BatchNorm3d(self.c2x),
            nn.BatchNorm3d(self.c4x),
            nn.BatchNorm3d(self.c4x)
        ])

        self.conv2d_dec_up = nn.ModuleList([
            nn.Conv3d(self.c1x, self.c1x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # 64 -> 64
            nn.ConvTranspose3d(self.c2x, self.c1x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 2, 2), output_padding=(0, 1, 1)), # 32 -> 64
            nn.ConvTranspose3d(self.c4x, self.c2x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 2, 2), output_padding=(0, 1, 1)), # 16 -> 32
            nn.Conv3d(self.c1x, 1, kernel_size=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate') # 32 -> 32
        ])

        self.bn_conv2d_dec_up = nn.ModuleList([
            nn.BatchNorm3d(self.c1x),
            nn.BatchNorm3d(self.c1x),
            nn.BatchNorm3d(self.c2x)
        ])

        self.conv2d_dec_res = nn.ModuleList([
            nn.Conv3d(int(2*self.c1x), self.c1x, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # 64 -> 64
            nn.Conv3d(int(2*self.c2x), self.c2x, kernel_size=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate'), # 32 -> 32
            nn.Conv3d(int(2*self.c4x), self.c4x, kernel_size=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), stride=(1, 1, 1), padding_mode='replicate') # 32 -> 32
        ])

        self.bn_conv2d_dec_res = nn.ModuleList([
            nn.BatchNorm3d(self.c1x),
            nn.BatchNorm3d(self.c2x),
            nn.BatchNorm3d(self.c4x)
        ])
        # Initialize hidden states
        _ = self.init_hidden()

    def autoencoder(self, x_enc_in):
        # Pre-encode the inputs to (16, 32, 64) for processing to ConvLSTM later.
        # Inputs are of the shape (T, batch, c=1, h, w)
        x_enc_in = x_enc_in.permute(1, 2, 0, 3, 4) # Reshape from (T, b, c, h, w) -> (b, c, T, h, w)
        convlstm_enc_in = [None for i in range(len(self.conv2d_enc))]

        """
        pre-ConvLSTM Encoder
        """
        # 1x 2D Conv
        out = self.cnn2d_enc(x_enc_in, idx=0) # b, 1, T, 32, 32 -> b, 8, T, 64, 64
        convlstm_enc_in[0] = self.cnn2d_enc(out, idx=1) # b, 1, T, 32, 32 -> b, 8, T, 64, 64
        # CHANNEL c1x
        convlstm_enc_in[0] = torch.cat([convlstm_enc_in[0], out], dim=1)

        # 2x 2D Conv
        out = self.conv2d_maxpool[0](convlstm_enc_in[0])
        convlstm_enc_in[2] = self.cnn2d_enc(out, idx=2)  # b, 8, T, 64, 64 -> b, 16, T, 32, 32
        out = self.cnn2d_enc(convlstm_enc_in[2], idx=3)  # b, 8, T, 64, 64 -> b, 16, T, 32, 32

        # CHANNEL c2x
        convlstm_enc_in[2] = torch.cat([convlstm_enc_in[2], out], dim=1)
        # 4x 2D Conv
        out = self.conv2d_maxpool[1](convlstm_enc_in[2])
        convlstm_enc_in[4] = self.cnn2d_enc(out, idx=4)  # b, 8, T, 64, 64 -> b, 16, T, 32, 32
        out = self.cnn2d_enc(convlstm_enc_in[4], idx=5)  # b, 8, T, 64, 64 -> b, 16, T, 32, 32
        # CHANNEL c4x
        convlstm_enc_in[4] = torch.cat([convlstm_enc_in[4], out], dim=1)

        # 3D
        if self.args.use_conv3d_preenc:
            # CHANNEL c1x
            convlstm_enc_in[1] = self.cnn3d_enc(convlstm_enc_in[0], idx=0)  # b, 8, T, 64, 64 -> b, 16, T, 32, 32
            # CHANNEL c2x
            convlstm_enc_in[3] = self.cnn3d_enc(convlstm_enc_in[2], idx=1)  # b, 8, T, 64, 64 -> b, 16, T, 32, 32
            # CHANNEL c4x
            convlstm_enc_in[5] = self.cnn3d_enc(convlstm_enc_in[4], idx=2)  # b, 8, T, 64, 64 -> b, 16, T, 32, 32

        """
        ConvLSTM Encoder-Decoder Framework
        """
        # From here we work in Time Major
        # Iterate over each depth/time in the representations learned using ConvLSTM
        # to learn long term dependencies
        fhidden_enc = [None for i in range(len(self.f_convlstm2d_enc))]
        bhidden_enc = [None for i in range(len(self.b_convlstm2d_enc))]
        hidden_dec = [None for i in range(len(self.convlstm2d_dec))]
        convlstm_res_out = [None for i in range(len(self.merge_hidden_enc))]
        # NOTE: Time major convention here across all loop operations.
        # 1st LAYER ConvLSTM Encoder: Forward and Backward ConvLSTM Processing (T=In Seq Len)
        # Output is just the hidden output from LSTM, not the cell state/tuple
        # In: b, 8, T, 64, 64
        # Out: # T, b, 8, 64, 64
        idx = [0, 2, 4]
        in_idxs = [0, 2, 4]
        if self.args.use_conv3d_preenc:
            in_idxs = [1, 3, 5]
        for i, id in enumerate(idx):
            # First Encoder Stage
            fhidden_enc[id], bhidden_enc[id] = self.loop_biconvlstm_enc(convlstm_enc_in[in_idxs[i]], id)
            # Merge ConvLSTM hidden layers from first encoder layer
            convlstm_res_out[id] = self.merge_biconvlstm_next(fhidden_enc[id], bhidden_enc[id], id)
            # Second Encoder Stage
            fhidden_enc[id+1], bhidden_enc[id+1] = self.loop_biconvlstm_enc_2(fhidden_enc[id], bhidden_enc[id], id+1)
            # Merge ConvLSTM hidden layers from second encoder layer
            convlstm_res_out[id+1] = self.merge_biconvlstm_next(fhidden_enc[id+1], bhidden_enc[id+1], id+1, last_only=True)

            # Set hidden values of forward encoder to the decoder
            self.hidden_dec[id] = self.fhidden_enc[id]
            self.hidden_dec[id+1] = self.fhidden_enc[id+1]
            # Decoder 1 # Returns T, b, c, h, w
            convlstm_res_out[id] = self.loop_convlstm_dec(convlstm_res_out[id+1], idx=id, time_stack=0)
            # Decoder 2 # Returns b, c, T, h, w
            convlstm_res_out[id+1] = self.loop_convlstm_dec2(convlstm_res_out[id], idx=id+1, time_stack=-3)
            convlstm_res_out[id] = convlstm_res_out[id].permute(1, 2, 0, 3, 4)

        """
        post-ConvLSTM Decoders for final representation
        """
        # Residual connection from pre-ConvLSTM outputs
        # In: b, 16, T, 16, 16 + b, 16, T, 16, 16
        # Out: b, 16, T, 16, 16
        postcnn_res_out = [None for i in range(len(self.conv2d_dec_up))]

        # 4x Scale Up
        # ConvLSTM Merge
        out = torch.cat([convlstm_res_out[5], convlstm_res_out[4]], dim=1) # CHANNEL 2*c4x
        out = self.cnn2d_dec(out, idx=5, bn=True, act=True) # Channel: 2*c4x -> c4x
        out = self.cnn2d_dec(out, idx=4, bn=True, act=True) # c4x -> c4x
        # Long Term Residual
        out = self.cnn2d_dec_res(torch.cat([out, convlstm_enc_in[4]], dim=1), idx=2, bn=True, act=True)
        # Upscale
        up_out = self.cnn2d_dec_up(out, idx=2, bn=False, act=False, last=False) # c4x -> c2x
        # 2x Scale Up
        # CHANNEL 2*c2x
        # ConvLSTM Merge
        out = torch.cat([convlstm_res_out[3], convlstm_res_out[2]], dim=1)
        out = self.cnn2d_dec(out, idx=3, bn=True, act=True) # Channel: 2*c2x -> c2x
        out = self.cnn2d_dec_res(torch.cat([out, convlstm_enc_in[2]], dim=1), idx=1, bn=True, act=True)
        # Upscale Merge
        out = torch.cat([out, up_out], dim=1) # # Channel: 2*c2x -> 2*c2x
        out = self.cnn2d_dec(out, idx=2, bn=True, act=True) # 2*c2x -> c2x
        # Upscale
        up_out = self.cnn2d_dec_up(out, idx=1, bn=False, act=False, last=False) # c2x -> c1x

        # 1x Scale Up
        # CHANNEL 2*c1x
        # ConvLSTM Merge
        out = torch.cat([convlstm_res_out[1], convlstm_res_out[0]], dim=1) # 2*c1x
        out = self.cnn2d_dec(out, idx=1, bn=True, act=True) # Channel: 2*c1x -> c1x
        if self.args.final_residue:
            res_out = out
        out = self.cnn2d_dec_res(torch.cat([out, convlstm_enc_in[0]], dim=1), idx=0, bn=True, act=True)
        # Upscale Merge
        out = torch.cat([out, up_out], dim=1) # # Channel: 2*c1x -> 2*c1x
        out = self.cnn2d_dec(out, idx=0, bn=True, act=True) # 2*c1x -> c1x
        # Residual
        if self.args.final_residue:
            out = torch.add(res_out, out)
            out = nn.LeakyReLU()(out)
        # Final Prediction
        out = self.cnn2d_dec_up(out, idx=0, bn=True, act=True, last=False) # c1x -> c1x

        out = self.cnn2d_dec_up(out, idx=3, bn=False, act=False, last=True) # c1x -> c1x
        out = torch.squeeze(out, dim=-4)

        return out

    """
    Functions
    """

    def cnn3d_enc(self, input, idx):
        out = self.conv3d_enc[idx](input)
        out = self.bn_conv3d_enc[idx](out)
        out = nn.LeakyReLU()(out)
        out = F.dropout3d(out, p=self.args.enc_droprate, training=self.args.mcmcdrop)

        return out

    def cnn2d_enc(self, input, idx):
        out = self.conv2d_enc[idx](input)
        out = self.bn_conv2d_enc[idx](out)
        out = nn.LeakyReLU()(out)
        out = F.dropout3d(out, p=self.args.enc_droprate, training=self.args.mcmcdrop)

        return out

    def cnn2d_dec(self, input, idx, bn=True, act=True):
        out = self.conv2d_dec[idx](input)
        if bn:
            out = self.bn_conv2d_dec[idx](out)
        if act:
            out = nn.LeakyReLU()(out)
        out = F.dropout3d(out, p=self.args.enc_droprate, training=self.args.mcmcdrop)

        return out

    def cnn2d_dec_up(self, input, idx, bn=False, act=False, last=False):
        out = self.conv2d_dec_up[idx](input)
        if bn:
            out = self.bn_conv2d_dec_up[idx](out)
        if act:
            out = nn.LeakyReLU()(out)
        if not last:
            out = F.dropout3d(out, p=self.args.enc_droprate, training=self.args.mcmcdrop)
        else:
            out = out
        return out

    def cnn2d_dec_res(self, input, idx, bn=False, act=False):
        out = self.conv2d_dec_res[idx](input)
        if bn:
            out = self.bn_conv2d_dec_res[idx](out)
        if act:
            out = nn.LeakyReLU()(out)
        out = F.dropout3d(out, p=self.args.enc_droprate, training=self.args.mcmcdrop)

        return out


    def loop_biconvlstm_enc(self, input, idx):
        foutput = []
        boutput = []

        # Reshape from  (b, c, T, h, w) -> (T, b, c, h, w)
        input = input.permute(2, 0, 1, 3, 4)

        for t in range(self.args.in_seq_len):
            # Forward and Backward ConvLSTM Encoders
            self.fhidden_enc[idx] = self.f_convlstm2d_enc[idx](input[t], self.fhidden_enc[idx])
            self.bhidden_enc[idx] = self.b_convlstm2d_enc[idx](input[self.args.in_seq_len - t - 1], self.bhidden_enc[idx])

            # Only stack the hidden state, and not the cell state
            foutput.append(self.fhidden_enc[idx][0])
            boutput.append(self.bhidden_enc[idx][0])

        foutput = torch.stack(foutput, dim=0)
        boutput = torch.stack(boutput, dim=0)

        return foutput, boutput

    def loop_biconvlstm_enc_2(self, res_finput, res_binput, idx):
        foutput = []
        boutput = []

        for t in range(self.args.in_seq_len):
            # Forward and Backward ConvLSTM Encoders
            self.fhidden_enc[idx] = self.f_convlstm2d_enc[idx](res_finput[t], self.fhidden_enc[idx])
            self.bhidden_enc[idx] = self.b_convlstm2d_enc[idx](res_binput[t], self.bhidden_enc[idx])

            # Only stack the hidden state, and not the cell state
            foutput.append(self.fhidden_enc[idx][0])
            boutput.append(self.bhidden_enc[idx][0])

        foutput = torch.stack(foutput, dim=0)
        boutput = torch.stack(boutput, dim=0)

        return foutput, boutput

    def merge_biconvlstm_next(self, fhidden_out, bhidden_out, idx, last_only=False):
        merged_output = []
        if last_only:
            hidden = self.merge_hidden_enc[idx](torch.cat((fhidden_out[-1], bhidden_out[-1]), dim=-3))
            hidden = self.bn_merge_hidden_enc[idx](hidden)

            hidden = nn.LeakyReLU()(hidden)
            hidden = F.dropout2d(hidden, p=self.args.enc_droprate, training=self.args.mcmcdrop)
            merged_output = hidden
        else:
            for t in range(self.args.in_seq_len):
                # fhidden_out[t] = self.bn_f_convlstm2d_enc[idx](fhidden_out[t])
                # bhidden_out[t] = self.bn_b_convlstm2d_enc[idx](bhidden_out[t])
                # Process the hidden state and not the cell state
                hidden = self.merge_hidden_enc[idx](torch.cat((fhidden_out[t], bhidden_out[t]), dim=-3))
                hidden = self.bn_merge_hidden_enc[idx](hidden)

                hidden = nn.LeakyReLU()(hidden)
                hidden = F.dropout2d(hidden, p=self.args.enc_droprate, training=self.args.mcmcdrop)

                merged_output.append(hidden)

            merged_output = torch.stack(merged_output, dim=0)

        return merged_output

    def loop_convlstm_dec(self, input, idx, time_stack):
        # Decoder 1
        output = []
        for t in range(self.args.out_seq_len):
            self.hidden_dec[idx] = self.convlstm2d_dec[idx](input, self.hidden_dec[idx])
            output.append(self.hidden_dec[idx][0])

        output = torch.stack(output, dim=time_stack)

        return output

    def loop_convlstm_dec2(self, input, idx, time_stack):
        output = []
        for t in range(self.args.out_seq_len):
            # Decoder 2
            self.hidden_dec[idx] = self.convlstm2d_dec[idx](input[t], self.hidden_dec[idx])
            output.append(self.hidden_dec[idx][0])

        output = torch.stack(output, dim=time_stack)

        return output

    def init_hidden(self):
        # Sample based hidden encoders
        self.fhidden_enc = [None for i in range(len(self.f_convlstm2d_enc))]
        self.bhidden_enc = [None for i in range(len(self.b_convlstm2d_enc))]
        self.hidden_dec = [None for i in range(len(self.convlstm2d_dec))]

        return

    def forward(self, xb, idx=None):
        seq_outputs = []
        x_enc = xb

        # Initialize hidden states
        self.init_hidden()
        # print(f'Before reshape: {x_enc[1].shape}')
        # Change from (b, t, h, w) to (t, b, h, w)
        x_enc = torch.transpose(x_enc, 1, 0)
        # print(f'After time major: {x_enc.shape}')
        # Change from (t, b, h, w) to (t, b, c=1, h, w)
        x_enc = x_enc.unsqueeze(-3)
        # print(f'After new dim: {x_enc.shape}')
        # print(f'Main forward model: {x_enc.shape}')
        seq_outputs = self.autoencoder(x_enc)

        if idx is not None:
            return seq_outputs, idx
        else:
            return seq_outputs
