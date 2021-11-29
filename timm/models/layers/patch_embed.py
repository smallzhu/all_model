""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""

from torch import nn as nn

from .helpers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        b_image_size = img_size[0]
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        #
        # self.pre = nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # # self.biLSTM_1 = BiLSTM(b_image_size, b_image_size * 2, b_image_size)
        # # self.biLSTM_2 = BiLSTM(b_image_size, b_image_size * 2, b_image_size)
        # self.biLSTM = nn.Sequential(BiLSTM(embed_dim, embed_dim, embed_dim),
        #                             BiLSTM(embed_dim, embed_dim, embed_dim))

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x = self.pre(x)
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = x.transpose(1, 2)
        # x = self.biLSTM_1(x)
        # x = self.biLSTM_2(x)
        # x = x.transpose(1, 2)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        # x = x.transpose(1,2)
        # x = self.biLSTM(x)
        # x = x.transpose(1,2)
        return x

class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=False)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

