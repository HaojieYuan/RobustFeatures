"""
https://raw.githubusercontent.com/zh217/torch-dct/master/torch_dct/_dct.py
"""
import numpy as np
import torch


def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] = V[:, 0] / np.sqrt(N) * 2
        V[:, 1:] = V[:, 1:] / np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] = X_v[:, 0] * np.sqrt(N) * 2
        X_v[:, 1:] = X_v[:, 1:] * np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.irfft(V, 1, onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] = x[:, ::2] + v[:, :N - (N // 2)]
    x[:, 1::2] = x[:, 1::2] + v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


class DCTTransform(object):
    def __init__(self, channels=3):
        self.channels = channels

    def dct_batch(self, data):
        """
        input: rgb_data
        return: dct_data
        """
        N, C, H, W  = data.size()
        new_data = data.clone()
        for b_ix in range(N):
            for c in range(self.channels):
                new_data[b_ix, c, :, :] = dct_2d(new_data[b_ix, c, :, :])
        return new_data

    def idct_batch(self, data):
        """
        input: dct_data
        return: rgb_data
        """
        N, C, H, W = data.size()
        new_data = data.clone()
        for b_ix in range(N):
            for c in range(self.channels):
                new_data[b_ix, c, :, :] = idct_2d(new_data[b_ix, c, :, :])
        return new_data

    def dct_mask(self, data, ratio, to_rgb=True):
        """
        input: rgb_data
        return: rgb_data (if to_rgb == True)
        """
        N, C, H, W = data.size()
        # frequency mask
        freq_mask = torch.zeros_like(data)
        fill_h = int(H * ratio)
        fill_w = int(W * ratio)
        freq_mask[:, :, :fill_h, :fill_w] = 1

        # rgb_noise -> dct_noise -> dct_mask -> rgb_noise
        data = self.dct_batch(data)
        data = data * freq_mask
        # else:
        #     raise ValueError('unknown dct modality[{}] in _dct_mask func'.format(self.modality))
        if to_rgb:
            data = self.idct_batch(data)
        return data

    def dct_split(self, data, ratio, to_rgb=True):
        """
        input: rgb_data
        return rgb_data, rgb_data (if to_rgb == True)
        """
        N, C, H, W = data.size()
        # frequency mask
        freq_mask_low = torch.zeros_like(data)
        freq_mask_high = torch.ones_like(data)
        fill_h = int(H * ratio)
        fill_w = int(W * ratio)
        freq_mask_low[:, :, :fill_h, :fill_w] = 1
        freq_mask_high[:, :, :fill_h, :fill_w] = 0

        # rgb_noise -> dct_noise -> dct_mask -> rgb_noise
        data = self.dct_batch(data)
        data_low = data * freq_mask_low
        data_high = data * freq_mask_high
        # else:
        #     raise ValueError('unknown dct modality[{}] in _dct_mask func'.format(self.modality))
        if to_rgb:
            data_low = self.idct_batch(data_low)
            data_high = self.idct_batch(data_high)
        return data_low, data_high


if __name__ == '__main__':
    x = torch.Tensor(1, 3, 112, 112)
    x.normal_(0, 1)
    transform = DCTTransform(channels=3)
    dct_data = transform.dct_batch(x)
    rgb_data = transform.idct_batch(dct_data)
    diff = torch.max(torch.abs(x - rgb_data))
    assert diff < 0.0001
    masked_data = transform.dct_mask(rgb_data, ratio=0.5, to_rgb=True)
    print('Done.')
