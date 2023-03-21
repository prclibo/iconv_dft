import torch
from conv import iConv2d

def test_iconv2d():
    # torch.set_printoptions(precision=1, sci_mode=False, linewidth=250)
    B, C, H, W = 1, 1, 4, 6

    x = torch.randn(B, C, H, W)

    x_modes = ['WSWS', 'CSCS', 'WAWA']
    w_modes = ['S', 'A']

    modes = [
            (['WSWS', 'WSWS'], ['S', 'S']),
            (['WSWS', 'WAWA'], ['S', 'A']),
            (['WAWA', 'WSWS'], ['A', 'S']),
            (['WAWA', 'WAWA'], ['A', 'A']),
            (['HSHS', 'HSHS'], ['S', 'S']),
            (['circular', 'circular'], [None, None]),
    ]

    for x_mode, w_mode in modes:
        conv = iConv2d(C, 3, x_mode, w_mode)
        y, y_mode = conv(x)
        z, z_mode = conv.inverse(y)
        print(f'x_mode = {x_mode}, w_mode = {w_mode}, y_mode = {y_mode}, ', '|x - z| = ', (x - z).norm())

if __name__ == '__main__':
    test_iconv2d()
