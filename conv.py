import math
import torch
from torch import nn
import torch.nn.functional as F

def pad_dim(x, dim, length):
    ks = x.shape[dim]
    zero_shape = list(x.shape)
    zero_shape[dim] = length - ks
    zeros = x.new_zeros(*zero_shape)
    padded = torch.cat([x, zeros], dim=dim)
    padded = padded.roll(shifts=[-(ks // 2)], dims=[dim])
    return padded

def pad_weight(x, lengths):
    for i, length in enumerate(lengths):
        dim = x.dim() - len(lengths) + i
        x = pad_dim(x, dim=dim, length=length)
    return x

def reflect_dim(x, dim, mode):
    if mode == 'circular':
        return torch.cat([x, x], dim=dim)
    assert mode[1] == mode[3]
    reflected = x.flip(dims=[dim])
    sum_shape = (*x.shape[:dim], 1, *x.shape[dim + 1:])
    reshaped = x[None, ..., None].flatten(0, dim).flatten(2, -1)
    sum0 = reshaped[:, 0::2].sum(dim=1).view(sum_shape)
    sum1 = reshaped[:, 1::2].sum(dim=1).view(sum_shape)
    zeros = torch.zeros_like(sum0)
    if mode[2:] == 'WS':
        reflected = reflected.narrow(dim=dim, start=1, length=x.shape[dim] - 1)
        x = torch.cat([x, reflected], dim=dim)
    elif mode[2:] == 'HS':
        x = torch.cat([x, reflected], dim=dim)
    elif mode[2:] == 'ZS':
        x = torch.cat([x, -sum0 * 2, reflected], dim=dim)
    elif mode[2:] == 'HA':
        x = torch.cat([x, -reflected], dim=dim)
    elif mode[2:] == 'WA':
        x = torch.cat([x, zeros, -reflected], dim=dim)
    else:
        raise NotImplementedError
        
    if mode[:2] == 'WS':
        x = x.narrow(dim=dim, start=0, length=x.shape[dim] - 1)
    elif mode[:2] == 'HS':
        pass
    elif mode[:2] == 'ZS':
        x = torch.cat([x, -sum1 * 2], dim=dim)
    elif mode[:2] == 'HA':
        pass
    elif mode[:2] == 'WA':
        x = torch.cat([x, zeros], dim=dim)
    else:
        raise NotImplementedError

    return x

def reflect_tensor(x, modes):
    assert x.dim() >= len(modes)
    for i, mode in enumerate(modes):
        dim = x.dim() - len(modes) + i
        x = reflect_dim(x, dim=dim, mode=mode)

    return x

def crop_dim(x, dim, mode):
    length = x.shape[dim]
    assert length % 2 == 0
    if mode.lower() == 'circular':
        length //= 2
        x = x.narrow(dim=dim, start=0, length=length)
        return x

    if mode[:2] == 'WS':
        length += 1
    elif mode[:2] == 'HS':
        pass
    elif mode[:2] == 'ZS':
        length = length - 1
    elif mode[:2] == 'HA':
        pass
    elif mode[:2] == 'WA':
        length -= 1
    else:
        raise NotImplementedError

    if mode[2:] == 'WS':
        length += 1
    elif mode[2:] == 'HS':
        pass
    elif mode[2:] == 'ZS':
        length = length - 1
    elif mode[2:] == 'HA':
        pass
    elif mode[2:] == 'WA':
        length -= 1
    else:
        raise NotImplementedError
    length //= 2
    x = x.narrow(dim=dim, start=0, length=length)
    return x

def crop_tensor(x, modes):
    for i, mode in enumerate(modes):
        dim = x.dim() - len(modes) + i
        x = crop_dim(x, dim, mode)
    return x

def reflect_add_tensor(x, modes):
    signs = {'S': 1, 'A': -1}
    for i, mode in enumerate(modes):
        dim = x.dim() - len(modes) + i

        # zero_shape = list(x.shape)
        # zero_shape[dim] = 0#  x.shape[dim] % 2
        # zeros = x.new_zeros(*zero_shape)
        # x = torch.cat([x, zeros], dim=dim)
        reflected = x.flip(dims=[dim])
        if mode == 'S':
            x = x + reflected
        else:
            x = x - reflected
    return x

class FlipAdd2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, modes):
        ctx.modes = modes
        for i, mode in enumerate(ctx.modes):
            dim = x.dim() - len(ctx.modes) + i
            reflected = x.flip(dims=[dim])
            if mode == 'S':
                x = x + reflected
            elif mode == 'A':
                x = x - reflected
        return x

    @staticmethod
    def backward(ctx, g):
        for i, mode in enumerate(ctx.modes):
            dim = g.dim() - len(ctx.modes) + i
            reflected = g.flip(dims=[dim])
            if mode == 'S':
                g = g + reflected
            else:
                raise NotImplementedError
        return g, None

def get_symmetric_weight2d(w0):
    ks = w0.shape[-2:]
    split_r = ks[0] - ks[0] // 2
    split_c = ks[1] - ks[1] // 2
    w = w0[:, :, :split_r, :split_c]
    # if ks[0] % 2 == 1 and ks[1] % 2 == 1:
    #     w = F.pad(w, [0, ks[1] - split_c, 0, ks[0] - split_r], mode='reflect')
    #     return w
    w = torch.cat([w, w[:, :, :ks[0] - split_r, :].flip(-2)], dim=-2)
    w = torch.cat([w, w[:, :, :, :ks[1] - split_c].flip(-1)], dim=-1)
    return w


def complex_mv(m, v):
    real = m.real @ v.real - m.imag @ v.imag
    imag = m.real @ v.imag + m.imag @ v.real
    return torch.complex(real, imag)

def complex_inv(m):
    # Assumes that m is C x C x ...
    assert m.shape[0] == m.shape[1]
    # 2C x 2C x ...
    phi = torch.cat([torch.cat([ m.real, -m.imag], dim=1),
                     torch.cat([ m.imag,  m.real], dim=1)], dim=0)
    # Now ... x 2C x 2C
    phi = phi.movedim(source=[0, 1], destination=[-2, -1])
    inv = phi.inverse()
    # Now 2C x 2C x ...
    inv = inv.movedim(source=[-2, -1], destination=[0, 1])
    AB, _ = torch.chunk(inv, 2, dim=0)
    A, B = torch.chunk(AB, 2, dim=1)
    return torch.complex(A, B)

def fft_conv1d_circular(x, weight):
    f_x = torch.fft.fft(x)
    padded_w = pad_weight(weight, x.shape[2:])
    f_w = torch.fft.fft(padded_w)
    print('x', x)
    print('f_x', f_x)
    print('w', padded_w)
    print('f_w', f_w)
    
    f_x = f_x.permute(2, 1, 0) # L x C x B
    f_w = f_w.permute(2, 0, 1) # L x C x C
    f_y = complex_mv(f_w, f_x)
    f_y = f_y.permute(2, 1, 0) # B x C x L
    
    y = torch.fft.ifft(f_y)
    print('f_y', f_y)
    print('y', y)
    return y

def fft_conv1d_reflect(x, weight, mode, return_cropped=True):
    # assert mode in ['ZSZS', 'WAWA']
    reflected_x = reflect_tensor(x, modes=[mode])

    y = fft_conv1d_circular(reflected_x, weight)

    # padded_w = pad_weight(reflected_w, reflected_x.shape[2:])

    # f_x = torch.fft.fft(reflected_x)
    # print('fx', f_x[0, 0])
    # f_w = torch.fft.fft(padded_w)
    # print('fw', f_w[0, 0])

    # f_x = f_x.permute(2, 1, 0) # L x C x B
    # f_w = f_w.permute(2, 0, 1) # L x C x C
    # print(f_w.shape, f_x.shape)
    # f_y = complex_mv(f_w, f_x)
    # f_y = f_y.permute(2, 1, 0) # B x C x L
    # print('fy', f_y[0, 0])
    
    # y = torch.fft.ifft(f_y)
    if return_cropped:
        y = crop_tensor(y, modes=[mode])

    return y

def fft_conv2d_reflect(x, weight, modes):
    reflected_x = reflect_tensor(x, modes=modes)
    reflected_w = reflect_add_tensor(weight, modes=[modes[0][1], modes[1][1]])

    padded_w = pad_weight(reflected_w, reflected_x.shape[-2:])

    f_x = torch.fft.fft2(reflected_x)
    print('reflected_x', reflected_x[0, 0])
    print('fx', f_x[0, 0])
    f_w = torch.fft.fft2(padded_w)
    print('fw', f_w[0, 0])

    f_x = f_x.permute(2, 3, 1, 0) # H x W x C x B
    f_w = f_w.permute(2, 3, 0, 1) # H x W x C x C
    print(f_w.shape, f_x.shape)
    f_y = complex_mv(f_w, f_x) # H x W x C x B
    f_y = f_y.permute(3, 2, 0, 1) # B x C x H x W
    print('fy', f_y[0, 0])
    
    y = torch.fft.ifft2(f_y)
    print('y', y)
    return y

def fft_invconv2d_reflect(y, weight, modes):
    y = reflect_tensor(y, modes=['HSWS', 'HSHS'])
    weight = reflect_add_tensor(weight, modes=[modes[0][1], modes[1][1]])
    padded_w = pad_weight(weight, lengths=y.shape[-2:])

    f_y_ = torch.fft.fft2(y)
    f_w_ = torch.fft.fft2(padded_w)
    mask = f_w_.norm(dim=[0, 1]) > 1e-5
    f_iw = complex_inv(f_w_[..., mask])
    f_y = f_y_[..., mask]
    f_y = f_y.permute(2, 1, 0) # mask.sum() x C x B
    f_iw = f_iw.permute(2, 0, 1) # mask.sum() x C x C
    print(f_iw.shape, f_y.shape)
    f_x = complex_mv(f_iw, f_y) # mask.sum() x C x B
    f_x = f_x.permute(2, 1, 0) # B x C x mask.sum()
    f_x = torch.zeros_like(f_y_).masked_scatter(mask, f_x) # B x C x H x W
    
    x = torch.fft.ifft2(f_x)
    x = crop_tensor(x, modes=modes)
    return x


        
def test_fft_conv1d_inv():
    channels = 1
    x = torch.randn(1, channels, 4)
    w = torch.randn(channels, channels, 3)
    kl = w.shape[-1]

    reflected_w = reflect_add_tensor(w, modes=['A'])
    reflected_x = reflect_tensor(x, modes=['ZSZS'])
    reflected_x = reflected_x.roll(kl // 2, -1)
    reflected_x = reflected_x[:, :, :x.shape[-1] + kl - 1]

    y_gt = F.conv1d(reflected_x, reflected_w)
    y_fft = fft_conv1d_reflect(x, w, mode='A')
    import pdb; pdb.set_trace()

def test_fft_conv1d_comb():
    channels = 1
    x = torch.randn(1, channels, 5)
    w0 = torch.randn(channels, channels, 4)
    w1 = torch.randn(channels, channels, 4)
    w0 = reflect_add_tensor(w0, 'A')
    w1 = reflect_add_tensor(w1, 'A')

    y0 = fft_conv1d_reflect(x, w0, mode='WAWA')
    quit()
    import pdb; pdb.set_trace()
    print('---------')
    y1 = fft_conv1d_reflect(y0, w1, mode='WAWA')
    print('---------')

    w0 = pad_weight(w0, torch.tensor(x.shape[2:]) * 2 + 2)
    w2 = fft_conv1d_circular(w0, w1)
    print('---------')
    y2 = fft_conv1d_reflect(x, w2, 'ZSZS')
    print('---------')
    import pdb; pdb.set_trace()

def test_fft_conv1d_ha_inv():
    channels = 3
    x = torch.randn(5, channels, 4)
    w = torch.randn(channels, channels, 3)
    kl = w.shape[-1]

    reflected_w = reflect_add_tensor(w, signs=[-1], shifts=[0])
    reflected_x = reflect_tensor(x, modes=['HAHA'])
    reflected_x = reflected_x.roll(kl // 2, -1)
    reflected_x = reflected_x[:, :, :x.shape[-1] + kl - 1]

    y_gt = F.conv1d(reflected_x, reflected_w)
    y_fft = fft_conv1d_reflect(x, w, mode='HAHA')
    import pdb; pdb.set_trace()

def test_fft_conv2d_hs_inv():
    channels = 1
    x = torch.randn(1, channels, 4, 5)
    w = torch.randn(channels, channels, 3, 3)
    kl = w.shape[2:]
    modes = ['WAWA', 'WSWS']
    modes = ['WAWA', 'HSHS']
    x_modes = modes
    w_modes = [modes[0][1], modes[1][1]]

    reflected_w = reflect_add_tensor(w, modes=w_modes)
    reflected_x = reflect_tensor(x, modes=modes)
    reflected_x = reflected_x.roll(shifts=[kl[0] // 2, kl[1] // 2], dims=[-2, -1])
    reflected_x = reflected_x[:, :, :x.shape[-2] + kl[0] - 1, :x.shape[-1] + kl[1] - 1]

    y_gt = iconv2d_forward(x, reflected_w, x_mode=x_modes, w_mode=w_modes)
    y_fft = fft_conv2d_reflect(x, w, modes=modes)
    import pdb; pdb.set_trace()
    x_sol = fft_invconv2d_reflect(y_fft[:, :, :5, :5], w, modes=modes)
    print((x_sol.real - x).norm())
    import pdb; pdb.set_trace()

def get_output_mode(x_mode, w_mode):
    mode_map = {('circular', None): 'circular',
            ('HSHS', 'S'): 'HSHS', ('WSWS', 'S'): 'WSWS',
            ('ZSZS', 'S'): 'ZSZS', ('ZSZS', 'A'): 'WAWA',
            ('WAWA', 'S'): 'WAWA', ('WAWA', 'A'): 'ZSZS'}
    y_mode = [None for _ in x_mode]
    for i in range(len(y_mode)):
        assert x_mode[i] in ['circular', 'WSWS', 'HSHS', 'ZSZS', 'WAWA']
        y_mode[i] = mode_map[(x_mode[i], w_mode[i])]
    return y_mode

def get_weight_mode(w):
    assert w.ndim == 4
    w_mode = [None, None]
    for i in [0, 1]:
        if w.allclose(w.flip(i + 2)):
            w_mode[i] = 'S'
        elif w.allclose(-w.flip(i + 2)):
            w_mode[i] = 'A'
        else:
            w_mode[i] = None
    return w_mode

def iconv2d_forward(x, w, x_mode=None, w_mode=None, check_mode=True, check_invertibility=True):
    assert w.shape[0] == w.shape[1]
    if isinstance(x_mode, str):
        x_mode = [x_mode, x_mode]

    y_mode = get_output_mode(x_mode, w_mode)

    if all(_ == 'WSWS' for _ in x_mode) and all(_ == 'S' for _ in w_mode):
        ks = w.shape[-2:]
        top, left = ks[0] // 2, ks[1] // 2
        padded_x = F.pad(x, pad=[left, ks[1] - left - 1, top, ks[0] - top - 1],
                mode='reflect')
        y = F.conv2d(padded_x, w)
        return y, y_mode

    ks = w.shape[2:]
    padded = reflect_tensor(x, modes=x_mode)
    padded = padded.roll(shifts=[ks[0] // 2, ks[1] // 2], dims=[-2, -1])
    padded = padded[:, :, :x.shape[-2] + ks[0] - 1, :x.shape[-1] + ks[1] - 1]
    y = F.conv2d(padded, w)

    return y, y_mode

def pad_rfft2_inv(w, shape):
    padded_w = pad_weight(w, lengths=shape)
    f_w_ = torch.fft.rfft2(padded_w)
    mask = f_w_.norm(dim=[0, 1]) > 1e-5
    f_iw = complex_inv(f_w_[..., mask])
    return f_iw

def pad_rfft2_inv(w, shape):
    numel = math.prod(w.shape[:2] + shape)
    ref_numel = int(2e8)
    if numel < ref_numel:
        padded_w = pad_weight(w, lengths=shape[-2:])
        f_w = torch.fft.rfft2(padded_w)
        f_iw = complex_inv(f_w)
    else:
        w_ = w.flatten(0, 1)
        step = math.ceil(ref_numel * w_.shape[0] / numel)

        f_w = []
        for i in range(0, w_.shape[0], step):
            print(i, w_.shape[0])
            padded_w = pad_weight(w_[i:i + step], lengths=shape[-2:])
            f_w.append(torch.fft.rfft2(padded_w).cpu())

        f_w = torch.cat(f_w, dim=0)
        f_w = f_w.unflatten(0, w.shape[:2])
        f_w_ = f_w.flatten(-2, -1)

        step = math.ceil(ref_numel * f_w_.shape[-1] / numel)
        f_iw = []
        for i in range(0, f_w_.shape[-1], step):
            print(i, f_w_.shape[-1])
            temp = f_w_[..., i:i + step].to(w.device)
            f_iw.append(complex_inv(temp).cpu())
        f_iw = torch.cat(f_iw, dim=-1)
        f_iw = f_iw.unflatten(-1, f_w.shape[-2:])
        f_iw = f_iw.to(w.device)

        torch.cuda.empty_cache()

    return f_iw


def iconv2d_inverse(y, w, y_mode=None):
    '''
    This implementation assume symmetric weight, i.e. all elements in f_w is 
    non-zero.
    '''
    assert w.shape[0] == w.shape[1]
    if isinstance(y_mode, str):
        y_mode = [y_mode, y_mode]

    w_mode = get_weight_mode(w)
    assert all(_ == 'S' for _ in w_mode)
    x_mode = get_output_mode(y_mode, w_mode)

    y = reflect_tensor(y, modes=y_mode)
    f_y = torch.fft.rfft2(y)
    f_y = f_y.permute(2, 3, 1, 0) # H x W x C x B

    f_iw = pad_rfft2_inv(w, y.shape[-2:])
    f_iw = f_iw.permute(2, 3, 0, 1) # H x W x C x C

    print(f_iw.shape, f_y.shape)
    f_x = complex_mv(f_iw, f_y) # mask.sum() x C x B
    del f_iw
    f_x = f_x.permute(3, 2, 0, 1) # B x C x H x W
    
    x = torch.fft.irfft2(f_x, s=y.shape[-2:])
    x = crop_tensor(x, modes=x_mode)
    torch.cuda.empty_cache()
    return x, x_mode

@torch.no_grad()
def iconv2d_inverse_(y, w, y_mode=None):
    ''' A general version '''
    assert w.shape[0] == w.shape[1]
    if isinstance(y_mode, str):
        y_mode = [y_mode, y_mode]

    if y_mode == 'circular':
        w_mode = None
    else:
        w_mode = get_weight_mode(w)
    x_mode = get_output_mode(y_mode, w_mode)

    y = reflect_tensor(y, modes=y_mode)
    padded_w = pad_weight(w, lengths=y.shape[-2:])
    # import pdb; pdb.set_trace()

    f_y_ = torch.fft.rfft2(y)
    f_w_ = torch.fft.rfft2(padded_w)
    mask = f_w_.norm(dim=[0, 1]) > 1e-5
    f_iw = complex_inv(f_w_[..., mask])
    f_y = f_y_[..., mask]
    f_y = f_y.permute(2, 1, 0) # mask.sum() x C x B
    f_iw = f_iw.permute(2, 0, 1) # mask.sum() x C x C
    print(f_iw.shape, f_y.shape)
    f_x = complex_mv(f_iw, f_y) # mask.sum() x C x B
    f_x = f_x.permute(2, 1, 0) # B x C x mask.sum()
    f_x = torch.zeros_like(f_y_).masked_scatter(mask, f_x) # B x C x H x W
    
    x = torch.fft.irfft2(f_x, s=y.shape[-2:])
    x = crop_tensor(x, modes=x_mode)
    return x, x_mode

def iconv2d_inverse_fft(y, w, y_mode=None):
    '''
    I forget how this version is used
    '''
    assert w.shape[0] == w.shape[1]
    if isinstance(y_mode, str):
        y_mode = [y_mode, y_mode]

    w_mode = get_weight_mode(w)
    x_mode = get_output_mode(y_mode, w_mode)

    y = reflect_tensor(y, modes=y_mode)
    padded_w = pad_weight(w, lengths=y.shape[-2:])
    
    f_y_ = torch.fft.fft2(y)
    f_w_ = torch.fft.fft2(padded_w)
    mask = f_w_.norm(dim=[0, 1]) > 1e-5
    f_iw = complex_inv(f_w_[..., mask])
    f_y = f_y_[..., mask]
    f_y = f_y.permute(2, 1, 0) # mask.sum() x C x B
    f_iw = f_iw.permute(2, 0, 1) # mask.sum() x C x C
    print(f_iw.shape, f_y.shape)
    f_x = complex_mv(f_iw, f_y) # mask.sum() x C x B
    f_x = f_x.permute(2, 1, 0) # B x C x mask.sum()
    f_x = torch.zeros_like(f_y_).masked_scatter(mask, f_x) # B x C x H x W
    
    x = torch.fft.ifft2(f_x)
    x = crop_tensor(x, modes=x_mode)
    return x, x_mode

class iConv2d(nn.Module):
    def __init__(self, conv_channels, kernel_size, x_mode, w_mode=None, direct='forward'):
        super().__init__()
        self.conv_channels = conv_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.x_mode = x_mode
        self.w_mode = w_mode
        self.y_mode = get_output_mode(x_mode, w_mode)
        weight = torch.randn(conv_channels, conv_channels, *kernel_size)
        # nn.init.eye_(weight)
        self.weight = nn.Parameter(weight)
        self.direct = direct
        assert direct in ['forward', 'inverse']

        if direct == 'forward':
            self.forward, self.inverse = self._forward, self._inverse
        else:
            self.forward, self.inverse = self._inverse, self._forward

    def _forward(self, x):
        # w = reflect_add_tensor(self.weight, modes=self.w_mode)
        w = FlipAdd2d.apply(self.weight, self.w_mode)
        # assert all(_ == 'S' for _ in self.w_mode)
        # w = get_symmetric_weight2d(self.weight)
        y, y_mode = iconv2d_forward(x, w, x_mode=self.x_mode, w_mode=self.w_mode)
        return y, y_mode

    def _inverse(self, y):
        # w = reflect_add_tensor(self.weight, modes=self.w_mode)
        w = FlipAdd2d.apply(self.weight, self.w_mode)
        x, x_mode = iconv2d_inverse_(y, w, y_mode=self.y_mode)
        return x, x_mode


def test_iconv2d():
    B, C, H, W = 2, 3, 4, 5

    x = torch.randn(B, C, H, W)
    x_mode = ['WSWS', 'WSWS']
    conv = iConv2d(C, 3, x_mode, ['S', 'S'])
    y0 = conv(x)

    reflected_w = get_symmetric_weight2d(conv.weight)
    y1, y1_mode = iconv2d_forward(x, reflected_w, x_mode=x_mode)
    with torch.no_grad():
        z1, z1_mode = iconv2d_inverse(y1, reflected_w, y_mode=y1_mode)

    import pdb; pdb.set_trace()

    return 
    w = torch.randn(C, C, 3, 3)
    kl = w.shape[-1]

    reflected_w = reflect_add_tensor(w, modes=[x_mode[0][1], x_mode[1][1]])
    y, y_mode = iconv2d_forward(x, reflected_w, x_mode=x_mode)
    z, z_mode = iconv2d_inverse(y, reflected_w, y_mode=y_mode)
    import pdb; pdb.set_trace()

def test_flip_add():
    x0 = torch.randn(2, 3, 4, 5)
    x0.requires_grad_()
    x1 = x0.clone().detach()
    x1.requires_grad_()

    
    modes = ['S', 'S']
    w0 = reflect_add_tensor(x0, modes)
    w1 = FlipAdd2d.apply(x1, modes)

    l0 = w0.norm()
    l1 = w1.norm()

    l0.backward()
    l1.backward()

    import pdb; pdb.set_trace()

def test_partitioned_fft():
    x = torch.randn(4, 5, 3, 4)
    fx0 = torch.fft.rfft2(x)
    fx1 = [torch.fft.rfft2(x[i:i + 1]) for i in range(x.shape[0])]
    fx1 = torch.cat(fx1, dim=0)
    import pdb; pdb.set_trace()



if __name__ == '__main__':
    torch.set_printoptions(linewidth=280, sci_mode=False)
    seed = 234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    
    test_fft_conv1d_comb()
    import pdb; pdb.set_trace()
    test_fft_conv1d_inv()
    test_fft_conv2d_hs_inv()
    test_iconv2d()
    test_partitioned_fft()
    test_flip_add()



