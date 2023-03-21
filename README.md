# `iconv_dft`: PyTorch Invertible Convolution by DFT

This repo implements an invertible convolution operator in PyTorch. Please refer to the corresponding [report]() for the derviation of the invertible convolution using DFT.

Basically it allows to invert an output feature map to its input in a following simple manner:

```
B, C, H, W = 2, 3, 4, 6
x = torch.randn(B, C, H, W)
conv = iConv2d(conv_channels=C, kernel_size=3, x_mode=['WSWS', 'WSWS'], w_mode=['S', 'S'])
y, y_mode = conv(x)
z, z_mode = conv.inverse(y)
assert (x - z).norm() < 1e-4
```

## Remarks

* Invertibility can be achieved if the input is padded circularly, symmetrically, or anti-symmetrically. If the padding mode is symmetric or anti-symmetric, the kernel weight should also be reflected to be of some symmetric form, c.f. the [report]().
* Only supports even size of input.
* The underlying forward pass is just the PyTorch vanilla convolution. So compare to other invertible convolution operators like coupling layer, it has no extra structures and retains most similar behaviour to normal convolution.
* Be careful about the numerical stability of inversion when stacking many layers together.
