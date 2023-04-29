import torch


# Due to the difference in implementation method, there may be very slight differences in result values.
class SoftPooling1D(torch.nn.Module):
    def __init__(self, kernel_size, strides=None, padding=0, ceil_mode=False, count_include_pad=True):
        super(SoftPooling1D, self).__init__()
        self.avgpool = torch.nn.AvgPool1d(kernel_size, strides, padding, ceil_mode, count_include_pad)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, strides=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, strides, padding, ceil_mode, count_include_pad, divisor_override)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class SoftPooling3D(torch.nn.Module):
    def __init__(self, kernel_size, strides=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        super(SoftPooling3D, self).__init__()
        self.avgpool = torch.nn.AvgPool3d(kernel_size, strides, padding, ceil_mode, count_include_pad, divisor_override)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


def test():
    SoftPooling1D(2, 2)(torch.ones((1, 1, 32)))
    SoftPooling2D(2, 2)(torch.ones((1, 1, 32, 32)))
    SoftPooling3D(2, 2)(torch.ones((1, 1, 32, 32, 32)))