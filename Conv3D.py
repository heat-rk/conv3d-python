import numpy as np


class Conv3D:
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            padding_mode='zeros',
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = (kernel_size, kernel_size, kernel_size)

        if isinstance(stride, tuple):
            self.stride = stride
        else:
            self.stride = (stride, stride, stride)

        if isinstance(padding, tuple):
            self.pad = padding
        elif padding == 'same':
            if self.stride[0] != 1 or self.stride[1] != 1 or self.stride[2] != 1:
                raise ValueError('padding == \'same\' can be applied only with stride = 1')
            self.pad = (self.kernel_size[0] - 1, self.kernel_size[1] - 1, self.kernel_size[2] - 1)
        elif padding == 'valid':
            self.pad = (0, 0, 0)
        else:
            self.pad = (padding, padding, padding)

        if isinstance(dilation, tuple):
            self.dilation = dilation
        else:
            self.dilation = (dilation, dilation, dilation)

        self.pad_mode = padding_mode
        self.bias = bias

    def forward(self, input, weight, bias):
        """
        :param input: array of input images of format NCDHW
        :param weight: array of learning weights
        :param bias: array of biases
        :return: output images
        """

        batches = len(input)
        out = []

        for b in range(batches):
            d_in = input[b].shape[1]
            h_in = input[b].shape[2]
            w_in = input[b].shape[3]

            if self.kernel_size[0] > h_in or self.kernel_size[1] > w_in or self.kernel_size[2] > d_in:
                raise ValueError('kernel size can\'t be greater than input size')

            d_out = int(
                (d_in + 2 * self.pad[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / (self.stride[0]) + 1)

            h_out = int(
                (h_in + 2 * self.pad[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / (self.stride[1]) + 1)

            w_out = int(
                (w_in + 2 * self.pad[2] - self.dilation[2] * (self.kernel_size[2] - 1) - 1) / (self.stride[2]) + 1)

            out.append(np.zeros((self.out_channels, d_out, h_out, w_out)))

            for c_out in range(self.out_channels):
                for z_out in range(d_out):
                    for y_out in range(h_out):
                        for x_out in range(w_out):
                            sum = 0
                            for c_in in range(self.in_channels):
                                for kernel_z in range(self.kernel_size[0]):
                                    for kernel_y in range(self.kernel_size[1]):
                                        for kernel_x in range(self.kernel_size[2]):
                                            z_in = z_out * self.stride[0] + kernel_z * self.dilation[0] - self.pad[0]
                                            y_in = y_out * self.stride[1] + kernel_y * self.dilation[1] - self.pad[1]
                                            x_in = x_out * self.stride[2] + kernel_x * self.dilation[2] - self.pad[2]
                                            if 0 <= z_in < d_in and 0 <= y_in < h_in and 0 <= x_in < w_in:
                                                sum += input[b][c_in][z_in][y_in][x_in] * weight[c_out][c_in][kernel_z][kernel_y][kernel_x]
                                            elif self.pad_mode == 'replicate':
                                                z_in = max(0, min(z_in, d_in - 1))
                                                y_in = max(0, min(y_in, h_in - 1))
                                                x_in = max(0, min(x_in, w_in - 1))
                                                sum += input[b][c_in][z_in][y_in][x_in] * weight[c_out][c_in][kernel_z][kernel_y][kernel_x]

                            out[b][c_out][z_out][y_out][x_out] = sum + (bias[c_out] if self.bias else 0)

        return np.array(out)



