from torch.nn import Conv3d as TorchConv3D
from torch import from_numpy
import numpy as np
import unittest
from Conv3D import Conv3D
from numpy.testing import assert_array_equal


class TestConv3D(unittest.TestCase):
    def test_kernel_size_2(self):
        input, weight, bias = self.__get_data()

        own = Conv3D(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
        )

        torch = TorchConv3D(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
        )

        self.__assert(own, torch, input, weight, bias)

    def test_stride_2(self):
        input, weight, bias = self.__get_data()

        own = Conv3D(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
            stride=2,
        )

        torch = TorchConv3D(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
            stride=2,
        )

        self.__assert(own, torch, input, weight, bias)

    def test_padding_1(self):
        input, weight, bias = self.__get_data()

        own = Conv3D(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
            padding=1,
        )

        torch = TorchConv3D(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
            padding=1,
        )

        self.__assert(own, torch, input, weight, bias)

    def test_padding_2(self):
        input, weight, bias = self.__get_data()

        own = Conv3D(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
            padding=2,
        )

        torch = TorchConv3D(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
            padding=2,
        )

        self.__assert(own, torch, input, weight, bias)

    def test_dilation_2(self):
        input, weight, bias = self.__get_data()

        own = Conv3D(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
            dilation=2,
        )

        torch = TorchConv3D(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
            dilation=2,
        )

        self.__assert(own, torch, input, weight, bias)

    def test_bias(self):
        input, weight, bias = self.__get_data(bias=[1, 1])

        own = Conv3D(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
            bias=True,
        )

        torch = TorchConv3D(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
            bias=True,
        )

        self.__assert(own, torch, input, weight, bias)

    def test_padding_mode(self):
        input, weight, bias = self.__get_data(bias=[1, 1])

        own = Conv3D(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
            padding=1,
            padding_mode='replicate'
        )

        torch = TorchConv3D(
            in_channels=2,
            out_channels=2,
            kernel_size=2,
            padding=1,
            padding_mode='replicate'
        )

        self.__assert(own, torch, input, weight, bias)

    def __assert(self, own, torch, input, weight, bias):
        torch.weight.data = from_numpy(weight).float()
        torch.bias.data = from_numpy(bias).float()

        assert_array_equal(
            np.floor(own.forward(input, weight, bias)),
            np.floor(torch(from_numpy(input).float()).detach().numpy())
        )

    def __get_data(self, bias=None):
        ch1 = [[[1,   2,  3,  4],
                [5,   6,  7,  8],
                [9,  10, 11, 12],
                [13, 14, 15, 16]],
               [[1,   2,  3,  4],
                [5,   6,  7,  8],
                [9,  10, 11, 12],
                [13, 14, 15, 16]],
               [[1,   2,  3,  4],
                [5,   6,  7,  8],
                [9,  10, 11, 12],
                [13, 14, 15, 16]],
               [[1,   2,  3,  4],
                [5,   6,  7,  8],
                [9,  10, 11, 12],
                [13, 14, 15, 16]]]

        ch2 = [[[1,   2,  3,  4],
                [1,   2,  7,  8],
                [9,  10, 11, 12],
                [13, 14, 15, 16]],
               [[1,   2,  3,  4],
                [1,   2,  7,  8],
                [9,  10, 11, 12],
                [13, 14, 15, 16]],
               [[1,   2,  3,  4],
                [1,   2,  7,  8],
                [9,  10, 11, 12],
                [13, 14, 15, 16]],
               [[1,   2,  3,  4],
                [1,   2,  7,  8],
                [9,  10, 11, 12],
                [13, 14, 15, 16]]]

        wch1 = [[[1, 1],
                 [1, 1]],
                [[1, 1],
                 [1, 1]]]

        wch2 = [[[1, 1],
                 [1, 1]],
                [[1, 1],
                 [1, 1]]]

        input = np.array([[ch1, ch2]])
        weight = np.array([[wch1, wch2], [wch1, wch2]])
        bias = np.array([0, 0] if bias is None else bias)

        return input, weight, bias


if __name__ == '__main__':
    unittest.main()