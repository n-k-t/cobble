from tensor import Tensor
import numpy as np
import unittest

class TestTensor(unittest.TestCase):
    def test_creation(self):
        a = np.arange(start = 0, stop = 10, step = 1)
        b = Tensor(data = a)

        np.testing.assert_almost_equal(actual = b.data.buffer, desired = a, decimal = 3)
    
    def test_shape(self):
        a = np.arange(start = 0, stop = 10, step = 1)
        b = Tensor(data = a)

        self.assertEqual(first = a.shape, second = b.data.buffer.shape)

if __name__ == '__main__':
    unittest.main()