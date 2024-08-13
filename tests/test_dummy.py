#!/usr/bin/python3

import unittest
import pixutils
import pixutils.fourcc_str

class TestDummy(unittest.TestCase):
    def test_dummy(self):
        self.assertEqual(pixutils.PixelFormats.RGB888.name, 'RGB888')

class TestFourcc(unittest.TestCase):
    def test_fourcc(self):
        self.assertEqual(pixutils.fourcc_str.str_to_fourcc('XR24'), 0x34325258)
        self.assertEqual(pixutils.fourcc_str.fourcc_to_str(0x34325258), 'XR24')
        with self.assertRaises(ValueError):
            pixutils.fourcc_str.str_to_fourcc('ABCDE')
        with self.assertRaises(ValueError):
            pixutils.fourcc_str.str_to_fourcc('ABC')

if __name__ == '__main__':
    unittest.main()
