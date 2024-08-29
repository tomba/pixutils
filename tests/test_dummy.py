#!/usr/bin/python3

import unittest
import pixutils
import pixutils.formats

class TestDummy(unittest.TestCase):
    def test_dummy(self):
        self.assertEqual(pixutils.formats.PixelFormats.RGB888.name, 'RGB888')

class TestFourcc(unittest.TestCase):
    def test_fourcc(self):
        self.assertEqual(pixutils.formats.fourcc_str.str_to_fourcc('XR24'), 0x34325258)
        self.assertEqual(pixutils.formats.fourcc_str.fourcc_to_str(0x34325258), 'XR24')
        with self.assertRaises(ValueError):
            pixutils.formats.fourcc_str.str_to_fourcc('ABCDE')
        with self.assertRaises(ValueError):
            pixutils.formats.fourcc_str.str_to_fourcc('ABC')

if __name__ == '__main__':
    unittest.main()
