#!/usr/bin/python3

import unittest
import iob
import iob.fourcc_str

class TestDummy(unittest.TestCase):
    def test_dummy(self):
        self.assertEqual(iob.PixelFormats.RGB888.name, 'RGB888')

class TestFourcc(unittest.TestCase):
    def test_fourcc(self):
        self.assertEqual(iob.fourcc_str.str_to_fourcc('XR24'), 0x34325258)
        self.assertEqual(iob.fourcc_str.fourcc_to_str(0x34325258), 'XR24')
        with self.assertRaises(ValueError):
            iob.fourcc_str.str_to_fourcc('ABCDE')
        with self.assertRaises(ValueError):
            iob.fourcc_str.str_to_fourcc('ABC')

if __name__ == '__main__':
    unittest.main()
