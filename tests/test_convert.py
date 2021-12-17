#!/usr/bin/env python3

import unittest
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from daltonlens import convert

import colour

class TestConversions(unittest.TestCase):

    def test_lab(self):
        im_srgb_uint8 = np.array([[
            [0, 0, 0], 
            [255,255,255], 
            
            [255, 0, 0],
            [0, 255, 0], 
            [0, 0, 255],

            [0, 255, 255],
            [255, 255, 0],
            [255, 0, 255],
            
            [10, 20, 30], 
            [220, 110, 35], 
            [89, 240, 28], 
        ]])


        linearRGB = convert.linearRGB_from_sRGB (convert.as_float32(im_srgb_uint8))
        xyz = convert.apply_color_matrix (linearRGB, convert.XYZ_from_linearRGB_BT709)
        np.set_printoptions(precision=3, suppress=True)
        lab = convert.Lab_from_XYZ (xyz)

        ref_from_colour = colour.XYZ_to_Lab(xyz)
        max_diff = np.max (np.abs(ref_from_colour - lab))        
        if (max_diff > 0.05):
            print (lab)
            print (ref_from_colour)
            self.fail ("Lab conversion is too different.")

if __name__ == '__main__':
    unittest.main()
