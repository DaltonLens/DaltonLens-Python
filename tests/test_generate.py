#!/usr/bin/env python3

import unittest
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from daltonlens import convert, simulate, generate

class TestIshiharaPlate(unittest.TestCase):

    def test_protanopia(self):
        im = generate.ishihara_plate(simulate.Deficiency.PROTAN, 1.0, "Protan Severity 1.0")
        Image.fromarray(im).save("plate_protan_1.0.png")

    def test_protanomaly(self):
        im = generate.ishihara_plate(simulate.Deficiency.PROTAN, 0.5)
        Image.fromarray(im).save("plate_protan_0.5.png")

    def test_deuteranopia(self):
        im = generate.ishihara_plate(simulate.Deficiency.DEUTAN, 1.0)
        Image.fromarray(im).save("plate_deutan_1.0.png")

    def test_tritanopia(self):
        im = generate.ishihara_plate(simulate.Deficiency.TRITAN, 1.0)
        Image.fromarray(im).save("plate_tritan_1.0.png")

if __name__ == '__main__':
    unittest.main()
