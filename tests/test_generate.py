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
        im = generate.ishihara_plate_dichromacy(simulate.Deficiency.PROTAN, "Protan Severity 1.0")
        Image.fromarray(im).save("plate_protan_1.0.png")

    def test_protanomaly(self):
        im = generate.ishihara_plate_dichromacy(simulate.Deficiency.PROTAN)
        Image.fromarray(im).save("plate_protan_0.5.png")

    def test_deuteranopia(self):
        im = generate.ishihara_plate_dichromacy(simulate.Deficiency.DEUTAN)
        Image.fromarray(im).save("plate_deutan_1.0.png")

    def test_tritanopia(self):
        im = generate.ishihara_plate_dichromacy(simulate.Deficiency.TRITAN)
        Image.fromarray(im).save("plate_tritan_1.0.png")

if __name__ == '__main__':
    unittest.main()
