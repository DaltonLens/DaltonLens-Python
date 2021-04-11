#!/usr/bin/env python3

import unittest
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from daltonlens import convert, simulate, generate

test_images_path = Path(__file__).parent.absolute() / "images"

class TestCVD(unittest.TestCase):

    def checkModels(self, im, models_to_test):        
        for simulator, deficiency, severity, gt_im in models_to_test:
            out = simulator.simulate_cvd(im, deficiency=deficiency, severity=severity)
            # Uncomment to generate a new ground truth.
            # Image.fromarray(out).save(test_images_path / gt_im)
            gt = np.asarray(Image.open(test_images_path / gt_im).convert('RGB'))
            self.assertTrue(np.allclose(out, gt))

    def test_vienot1999(self):
        vienot1999 = simulate.Simulator_Vienot1999(convert.LMSModel_sRGB_SmithPokorny75())

        models_to_test = {
            (vienot1999, simulate.Deficiency.PROTAN, 1.0, "vienot1999_protan_1.0.png"),
            (vienot1999, simulate.Deficiency.DEUTAN, 1.0, "vienot1999_deutan_1.0.png"),
            (vienot1999, simulate.Deficiency.TRITAN, 1.0, "vienot1999_tritan_1.0.png"),

            (vienot1999, simulate.Deficiency.PROTAN, 0.55, "vienot1999_protan_0.55.png"),
            (vienot1999, simulate.Deficiency.DEUTAN, 0.55, "vienot1999_deutan_0.55.png"),
            (vienot1999, simulate.Deficiency.TRITAN, 0.55, "vienot1999_tritan_0.55.png"),
        }

        im = generate.rgb_span(27*8, 27*8)
        self.checkModels (im, models_to_test)
    
    def test_brettel1997(self):
        brettel1997 = simulate.Simulator_Brettel1997(convert.LMSModel_sRGB_SmithPokorny75())

        models_to_test = {
            (brettel1997, simulate.Deficiency.PROTAN, 1.0, "brettel1997_protan_1.0.png"),
            (brettel1997, simulate.Deficiency.DEUTAN, 1.0, "brettel1997_deutan_1.0.png"),
            (brettel1997, simulate.Deficiency.TRITAN, 1.0, "brettel1997_tritan_1.0.png"),

            (brettel1997, simulate.Deficiency.PROTAN, 0.55, "brettel1997_protan_0.55.png"),
            (brettel1997, simulate.Deficiency.DEUTAN, 0.55, "brettel1997_deutan_0.55.png"),
            (brettel1997, simulate.Deficiency.TRITAN, 0.55, "brettel1997_tritan_0.55.png"),
        }

        im = generate.rgb_span(27*8, 27*8)
        self.checkModels (im, models_to_test)

    def test_machado2009(self):
        machado2009 = simulate.Simulator_Machado2009()

        models_to_test = {
            (machado2009, simulate.Deficiency.PROTAN, 1.0, "machado2009_protan_1.0.png"),
            (machado2009, simulate.Deficiency.DEUTAN, 1.0, "machado2009_deutan_1.0.png"),
            (machado2009, simulate.Deficiency.TRITAN, 1.0, "machado2009_tritan_1.0.png"),

            (machado2009, simulate.Deficiency.PROTAN, 0.55, "machado2009_protan_0.55.png"),
            (machado2009, simulate.Deficiency.DEUTAN, 0.55, "machado2009_deutan_0.55.png"),
            (machado2009, simulate.Deficiency.TRITAN, 0.55, "machado2009_tritan_0.55.png"),
        }

        im = generate.rgb_span(27*8, 27*8)
        self.checkModels (im, models_to_test)

    def test_auto(self):
        im = generate.rgb_span(27, 27)
        machado2009 = simulate.Simulator_Machado2009()
        brettel1997 = simulate.Simulator_Brettel1997(convert.LMSModel_sRGB_SmithPokorny75())
        vienot1999 = simulate.Simulator_Vienot1999(convert.LMSModel_sRGB_SmithPokorny75())
        auto = simulate.Simulator_AutoSelect()
        
        out_auto = auto.simulate_cvd(im, simulate.Deficiency.PROTAN, severity=0.3)
        out_ref = machado2009.simulate_cvd(im, simulate.Deficiency.PROTAN, severity=0.3)
        self.assertTrue(np.allclose(out_auto, out_ref))

        out_auto = auto.simulate_cvd(im, simulate.Deficiency.TRITAN, severity=0.3)
        out_ref = brettel1997.simulate_cvd(im, simulate.Deficiency.TRITAN, severity=0.3)
        self.assertTrue(np.allclose(out_auto, out_ref))

        out_auto = auto.simulate_cvd(im, simulate.Deficiency.DEUTAN, severity=1.0)
        out_ref = vienot1999.simulate_cvd(im, simulate.Deficiency.DEUTAN, severity=1.0)
        self.assertTrue(np.allclose(out_auto, out_ref))

if __name__ == '__main__':
    unittest.main()
