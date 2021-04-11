import colour
import os
import sys
from PIL import Image
import numpy as np

image_linearRGB = colour.cctf_decoding(
    colour.read_image(sys.argv[1]),
    function='sRGB')

params = [
    ('Protanomaly', 1.0, "colour_protan_1.0.png"),
    ('Protanomaly', 0.55, "colour_protan_0.55.png"),

    ('Deuteranomaly', 1.0, "colour_deutan_1.0.png"),
    ('Deuteranomaly', 0.55, "colour_deutan_0.55.png"),

    ('Tritanomaly', 1.0, "colour_tritan_1.0.png"),
    ('Tritanomaly', 0.55, "colour_tritan_0.55.png"),
]

for anomaly, severity, filename in params:
    M_a = colour.blindness.matrix_cvd_Machado2009(deficiency=anomaly, severity=severity)
    cvd_image_linearRGB = colour.algebra.common.vector_dot(M_a, image_linearRGB)
    output_image = colour.cctf_encoding(cvd_image_linearRGB)
    output_image = (np.clip(output_image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(output_image).save(filename)

