# DaltonLens-Python

[![Unit Tests](https://github.com/DaltonLens/DaltonLens-Python/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/DaltonLens/DaltonLens-Python/actions/workflows/unit_tests.yml)

This python package is a companion to the desktop application [DaltonLens](https://github.com/DaltonLens/DaltonLens). Its main goal is to help the research and development of better color filters for people with color vision deficiencies. It also powers the Jupyter notebooks used for the technical posts of [daltonlens.org](https://daltonlens.org). The current features include:

* Simulate color vision deficiencies using the Viénot 1999, Brettel 1997 or Machado 2009 models.
* Provide conversion functions to/from sRGB, linear RGB and LMS
* Implement several variants of the LMS model
* Generate Ishihara-like test images

For a discussion about which CVD simulation algorithms are the most accurate see our [Review of Open Source Color Blindness Simulations](https://daltonlens.org/opensource-cvd-simulation/).

For more information about the math of the chosen algorithms see our article [Understanding CVD Simulation](https://daltonlens.org/understanding-cvd-simulation/).

## Install

`python3 -m pip install daltonlens`

## How to use

### From the command line

```
daltonlens-python --help
usage: daltonlens-python [-h] 
       [--model MODEL] [--filter FILTER]
       [--deficiency DEFICIENCY] [--severity SEVERITY]
       input_image output_image

Toolbox to simulate and filter color vision deficiencies.

positional arguments:
  input_image           Image to process.
  output_image          Output image

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Color model to apply: vienot, brettel, machado or auto (default: auto)
  --filter FILTER, -f FILTER
                        Filter to apply: simulate or daltonize. (default: simulate)
  --deficiency DEFICIENCY, -d DEFICIENCY
                        Deficiency type: protan, deutan or tritan (default: protan)
  --severity SEVERITY, -s SEVERITY
                        Severity between 0 and 1 (default: 1.0)
```

### From code

```python
from daltonlens import convert, simulate, generate
import PIL
import numpy as np

# Generate a test image that spans the RGB range
im = np.asarray(PIL.Image.open("test.png").convert('RGB'))

# Create a simulator using the Viénot 1999 algorithm.
simulator = simulate.Simulator_Vienot1999()

# Apply the simulator to the input image to get a simulation of protanomaly
protan_im = simulator.simulate_cvd (im, simulate.Deficiency.PROTAN, severity=0.8)
```
