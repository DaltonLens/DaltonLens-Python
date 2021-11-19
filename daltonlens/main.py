#!/usr/bin/env python3

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

from daltonlens import convert, simulate, generate

def parse_command_line():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Toolbox to simulate and filter color vision deficiencies.',
                            formatter_class=ArgumentDefaultsHelpFormatter)    

    parser.add_argument("input_image", type=Path, help="Image to process.")
    parser.add_argument("output_image", type=Path, help="Output image")
 
    parser.add_argument("--model", "-m", type=str, default="auto",
                        help="Color model to apply: auto, vienot, brettel, machado, vischeck, coblisV1, coblisV2")

    parser.add_argument("--filter", "-f", type=str, default="simulate",
                        help="Filter to apply: simulate or daltonize.")

    parser.add_argument("--deficiency", "-d", type=str, default="protan",
                        help="Deficiency type: protan, deutan or tritan")

    parser.add_argument("--severity", "-s", type=float, default="1.0",
                        help="Severity between 0 and 1")

    args = parser.parse_args()
    return args

deficiency_from_str = {
    'protan': simulate.Deficiency.PROTAN,
    'deutan': simulate.Deficiency.DEUTAN,
    'tritan': simulate.Deficiency.TRITAN,
}

simulator_from_str = {
    'vienot': simulate.Simulator_Vienot1999(convert.LMSModel_sRGB_SmithPokorny75()),
    'brettel': simulate.Simulator_Brettel1997(convert.LMSModel_sRGB_SmithPokorny75()),
    'vischeck': simulate.Simulator_Vischeck(),
    'machado': simulate.Simulator_Machado2009(),
    'coblisV1': simulate.Simulator_CoblisV1(),
    'coblisV2': simulate.Simulator_CoblisV2(),
    'auto': simulate.Simulator_AutoSelect()
}

def main():
    args = parse_command_line ()

    deficiency = deficiency_from_str[args.deficiency]
    simulator: simulate.Simulator = simulator_from_str[args.model]
    im = np.asarray(Image.open(args.input_image).convert('RGB'))

    if args.filter == 'simulate':
        out = simulator.simulate_cvd(im, deficiency=deficiency, severity=args.severity)
    elif args.filter == 'daltonize':
        raise NotImplementedError()
    else:
        print(f"ERROR: invalid filter '{args.filter}'. Supported filters are 'simulate' and 'daltonize'")
        sys.exit (1)
    
    Image.fromarray(out).save(args.output_image)
    
if __name__ == '__main__':
    main()
