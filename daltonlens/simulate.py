from daltonlens import convert
from collections import namedtuple

import math
import numpy as np
import sys

from abc import ABC, abstractmethod

from enum import Enum
class Deficiency(Enum):
    PROTAN = 0
    DEUTAN = 1
    TRITAN = 2

class Simulator (ABC):
    """Base class for all CVD simulators."""

    def simulate_cvd (self, image_srgb_uint8, deficiency: Deficiency, severity: float):
        """Simulate the appearance of an image for the given color vision defiency
    
        Parameters
        ==========
        image_srgb_uint8 : array of shape (M,N,3) with dtype uint8
            The input sRGB image, with values in [0,255].

        deficiency: Deficiency
            The deficiency to simulate.

        severity: float
            The severity between 0 (normal vision) and 1 (complete dichromacy).
    
        Returns
        =======
        im : array of shape (M,N,3) with dtype uint8
            The simulated sRGB image with values in [0,255].
        """
        im_linear_rgb = convert.linearRGB_from_sRGB(convert.as_float32(image_srgb_uint8))
        im_cvd_linear_rgb = self._simulate_cvd_linear_rgb(im_linear_rgb, deficiency, severity)
        return convert.as_uint8(convert.sRGB_from_linearRGB(im_cvd_linear_rgb))

    @abstractmethod
    def _simulate_cvd_linear_rgb (self, image_linear_rgb_float32, deficiency: Deficiency, severity: float):
        """All subclasses must implement this."""
        pass

class DichromacySimulator (Simulator):
    """Base class for CVD simulators that only support dichromacy
    
        Anomalous trichromacy will be implemented on top of the
        dichromacy simulator by linearly interpolating between the
        original image and the dichromat version. 
        
        This is not backed back a strong theory, but it works well in
        practice and is similar in spirit to the 
        'So that's what you see": building understanding with personalized
        simulations of colour vision deficiency'
        paper by D. Flatla and C. Gutwin, with the difference that they
        use a fixed step.
    """

    def _simulate_cvd_linear_rgb (self, image_linear_rgb_float32, deficiency: Deficiency, severity: float):
        im_dichromacy = self._simulate_dichromacy_linear_rgb(image_linear_rgb_float32, deficiency)
        if severity < 0.99999:
            return im_dichromacy*severity + image_linear_rgb_float32*(1.0-severity)
        else:
            return im_dichromacy

    @abstractmethod
    def _simulate_dichromacy_linear_rgb (self, image_linear_rgb_float32, deficiency: Deficiency, severity: float):
        pass

def plane_projection_matrix(plane_normal, deficiency: Deficiency):
    """Utility function for Vienot and Brettel.
    
    Given the projection plane normal, it returns the projection
    matrix along the deficiency axis that will project an LMS
    color to the plane. We don't need to take an origin since
    black (0,0,0) is always on the plane.
    """
    n = plane_normal
    # Projection along the L axis
    if deficiency == Deficiency.PROTAN:
        return np.array([
            [0., -n[1]/n[0], -n[2]/n[0]],
            [0, 1, 0],
            [0, 0, 1]
        ])
    # Projection along the M axis
    if deficiency == Deficiency.DEUTAN:
        return np.array([
            [1, 0, 0],
            [-n[0]/n[1], 0, -n[2]/n[1]],
            [0, 0, 1]
        ])
    # Projection along the S axis
    if deficiency == Deficiency.TRITAN:
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-n[0]/n[2], -n[1]/n[2], 0]
        ])
    return None

def lms_confusion_axis(deficiency: Deficiency):
    """Return the LMS axis along which a dichromat will confuse the colors.
    """
    
    if deficiency == Deficiency.PROTAN: return np.array([1.0, 0.0, 0.0])
    if deficiency == Deficiency.DEUTAN: return np.array([0.0, 1.0, 0.0])
    if deficiency == Deficiency.TRITAN: return np.array([0.0, 0.0, 1.0])
    return None

class Simulator_Vienot1999 (DichromacySimulator):
    """Algorithm of (Viénot & Brettel & Mollon, 1999).

    'Digital video colourmaps for checking the legibility of displays by dichromats.'

    Recommended for protanopia and deuteranopia, but not accurate for tritanopia.    
    """

    def __init__(self, color_model: convert.LMSModel = convert.LMSModel_sRGB_SmithPokorny75()):
        self.color_model = color_model

    def _simulate_dichromacy_linear_rgb (self, image_linear_rgb_float32, deficiency: Deficiency):
        lms_projection_matrix = None
        if deficiency == Deficiency.PROTAN or deficiency == Deficiency.DEUTAN:
            lms_blue = self.color_model.LMS_from_linearRGB @ np.array([0.0, 0.0, 1.0])
            lms_yellow = self.color_model.LMS_from_linearRGB @ np.array([1.0, 1.0, 0.0])
            v_blue = lms_blue # - lms_black which is ommitted since it's zero
            v_yellow = lms_yellow # - lms_black which is ommitted since it's zero

            # Deutan and Protan plane normal
            n = np.cross(v_yellow, v_blue)
            lms_projection_matrix = plane_projection_matrix(n, deficiency)
        else:
            print ("WARNING: Viénot 1999 is not accurate for tritanopia. Use Brettel 1997 instead.")
            v_red = self.color_model.LMS_from_linearRGB @ np.array([1.0, 0.0, 0.0]) # - lms_black which is ommitted since it's zero
            v_cyan = self.color_model.LMS_from_linearRGB @ np.array([0.0, 1.0, 1.0]) # - lms_black which is ommitted since it's zero
            n = np.cross(v_cyan, v_red)
            lms_projection_matrix = plane_projection_matrix(n, Deficiency.TRITAN)

        cvd_linear_rgb = self.color_model.linearRGB_from_LMS @ lms_projection_matrix @ self.color_model.LMS_from_linearRGB
        return convert.apply_color_matrix(image_linear_rgb_float32, cvd_linear_rgb)

class Simulator_Brettel1997 (DichromacySimulator):
    """Algorithm of (Brettel & Mollon, 1997).

    'Computerized simulation of color appearance for dichromats'

    This model is a bit more complex than (Viénot & Brettel & Mollon, 1999)
    but it works well for tritanopia. It is also the most solid reference
    in the litterature.
    """

    def __init__(self, color_model: convert.LMSModel = convert.LMSModel_sRGB_SmithPokorny75()):
        self.color_model = color_model

    def _simulate_dichromacy_linear_rgb (self, image_linear_rgb_float32, deficiency: Deficiency):
        # This is how these were computed. Saving the values to avoid a dependency.
        # import colour # from pip install colour-science
        # from colour import MSDS_CMFS
        # cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        # xyz_475 = colour.wavelength_to_XYZ(475, cmfs)
        # xyz_575 = colour.wavelength_to_XYZ(575, cmfs)
        # xyz_485 = colour.wavelength_to_XYZ(485, cmfs)
        # xyz_660 = colour.wavelength_to_XYZ(660, cmfs)

        xyz_475 = np.array([ 0.1421,  0.1126,  1.0419])
        xyz_575 = np.array([ 0.8425,  0.9154,  0.0018])
        xyz_485 = np.array([ 0.05795, 0.1693,  0.6162])
        xyz_660 = np.array([ 0.1649,  0.0610,  0.0000])

        # The equal-energy white point. By construction of CIE XYZ
        # it has X=Y=Z. The normalization does not matter to define
        # the diagonal direction, picking 0.8 to make it close to
        # sRGB white.
        xyz_E = [0.8, 0.8, 0.8]
        rgb_E = self.color_model.linearRGB_from_XYZ @ xyz_E
        lms_E = self.color_model.LMS_from_XYZ @ xyz_E
        # lms_W = self.color_model.LMS_from_linearRGB @ np.array([1.0,1.0,1.0])

        def compute_matrices(lms_E, lms_on_wing1, lms_on_wing2):
            n1 = np.cross(lms_E, lms_on_wing1) # first plane
            n2 = np.cross(lms_E, lms_on_wing2) # second plane
            n_sep_plane = np.cross(lms_E, lms_confusion_axis(deficiency)) # separation plane going through the diagonal
            # Swap the input so that wing1 is on the positive side of the separation plane
            if np.dot(n_sep_plane, lms_on_wing1) < 0:
                n1, n2 = n2, n1
                lms_on_wing1, lms_on_wing2 = lms_on_wing2, lms_on_wing1
            H1 = plane_projection_matrix(n1, deficiency)
            H2 = plane_projection_matrix(n2, deficiency)
            return (H1, H2, n_sep_plane)

        H1 = H2 = n_sep_plane = None
        if deficiency == Deficiency.PROTAN or deficiency == Deficiency.DEUTAN:
            lms_475 = self.color_model.LMS_from_XYZ @ xyz_475
            lms_575 = self.color_model.LMS_from_XYZ @ xyz_575
            H1, H2, n_sep_plane = compute_matrices(lms_E, lms_475, lms_575)
        else:
            lms_485 = self.color_model.LMS_from_XYZ @ xyz_485
            lms_660 = self.color_model.LMS_from_XYZ @ xyz_660
            H1, H2, n_sep_plane = compute_matrices(lms_E, lms_485, lms_660)

        im_lms = convert.apply_color_matrix(image_linear_rgb_float32, self.color_model.LMS_from_linearRGB)
        im_H1 = convert.apply_color_matrix(im_lms, H1)
        im_H2 = convert.apply_color_matrix(im_lms, H2)
        H2_indices = np.dot(im_lms, n_sep_plane) < 0

        # Start with H1, then overwrite the pixels that are closer to plane 2 with im_H2
        im_H = im_H1
        im_H[H2_indices] = im_H2[H2_indices]
        im_linear_rgb = convert.apply_color_matrix(im_H, self.color_model.linearRGB_from_LMS)
        return im_linear_rgb

"""
From https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html#Reference
Converted to numpy array by https://github.com/colour-science/colour/blob/develop/colour/blindness/datasets/machado2010.py
The severity key goes from 0.0 to 1.0 with 0.1 steps, but here the index is multiplied by 10 to make it an integer.
"""
machado_2009_matrices = {
    Deficiency.PROTAN: {
        0: np.array([ [1.000000, 0.000000, -0.000000], [0.000000, 1.000000, 0.000000], [-0.000000, -0.000000, 1.000000] ]),
        1: np.array([ [0.856167, 0.182038, -0.038205], [0.029342, 0.955115, 0.015544], [-0.002880, -0.001563, 1.004443] ]),
        2: np.array([ [0.734766, 0.334872, -0.069637], [0.051840, 0.919198, 0.028963], [-0.004928, -0.004209, 1.009137] ]),
        3: np.array([ [0.630323, 0.465641, -0.095964], [0.069181, 0.890046, 0.040773], [-0.006308, -0.007724, 1.014032] ]),
        4: np.array([ [0.539009, 0.579343, -0.118352], [0.082546, 0.866121, 0.051332], [-0.007136, -0.011959, 1.019095] ]),
        5: np.array([ [0.458064, 0.679578, -0.137642], [0.092785, 0.846313, 0.060902], [-0.007494, -0.016807, 1.024301] ]),
        6: np.array([ [0.385450, 0.769005, -0.154455], [0.100526, 0.829802, 0.069673], [-0.007442, -0.022190, 1.029632] ]),
        7: np.array([ [0.319627, 0.849633, -0.169261], [0.106241, 0.815969, 0.077790], [-0.007025, -0.028051, 1.035076] ]),
        8: np.array([ [0.259411, 0.923008, -0.182420], [0.110296, 0.804340, 0.085364], [-0.006276, -0.034346, 1.040622] ]),
        9: np.array([ [0.203876, 0.990338, -0.194214], [0.112975, 0.794542, 0.092483], [-0.005222, -0.041043, 1.046265] ]),
        10: np.array([ [0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998] ])
    },

    Deficiency.DEUTAN: {
        0: np.array([ [1.000000, 0.000000, -0.000000], [0.000000, 1.000000, 0.000000], [-0.000000, -0.000000, 1.000000] ]),
        1: np.array([ [0.866435, 0.177704, -0.044139], [0.049567, 0.939063, 0.011370], [-0.003453, 0.007233, 0.996220] ]),
        2: np.array([ [0.760729, 0.319078, -0.079807], [0.090568, 0.889315, 0.020117], [-0.006027, 0.013325, 0.992702] ]),
        3: np.array([ [0.675425, 0.433850, -0.109275], [0.125303, 0.847755, 0.026942], [-0.007950, 0.018572, 0.989378] ]),
        4: np.array([ [0.605511, 0.528560, -0.134071], [0.155318, 0.812366, 0.032316], [-0.009376, 0.023176, 0.986200] ]),
        5: np.array([ [0.547494, 0.607765, -0.155259], [0.181692, 0.781742, 0.036566], [-0.010410, 0.027275, 0.983136] ]),
        6: np.array([ [0.498864, 0.674741, -0.173604], [0.205199, 0.754872, 0.039929], [-0.011131, 0.030969, 0.980162] ]),
        7: np.array([ [0.457771, 0.731899, -0.189670], [0.226409, 0.731012, 0.042579], [-0.011595, 0.034333, 0.977261] ]),
        8: np.array([ [0.422823, 0.781057, -0.203881], [0.245752, 0.709602, 0.044646], [-0.011843, 0.037423, 0.974421] ]),
        9: np.array([ [0.392952, 0.823610, -0.216562], [0.263559, 0.690210, 0.046232], [-0.011910, 0.040281, 0.971630] ]),
        10: np.array([ [0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881] ])
    },

    Deficiency.TRITAN: {
        0: np.array([ [1.000000, 0.000000, -0.000000],  [ 0.000000, 1.000000, 0.000000], [-0.000000, -0.000000, 1.000000] ]),
        1: np.array([ [0.926670, 0.092514, -0.019184],  [ 0.021191, 0.964503, 0.014306], [0.008437, 0.054813, 0.936750] ]),
        2: np.array([ [0.895720, 0.133330, -0.029050],  [ 0.029997, 0.945400, 0.024603], [0.013027, 0.104707, 0.882266] ]),
        3: np.array([ [0.905871, 0.127791, -0.033662],  [ 0.026856, 0.941251, 0.031893], [0.013410, 0.148296, 0.838294] ]),
        4: np.array([ [0.948035, 0.089490, -0.037526],  [ 0.014364, 0.946792, 0.038844], [0.010853, 0.193991, 0.795156] ]),
        5: np.array([ [1.017277, 0.027029, -0.044306],  [-0.006113, 0.958479, 0.047634], [0.006379, 0.248708, 0.744913] ]),
        6: np.array([ [1.104996, -0.046633, -0.058363], [-0.032137, 0.971635, 0.060503], [0.001336, 0.317922, 0.680742] ]),
        7: np.array([ [1.193214, -0.109812, -0.083402], [-0.058496, 0.979410, 0.079086], [-0.002346, 0.403492, 0.598854] ]),
        8: np.array([ [1.257728, -0.139648, -0.118081], [-0.078003, 0.975409, 0.102594], [-0.003316, 0.501214, 0.502102] ]),
        9: np.array([ [1.278864, -0.125333, -0.153531], [-0.084748, 0.957674, 0.127074], [-0.000989, 0.601151, 0.399838] ]),
        10: np.array([ [1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900] ])
    }
}

class Simulator_Machado2009 (Simulator):
    """The model proposed by (MacHado & Oliveira & Fernandes, 2009)

    'A physiologically-based model for simulation of color vision deficiency'

    This model is similar to Brettel1997 for dichromacy (it actually uses it
    as a reference to scale the parameters), but is able to simulate various
    severity levels by shifting the peak wavelength for a given cone, which
    is more a more sounded way than simply interpolating with the original
    image. However that model does not work well for tritanopia.
    """

    def _simulate_cvd_linear_rgb (self, image_linear_rgb_float32, deficiency: Deficiency, severity: float):
        assert severity >= 0.0 and severity <= 1.0
        severity_lower = int(math.floor(severity*10.0))
        severity_higher = min(severity_lower + 1, 10)
        m1 = machado_2009_matrices[deficiency][severity_lower]
        m2 = machado_2009_matrices[deficiency][severity_higher]

        # alpha = 0 => only m1, alpha = 1.0 => only m2
        alpha = (severity - severity_lower/10.0)
        m = alpha*m2 + (1.0-alpha)*m1

        return convert.apply_color_matrix(image_linear_rgb_float32, m)

class Simulator_AutoSelect (Simulator):
    """Automatically selects the best algorithm for the given deficiency and severity.
    
    - For tritan simulations it always picks (Brettel & Molon, 1997)
    - For protanomaly/deuteranomly (severity < 1) it picks (Machado, 2009)
    - For protanopia/deuteranopia (severity = 1) it picks (Vienot, 1999)
    """
    def _simulate_cvd_linear_rgb (self, image_linear_rgb_float32, deficiency: Deficiency, severity: float):
        if deficiency == Deficiency.TRITAN:
            print ("Choosing Brettel 1997 for tritanopia / tritanomaly")
            simulator = Simulator_Brettel1997(convert.LMSModel_sRGB_SmithPokorny75())
        elif severity < 0.999:
            print("Anomalous trichromacy requested, using Machado 2009")
            simulator = Simulator_Machado2009()
        else:
            print("Choosing Viénot 1999 for " + ("protanopia" if deficiency == Deficiency.PROTAN else "deuteranopia"))
            simulator = Simulator_Vienot1999(convert.LMSModel_sRGB_SmithPokorny75())
        
        return simulator._simulate_cvd_linear_rgb (image_linear_rgb_float32, deficiency, severity)
