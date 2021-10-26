import numpy as np
import sys

from daltonlens import convert, simulate

try: import Geometry3D as geo3d
except ImportError as e:
    sys.stderr.write("Geometry3D is required to import this module: `pip install Geometry3D'\n")
    raise e

def geo3dPoint_from_np(p): return geo3d.Point(p[0], p[1], p[2])
def geo3dVector_from_np(p): return geo3d.Vector(p[0], p[1], p[2])
def np_from_geo3dPoint(p): return np.array([p.x, p.y, p.z])

def lms_confusion_segment(lms_color, color_model: convert.LMSModel, deficiency: simulate.Deficiency):
    """Return the endpoints of the confusion line passing by the given LMS point.
    
    The line is parallel to the dichromat missing axis, and the segment is the intersection
    of that line with the RGB gamut in the LMS space.

    Note: the Geometry3D package is required for that function.

    Parameters
    ==========
    lms_color : 3D vector
    Coordinate of a color through which the confusion line should pass
    in the LMS space.

    color_model : convert.LMSModel
    The color model to go from RGB to LMS. It is used to compute the
    RGB gamut in LMS.

    deficiency : Deficiency
    One of Deficiency.PROTAN, DEUTAN or TRITAN to determine the confusion axis in LMS.

    Returns
    =======
    (start, end) : tuple of 3D points
    The points correspond to the start and end points of the segment.
    """

    # First we build the linear RGB gamut (cube) in LMS. It becomes
    # a parallelepiped since it's a linear transform.
    lms_K = color_model.LMS_from_linearRGB @ np.array([0.,0.,0.])
    lms_R = color_model.LMS_from_linearRGB @ np.array([1.,0.,0.])
    lms_G = color_model.LMS_from_linearRGB @ np.array([0.,1.,0.])
    lms_B = color_model.LMS_from_linearRGB @ np.array([0.,0.,1.])

    # The parallelogram is defined by the origin (black) and the
    # 3 vectors.
    KBMRGCWY_geometry = geo3d.Parallelepiped(geo3dPoint_from_np(lms_K),
                                             geo3dVector_from_np(lms_R-lms_K),
                                             geo3dVector_from_np(lms_G-lms_K),
                                             geo3dVector_from_np(lms_B-lms_K))

    confusion_axis = simulate.lms_confusion_axis(deficiency)

    # Now we build the confusion line passing through the given point.
    line = geo3d.Line(geo3dPoint_from_np(lms_color),
                      geo3dPoint_from_np(lms_color + confusion_axis))

    # The intersection with the parallelepiped is our segment.
    segment = geo3d.intersection(KBMRGCWY_geometry, line)
    # display(segment)

    # It can be None if the provided color was not inside the parallelogram.
    if segment is None:
        print("ERROR: lms_confusion_segment: the provided color is not in the RGB gamut.")
        return None
    return tuple(np_from_geo3dPoint(p) for p in segment)

