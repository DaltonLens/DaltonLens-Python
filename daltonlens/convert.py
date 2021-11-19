import math
import numpy as np

from enum import Enum

class ImageEncoding(Enum):
    SRGB = 0
    LINEAR_RGB = 1 # assume the image is already in linearRGB and don't apply any transform
    GAMMA_22 = 2 # gamma of 2.2 (old CRTs, before the sRGB standard)

def as_uint8(im):
    """Multiply by 255 and cast the float image to uint8"""
    return (np.clip(im,0.,1.0)*255.0).astype(np.uint8)

def as_float32(im):
    """Divide by 255 and cast the uint8 image to float32"""
    return im.astype(np.float32)/255.0

def linearRGB_from_gamma22(im):
    """Gamma correction for old PCs/CRT monitors"""
    return np.power(im, 2.2)

def gamma22_from_linearRGB(im):
    """Inverse gamma correction for old PCs/CRT monitors"""
    return np.power(np.clip(im, 0., 1.), 1.0 / 2.2)

def linearRGB_from_sRGB(im):
    """Convert sRGB to linearRGB, removing the gamma correction.

    Formula taken from Wikipedia https://en.wikipedia.org/wiki/SRGB

    Parameters
    ==========
    im : array of shape (M,N,3) with dtype float
        The input sRGB image, normalized between [0,1]

    Returns
    =======
    im : array of shape (M,N,3) with dtype float
        The output RGB image
    """
    out = np.zeros_like(im)
    small_mask = im < 0.04045
    large_mask = np.logical_not(small_mask)
    out[small_mask] = im[small_mask] / 12.92
    out[large_mask] = np.power((im[large_mask] + 0.055) / 1.055, 2.4)
    return out

def desaturate_linearRGB_to_fit_in_gamut(im):
    """Make the RGB colors fit in the [0,1] range by desaturating.

    Inspired from https://www.fourmilab.ch/documents/specrend/
    Instead of just clipping to 0,1, we move the color towards
    the white point, desaturating it until it fits the RGB gamut.

    Parameters
    ==========
    im : array of shape (M,N,3) with dtype float
        The input linear RGB image

    Returns
    =======
    im : array of shape (M,N,3) with dtype float
        The output linear RGB image with values in [0, 1]
    """
    # Find the most negative value for each, or 0
    min_val = np.fmin(im[:,0], im[:,1])
    min_val = np.fmin(min_val, im[:,2])
    min_val = np.fmin(min_val, 0.0)
    # Add the same white component to all 3 values until none
    # is negative and clip the max of each component to 1.0
    return np.clip(im - min_val[:,np.newaxis], 0., 1.0)

def sRGB_from_linearRGB(im):
    """Convert linearRGB to sRGB, applying the gamma correction.

    Formula taken from Wikipedia https://en.wikipedia.org/wiki/SRGB

    Parameters
    ==========
    im : array of shape (M,N,3) with dtype float
        The input RGB image, normalized between [0,1].
        It will be clipped to [0,1] to avoid numerical issues with gamma.

    Returns
    =======
    im : array of shape (M,N,3) with dtype float
        The output sRGB image
    """
    out = np.zeros_like(im)
    # Make sure we're in range, otherwise gamma will go crazy.
    im = np.clip(im, 0., 1.)
    small_mask = im < 0.0031308
    large_mask = np.logical_not(small_mask)
    out[small_mask] = im[small_mask] * 12.92
    out[large_mask] = np.power(im[large_mask], 1.0 / 2.4) * 1.055 - 0.055
    return out

def apply_color_matrix(im, m):
    """Transform a color array with the given 3x3 matrix.

    Parameters
    ==========
    im : array of shape (...,3)
        Can be an image or a 1D array, as long as the last
        dimension is a 3-channels color.

    m : array of shape (3,3)
        Color matrix to apply.

    Returns
    =======
    im : array of shape (...,3)
        Output array, where each input color vector was multiplied by m.
    """
    # Another option is np.einsum('ij, ...j', m, im), but it can be much
    # slower, especially on on float32 types because the matrix multiplication
    # is heavily optimized.
    # So the matmul is generally (much) faster, but we need to take the
    # transpose of m as it gets applied on the right side. Indeed for each
    # column color vector v we wanted $v' = m . v$ . To flip the side we can
    # use $m . v = (v^T . m^T)^T$ . The transposes on the 1d vector are implicit
    # and can be ignored, so we just need to compute $v . m^T$. This is what
    # numpy matmul will do for all the vectors thanks to its broadcasting rules
    # that pick the last 2 dimensions of each array, so it will actually compute
    # matrix multiplications of shape (M,3) x (3,3) with M the penultimate dimension
    # of m. That will write a matrix of shape (M,3) with each row storing the
    # result of $v' = v . M^T$.
    return im @ m.T

class LMSModel:
    """Base class of all the LMS models.

    A convenient set of conversion matrices are built from the two required
    XYZ_from_linearRGB and LMS_from_XYZ input matrices.
    """
    def __init__(self, XYZ_from_linearRGB, LMS_from_XYZ):
        self.XYZ_from_linearRGB = XYZ_from_linearRGB
        self.LMS_from_XYZ = LMS_from_XYZ
        self.LMS_from_linearRGB = LMS_from_XYZ @ XYZ_from_linearRGB
        self.linearRGB_from_LMS = np.linalg.inv(self.LMS_from_linearRGB)
        self.linearRGB_from_XYZ = np.linalg.inv(self.XYZ_from_linearRGB)
        self.XYZ_from_LMS = np.linalg.inv(self.LMS_from_XYZ)

    def sRGB_from_LMS(self, lms):
        rgb = np.array([self.linearRGB_from_LMS @ lms])
        rgb = desaturate_linearRGB_to_fit_in_gamut(rgb)
        srgb = sRGB_from_linearRGB(rgb)*255.0
        return srgb.flatten().astype(np.uint8)

    def sRGB_hexstring_from_LMS(self, lms):
        r,g,b = self.sRGB_from_LMS(lms)
        return f'#{r:02x}{g:02x}{b:02x}'

LMS_from_XYZ_Smith_Pokorny_1975 = np.array([
        [ 0.15514, 0.54312, -0.03286],
        [-0.15514, 0.45684,  0.03286],
        [ 0      , 0      ,  0.01608]])

# Matrix for the sRGB standard
XYZ_from_linearRGB_BT709 = np.array([
        [0.412456, 0.3575761, 0.1804375],
        [0.212672, 0.7151522, 0.0721750],
        [0.019333, 0.1191920, 0.9503041]])

# We can safely normalize the rows of the transform without changing the output
# of any simulation as the angle between the axes are still similar and the
# inverse scaling will be applied when going back to XYZ/RGB.
# This can be useful to compare the matrices.
def normalize_LMS_from_XYZ(lms_from_XYZ):
    output = np.copy(lms_from_XYZ)
    output[0,:] = output[0,:] / np.linalg.norm(output[0,:])
    output[1,:] = output[1,:] / np.linalg.norm(output[1,:])
    output[2,:] = output[2,:] / np.linalg.norm(output[2,:])
    return output
class LMSModel_Vischeck_GIMP (LMSModel):
    """LMS model of Vischeck, as implemented in GIMP display filters.

     These LMS_from_linearRGB / linearRGB_from_LMS were taken from
     https://github.com/GNOME/gimp/blob/master/modules/display-filter-color-blind.c

     The comments say that these matrices were computed for CRT monitors with a
     spectral photometric and the Stockman human cone fundamentals and that they
     should not be used for LCD monitors. So these are probably not the best
     anymore given that CRT monitors are gone and modern monitors adopted the
     sRGB standard. But having this model is useful to use the Vischeck model as
     a reference.
    """

    vischeck_LMS_from_linearRGB = np.array([[0.05059983, 0.08585369, 0.00952420],
                                            [0.01893033, 0.08925308, 0.01370054],
                                            [0.00292202, 0.00975732, 0.07145979]])

    vischeck_linearRGB_from_LMS = np.array([[30.830854, -29.832659, 1.610474],
                                            [-6.481468, 17.715578, -2.532642],
                                            [-0.375690, -1.199062, 14.273846]])

    # For sRGB. It does not really matter though, since we directly provide the
    # final LMS_from_linearRGB. It's just used to compute the same set of
    # intermediate matrices as the other models.
    XYZ_from_linearRGB = XYZ_from_linearRGB_BT709

    LMS_from_XYZ = vischeck_LMS_from_linearRGB @ np.linalg.inv(XYZ_from_linearRGB)

    def __init__(self):
        super().__init__(self.XYZ_from_linearRGB, self.LMS_from_XYZ)

class LMSModel_sRGB_SmithPokorny75 (LMSModel):
    """LMS model of (Smith & Pokorny, 1975), adapted to SRGB monitors.

    The XYZ to LMS transform is the model used by (Viénot & Brettel & Mollon, 1999),
    but the RGB to XYZ one is updated to use the modern sRGB specification, from
    https://en.wikipedia.org/wiki/SRGB . The original model used by Viénot 1999 was
    for CRT monitors.

    This model is recommended for CVD simulation.
    """
    XYZ_from_linearRGB = XYZ_from_linearRGB_BT709

    # Vienot took this from [Smith, 1975], clarified in [Smith, 2003]
    LMS_from_XYZ = LMS_from_XYZ_Smith_Pokorny_1975

    # LMS_from_XYZ @ XYZ_from_linearRGB
    # LMS_from_linearRGB = np.array([
    #     [0.17885956, 0.43997117, 0.03596577],
    #     [0.03380394, 0.27515242, 0.03620635],
    #     [0.00031087, 0.00191661, 0.01528089]])

    # # Inverse of LMS_from_linearRGB
    # linearRGB_from_LMS = np.array([
    #     [  8.0053295 , -12.88195628,  11.68065329],
    #     [ -0.97821211,   5.26945022, -10.18300682],
    #     [ -0.04016557,  -0.39885551,  66.4807932 ]])

    def __init__(self):
        super().__init__(self.XYZ_from_linearRGB, self.LMS_from_XYZ)

class LMSModel_Vienot1999_SmithPokorny75 (LMSModel):
    """LMS model of (Viénot & Brettel & Mollon, 1999), unchanged.
    
    'Digital video colourmaps for checking the legibility of displays by dichromats.'

    WARNING: this model is not recommended as the sRGB to XYZ transform is not correct
    for modern monitors. It is mostly here for comparison purposes.
    """

    # Below XYZ assumes the BT709 standard.

    # Matrix from the paper divided by 255.
    # They take this matrix from old generic CRT monitors.
    XYZ_from_linearRGB = np.array([
        [0.1606149 , 0.13923176, 0.07026157],
        [0.08368196, 0.27715412, 0.03132078],
        [0.00730576, 0.04494902, 0.35779098]])

    # # The 255 scaled version is shown for easy comparison with
    # # other source code. Not meant to be really used here.
    # XYZ_from_linearRGB_255 = np.array([
    #     [40.9568, 35.5041, 17.9167],
    #     [21.3389, 70.6743, 7.98680],
    #     [1.86297, 11.4620, 91.2367]])

    # Vienot took this from [Smith, 1995], clarified in [Smith, 2003]
    LMS_from_XYZ = LMS_from_XYZ_Smith_Pokorny_1975

    # = LMS_from_XYZ @ XYZ_from_linearRGB
    # LMS_from_linearRGB = np.array([
    #     [0.07012706, 0.17065137, 0.01615431],
    #     [0.01355157, 0.10649176, 0.01516525],
    #     [0.00011748, 0.00072278, 0.00575329]])

    # = LMS_from_linearRGB * 255
    # This is the very widely used final transform, used that
    # [Fidaner, 2005] and many other places.
    # LMS_from_linearRGB_255 = np.array([
    #     [17.8824,  43.5161,   4.11935],
    #     [3.45565,  27.1554,   3.86714],
    #     [0.0299566, 0.184309, 1.46709]])

    def __init__(self):
        super().__init__(self.XYZ_from_linearRGB, self.LMS_from_XYZ)

class LMSModel_sRGB_HuntPointerEstevez (LMSModel):
    """Model using sRGB to go to XYZ and the Hunt-Pointer-Estevez transform to LMS

    WARNING: this model is not recommended for CVD simulation as it was designed
    for chromatic adaptation. It is mostly here for comparison purposes.
    """

    # Taken from the sRGB spec from https://en.wikipedia.org/wiki/SRGB
    XYZ_from_linearRGB = XYZ_from_linearRGB_BT709

    # From Wikipedia https://en.wikipedia.org/wiki/LMS_color_space
    # Hunt-Pointer-Esterez is older and supposedly worse than e.g. Bradford
    # from chromatic adaptation, but it is better to simulate the human
    # cone responses, which is what we need for CVD simulation.
    # This transform is the version normalized for D65, which means that
    # L=M=S=1 for the XYZ=D65 white point. It does not really matter
    # though, the simulation will be the same with a different
    # normalization. FIXME: is that true???
    LMS_from_XYZ = np.array([
        [ 0.4002, 0.7076, -0.0808],
        [-0.2263, 1.1653,  0.0457],
        [ 0,      0,       0.9182]])

    # np.linalg.inv(LMS_from_XYZ)
    # XYZ_from_LMS = np.array([
    #     [ 1.86006661, -1.12948008,  0.2198983 ],
    #     [ 0.36122292,  0.63880431, -0.00000713],
    #     [ 0.        ,  0.        ,  1.08908734]])

    # (LMS_from_XYZ @ XYZ_from_linearRGB)
    # LMS_from_linearRGB = np.array([
    #     [0.31398949, 0.63951294, 0.04649755],
    #     [0.15537141, 0.75789446, 0.08670142],
    #     [0.01775156, 0.10944209, 0.87256922]])

    def __init__(self):
        super().__init__(self.XYZ_from_linearRGB, self.LMS_from_XYZ)

class LMSModel_sRGB_MCAT02 (LMSModel):
    """Combines sRGB to XYZ and the sharpened transformation matrix of CIECAM02 to go to LMS.

    WARNING: this model is not recommended for CVD simulation as it was designed
    for chromatic adaptation. It is mostly here for comparison purposes.
    """

    # Taken from the sRGB spec from https://en.wikipedia.org/wiki/SRGB
    XYZ_from_linearRGB = XYZ_from_linearRGB_BT709

    LMS_from_XYZ = np.array([
        [0.7328, 0.4296, -0.1624],
        [-0.7036, 1.6975, 0.0061],
        [0.0030, 0.0136, 0.9834]])

    # = LMS_from_XYZ @ XYZ_from_linearRGB
    # LMS_from_linearRGB = np.array([
    #     [0.3904725 , 0.54990437, 0.00890159],
    #     [0.07092586, 0.96310739, 0.00135809],
    #     [0.02314268, 0.12801221, 0.93605194]])

    def __init__(self):
        super().__init__(self.XYZ_from_linearRGB, self.LMS_from_XYZ)

class LMSModel_sRGB_StockmanSharpe2000 (LMSModel):
    """Combines sRGB to XYZ and the LMS from XYZ matrix from Stockman & Sharpe 2000

    Should be a reasonable alternative to (Smith & Pokorny, 1975) for the 2° standard
    observer, but it has not been used extensively for CVD simulation.
    """

    # Taken from the sRGB spec from https://en.wikipedia.org/wiki/SRGB
    XYZ_from_linearRGB = XYZ_from_linearRGB_BT709

    LMS_from_XYZ = np.array([
        [1.94735469, -1.41445123, 0.36476327],
        [0.68990272,  0.34832189, 0],
        [0,           0,          1.93485343]])

    def __init__(self):
        super().__init__(self.XYZ_from_linearRGB, self.LMS_from_XYZ)
