#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:19:23 2023

@author: giavanna
"""

import xcompy as xc
import numpy as np
import cupy as cp

from back_project import get_recon
from forward_project import get_sino_from_recon
        
        
def bhc_water(d_sino_log, spec, ct, deg=4, EPS=1e-8, mu_eff=None):
    """
    Water beam hardening correction using polynomial fit method of Hsieh.
    Assumes sino_log is still on device (cupy array) 

    Parameters
    ----------
    d_sino_log : 2D cupy array
        The log'd sinogram.
    spec : xRaySpectrum
        Spectrum for the sinogram acquisition.
    ct : ScannerGeometry
        Geometry for the sinogram acquisition.
    deg : int, optional
        For water BHC, degree of the polynomial. The default is 4.
    EPS : float, optional
        Small value to avoid div by zero errors. The default is 1e-8.
    mu_eff : float, optional
        Effect water linear attenuation coefficient. The `spec` passed might
        include a value for this, but you might want to hand-tune.
        The default is None.
    Returns
    -------
    d_sino_corrected : 2D cupy array

    """
    d_sino = d_sino_log.clip(EPS)  # enforce non-negativity
    mu_water = xc.mixatten('H2O', spec.E)
    L = np.linspace(0, 50, 100) # pathlengths through cm water (max 50cm FOV?)
    
    # Compute the detector response function (efficiency + energy weighting)
    det_response = np.interp(spec.E, ct.det_E, ct.det_eta_E)
    if ct.eid:
        det_response *= spec.E
    
    # Polychromatic signal vs. water thickness, where signal `P` ~ mu * L
    counts_i = np.trapz(spec.I0 * det_response, x=spec.E)  # incident counts
    counts_atten = np.array([np.trapz(spec.I0 * det_response * np.exp(-mu_water*l), x=spec.E) for l in L])   
    P_polychrom = np.log(counts_i / counts_atten)
    
    # Estimate monochromatic data `P_mono` ~ mu_eff * L
    if mu_eff is None:  # check if spectrum has effective energy
        try:
            E_eff = spec.E_eff
        except:
            E_eff = np.sum(spec.E * spec.I0)/np.sum(spec.I0)  
        mu_eff = np.interp(E_eff, spec.E, mu_water)
    P_mono = mu_eff * L 

    # Correct polychromatic data using least squares polynomial fit
    coeffs = np.polyfit(P_polychrom, P_mono, deg)  
    f_bhc = cp.poly1d(coeffs)
    d_sino_corrected = f_bhc(d_sino.ravel()).reshape(d_sino.shape)

    return d_sino_corrected


def bhc_bone(sino_log, spec, ct, N_matrix, FOV, ramp,
                    deg=4, EPS=1e-8, mu_eff=None, mu_bone_thresh=0.3):
    """
    Apply the bone beam hardening correction described in Hsieh's textbook.
    This takes the original sinogram as input and then applies the water BHC.
    This bone BHC algorithm involves considerably more hand-tuning than the 
    polynomial fit water BHC implemented above.

    Parameters
    ----------
    sino_log : numpy array 
        Original log'd sinogram (without any BHC).
    spec : xRaySpectrum
        Spectrum for the sinogram acquisition.
    ct : ScannerGeometry
        Geometry for the sinogram acquisition.
    N_matrix : int
        Size of the reconstruction matrix.
    FOV : float
        Field-of-view for the reconstruction matrix [cm].
    ramp : float between 0 and 1
        Percentage of Nyquist frequency for FBP recon ramp filter.
    deg : int, optional
        For water BHC, degree of the polynomial. The default is 4.
    EPS : float, optional
        Small value to avoid div by zero errors. The default is 1e-8.
    mu_eff : float, optional
        Effect water linear attenuation coefficient. The `spec` passed might
        include a value for this, but you might want to hand-tune.
        The default is None.
    mu_bone_thresh : float, optional
        The cutoff linear attenuation coefficient used to create a "bone" 
        image. I hand-tuned it, but there is likely a better way.
        The default is 0.3 [cm^-1], or roughly 500 HU

    Returns
    -------
    recon_bone_corrected, recon_bone_corrected_HU : 2D numpy arrays

    """
    # First, apply water BHC to sinogram + reconstruct.
    d_sino_log = cp.array(sino_log, dtype=cp.float32)
    d_sino_waterBHC = bhc_water(d_sino_log, spec, ct, deg, EPS, mu_eff)    
    recon_waterBHC, _ = get_recon(d_sino_waterBHC.get(), ct, spec, N_matrix, FOV, ramp) 

    # mu_bone_thresh = 0.3 # cm^-1  ~ 500 HU (hand-tuned)
    # Create a "bone image" using bone mu threshold
    recon_bone = recon_waterBHC.copy()  # init
    recon_bone[recon_bone < mu_bone_thresh] = 0.0
    
    # Reconstruct an "artifact image"
    d_sino_bone = get_sino_from_recon(ct, recon_bone, FOV)  # error sinogram = squared bone image
    d_sino_bone2 = d_sino_bone**2  
    recon_bone_artifact, _ = get_recon(d_sino_bone2.get(), ct, spec, N_matrix, FOV, ramp)

    # Correct the artifact using a linear combination of the water BHC image + bone artifact image.
    # These artifact scales have been hand-tuned for a few spectra. There's probably a better way to compute them.
    scale_dict = {'80kV': 0.028, '120kV': 0.053, '140kV': 0.065}
    try:
        scale = scale_dict[spec.name]
    except:
        scale = 0.05  # default to 5% scale (arbitrary!! YMMV)

    recon_bone_corrected = recon_waterBHC + scale*recon_bone_artifact
    recon_bone_corrected_HU = 1000 * (recon_bone_corrected - spec.u_water) / (spec.u_water - spec.u_air)
    
    return recon_bone_corrected, recon_bone_corrected_HU





