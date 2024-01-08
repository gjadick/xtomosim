#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Oct 25 14:22:31 2023

@author: giavanna
'''

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from system import read_parameter_file
from forward_project import get_sino
from back_project import get_recon
from bhc import bhc_water, bhc_bone

param_file = 'input/parameters_example.txt'
main_output_dir = './output/'  
show_imgs = True
do_bhc_water = False   # TODO : move this to param file
do_bhc_bone = True

plt.rcParams.update({
    'figure.dpi': 300,
    'font.size':10,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': True,
    'axes.titlesize':10,
    'axes.labelsize':8,
    'axes.linewidth': .5,
    'xtick.top': True, 
    'ytick.right': True, 
    'xtick.direction': 'in', 
    'ytick.direction': 'in',
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    'axes.grid' : False, 
    'grid.color': 'lightgray',
    'grid.linestyle': ':',
    'legend.fontsize':8,
    'lines.linewidth':1,
    'image.cmap':'gray',
    'image.aspect':'auto'
    })


def ax_imshow(fig, ax, img, colorbar=True, title='', kw={}):
    ax.set_title(title)
    m = ax.imshow(img, **kw)
    if colorbar:
        fig.colorbar(m, ax=ax)


if __name__ == '__main__':

    all_params = read_parameter_file(param_file)
    N_runs = len(all_params)
        
    for i_run, params in enumerate(all_params):
        print('\n', f'*** Run {i_run + 1} / {N_runs} ***')
        
        ## Unpack the parameters for this loop.
        run_id, do_forward_projection, do_back_projection = params[:3]
        print(run_id)
        ct, phantom, spec = params[3:6]
        if do_back_projection:
            N_matrix, FOV, ramp = params[6:9]
        out_dir = os.path.join(main_output_dir, run_id, f'run_{i_run:03}/')
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(param_file, os.path.join(out_dir, 'params.txt'))
 
        if do_forward_projection:
            print('Forward projecting!')
            d_sino_raw, d_sino_log = get_sino(ct, phantom, spec, on_gpu=True)   
            sino_raw, sino_log = d_sino_raw.get(), d_sino_log.get()
            sino_raw.astype(np.float32).tofile(out_dir+'sino_raw_float32.bin')
            sino_log.astype(np.float32).tofile(out_dir+'sino_log_float32.bin')
            
            if do_bhc_water or do_bhc_bone: 
                d_sino_waterBHC = bhc_water(d_sino_log, spec, ct)
                sino_waterBHC = d_sino_waterBHC.get()
                sino_waterBHC.astype(np.float32).tofile(out_dir+'sino_waterBHC_float32.bin')
                
            if show_imgs:
                fig, ax = plt.subplots(1, 2, figsize=[7,3])
                ax_imshow(fig, ax[0], sino_log, title='Log sinogram')
                try:
                    ax_imshow(fig, ax[1], sino_waterBHC, title='Log + water BHC')
                except:
                    ax_imshow(fig, ax[1], sino_raw, title='Raw line integrals')
                fig.tight_layout()
                plt.show()
                
        if do_back_projection:
            print('Back projecting!')
        
            # Load the sinogram + assign output name
            if do_bhc_bone:
                sino_fname = 'sino_waterBHC_float32.bin'
                recon_fname = 'recon_boneBHC'
            elif do_bhc_water:
                sino_fname = 'sino_waterBHC_float32.bin'
                recon_fname = 'recon_waterBHC'
            else:
                sino_fname = 'sino_log_float32.bin'
                recon_fname = 'recon'
            sino_log = np.fromfile(out_dir + sino_fname, dtype=np.float32)
            sino_log = sino_log.reshape([ct.N_proj, ct.N_channels])    
            
            if do_bhc_bone:
                recon_raw, recon_HU = bhc_bone(sino_log, spec, ct, N_matrix, FOV, ramp)
            else:
                recon_raw, recon_HU = get_recon(sino_log, ct, spec, N_matrix, FOV, ramp) 
            recon_raw.astype(np.float32).tofile(out_dir + recon_fname + '_raw_float32.bin')
            recon_HU.astype(np.float32).tofile(out_dir + recon_fname + '_HU_float32.bin')
            
            if show_imgs:
                fig, ax = plt.subplots(1, 2, figsize=[7,3])
                ax_imshow(fig, ax[0], recon_raw, title='Raw reconstruction [cm$^{-1}$]')
                ax_imshow(fig, ax[1], recon_HU, title='Final reconstruction [HU]')
                fig.tight_layout()
                plt.show()
                



