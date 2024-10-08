#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 2 13:31:59 2022

@author: gjadick
"""
import numpy as np
import cupy as cp
import xcompy as xc
import json
import csv

ELEMENTS =  ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si',\
    'P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',\
    'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh',\
    'Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',\
    'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re',\
    'Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th',\
    'Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm']

    
class VoxelPhantom:
    '''
    Class to handle a 3D voxelized phantom and corresponding params.
    
    phantom_id -- name for the phantom (used for saving outputs)
    filename -- name of the binary file containing the phantom data. 
                should be formatted as a raveled array of integers, 
                each of which corresponds to a material with elemental
                composition and density listed in the matcomp file
    matcomp_filename -- file with the material compositions corresponding 
                        to each integer of the phantom file. Should be 
                        formatted with four tab-separated columns: ID, 
                        material name, density, elemental composition by 
                        weight (see example file)
    Nx, Ny, Nz -- shape of the phantom in 3 dimensions (if 2D, set Nz=1)
                  The x-y plane corresponds to each axial image.
    ind -- z-index of the phantom slice to use if 3D (if 2D, leave ind=0)
    dx, dy, dz -- size of each voxel in cm
    dtype -- data type for raveled phantom file (default uint8)
    '''
    def __init__(self, name, phantom_filename, matcomp_filename, Nx, Ny, Nz, 
                 dx=0.1, dy=0.1, dz=0.1, z_index=0, dtype=np.uint8):

        self.name = name
        self.phantom_filename = phantom_filename
        self.matcomp_filename = matcomp_filename

        # Number of voxels along each axis
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        
        # Size of voxels in each dimension [cm]
        self.dx = dx
        self.dy = dy
        self.dz = dz
        
        # Load the 3D phantom, each uint8 value corresponds to some material
        self.M3D = np.fromfile(phantom_filename, dtype=dtype).reshape([Nz, Ny, Nx])
        
        # Assign a 2D cross-section for imaging.
        self.z_index = z_index
        self.M = self.M3D[z_index]        
        
        # Create dictionary of density and atomic composition for each phantom voxel.
        if matcomp_filename.endswith('.csv'):
            self.matcomp_dict = material_csv_to_dict(matcomp_filename)
        else:
            self.matcomp_dict = get_matcomp_dict(matcomp_filename)
        self.matkeys = np.array(list(self.matcomp_dict.keys()), dtype=dtype)

    def set_M_index(self, z_index):
        self.M = self.M3D[z_index]
        self.z_index = z_index
        
    def M_mono_stack(self, energies):
        N_energies = len(energies)
        M_mono_stack = cp.zeros([N_energies, self.Nx, self.Ny], dtype=cp.float32)        
        for i_mat in self.matkeys:
            density, matcomp = self.matcomp_dict[i_mat] 
            u_E = density * cp.array(xc.mixatten(matcomp, energies), dtype=cp.float32)
            for i_E in range(N_energies):
                cp.place(M_mono_stack[i_E], self.M==i_mat, u_E[i_E])
        return M_mono_stack        
    
    def M_mono(self, E0, HU=True):  
        E0_arr = np.array([E0], dtype=np.float64)
        M_mono = cp.zeros([self.Nx, self.Ny], dtype=cp.float32)        
        for i_mat in self.matkeys:
            density, matcomp = self.matcomp_dict[i_mat] 
            mu_mono = density * float(xc.mixatten(matcomp, E0_arr))
            cp.place(M_mono, self.M==i_mat, mu_mono)
        if HU:  # convert attenuation to HU 
            mu_water = float(xc.mixatten('H(11.2)O(88.8)', E0_arr))
            M_mono = 1000 * (M_mono - mu_water) / mu_water
        return M_mono


def get_matcomp_dict(filename):
    '''
    Convert material composition file into a dictionary of density/matcomp strings.
    '''
    with open(filename, 'r') as f:
        L_raw = [l.strip() for l in f.readlines() if len(l.strip())]
    mat_dict = {}
    header = L_raw[0].split()
    for line in L_raw[1:]:
        split = line.split()  # separate into four columns
        N    = int(split[0])
        name = split[1]
        density = float(split[2])
        matcomp = split[3]        
        mat_dict[N] = [density, matcomp]  # add dictionary entry
    return mat_dict


def material_csv_to_dict(filename):
    """
    Read a CSV file containing material data into a dictionary. Each row
    gives the density and chemical composition by weight fraction for a given
    material name. These entries should correspond to a voxelized phantom with
    voxel values equal to different ID numbers (0 to 255) that are used to
    identify the material of that voxel. So the CSV file contains the needed
    material information for computing material-dependent parameters like the
    linear attenuation coefficient and anomolous scattering factors.
    
    An example file is `input/materials.csv`. 

    Parameters
    ----------
    filename : str
        Path to the material data file (csv format).

    Returns
    -------
    material_dictionary : dict
        Dictionary of material density and chemical composition by weight
        corresponding to each material ID number in a voxelized phantom.

    """
    material_dictionary = {}  
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = int(row['ID'])
            density = float(row['density'])
            matcomp = row_to_matcomp(row)
            material_dictionary[key] = [density, matcomp]
    return material_dictionary


def row_to_matcomp(row):
    """
    Convert a row from a CSV file (read from the file to a dictionary format)
    with the chemical weight fractions to a formatted material composition 
    string, e.g. for water 'H(11.2)O(88.8)'.

    Parameters
    ----------
    row : dict
        Dictionary of row items read from the material CSV file.

    Returns
    -------
    matcomp : str
        Formatted material composition string of elements by weight fractions.

    """
    matcomp = ''
    for elem in ELEMENTS:
        try:
            if row[elem] != '':
                matcomp += f'{elem}({row[elem]})'
        except: 
            pass  # elem not in composition
    return matcomp


class FanBeamGeometry:
    def __init__(self, eid=True, detector_file=None, detector_std_electronic=0, h_iso=1.0,
                 SID=50.0, SDD=100.0, N_channels=360, gamma_fan=np.pi/4, 
                 N_proj=1000, theta_tot=2*np.pi, spectral_threshold=None):
        
        self.geometry = 'fan_beam'
        self.noise = True  # set False for testing
        
        self.eid = eid  
        if eid: 
            self.det_mode = 'eid'  # energy integrating
        else:
            self.det_mode = 'pcd'  # photon counting
        self.spectral_threshold = spectral_threshold  # for spectral PCD mode [keV]

        self.std_e = detector_std_electronic  # electronic noise standard dev
        if (detector_file is None) or (detector_file == 'ideal'):  # ideal detector? 2 data points in case we need to interp
            self.det_E = np.array([1.0, 1000.0], dtype=np.float32)
            self.det_eta_E = np.array([1.0, 1.0], dtype=np.float32)
        else:
            data = np.fromfile(detector_file, dtype=np.float32)
            N_det_energy = len(data)//2
            self.det_E = data[:N_det_energy]      # 1st half is energies
            self.det_eta_E = data[N_det_energy:]  # 2nd half is detective efficiencies
 
        # name the geometry       
        # this is old! missing a few params (electronic noise, detector efficiency)
        self.geo_id = f'{int(SID)}cm_{int(SDD)}cm_{int(180*gamma_fan/np.pi)}fan_{N_proj}view_{N_channels}col_{self.det_mode}'
        
        # source-isocenter and source-detector distances
        self.SID = SID
        self.SDD = SDD
        
        # multi-channel fan-beam detector (angles `gamma`)
        self.N_channels = N_channels
        self.gamma_fan = gamma_fan
        self.dgamma = gamma_fan / N_channels  
        self.gammas = np.arange(-gamma_fan/2, gamma_fan/2, self.dgamma) + self.dgamma/2  ## might need to offset ?
        
        # detector pixel dimensions
        self.s = SDD*gamma_fan/N_channels  # width [cm] i.e. `s` sampling distance at the detector
        self.s_iso = SID*gamma_fan/N_channels  # width [cm] at isocenter `iso`
        self.h_iso = h_iso  # height [cm2], for fanbeam same at detector and isocenter
        self.A_iso = self.s_iso*self.h_iso   # area [cm2]
        
        # projection views (angles `theta`)
        self.N_proj = N_proj
        self.theta_tot = theta_tot
        self.dtheta = theta_tot/N_proj
        self.thetas = np.arange(0, theta_tot, self.dtheta )
        
        # just a check
        if len(self.thetas) > N_proj:
            self.thetas = self.thetas[:N_proj]

        if len(self.gammas) > N_channels:
            self.gammas = self.gammas[:N_channels]
            
            
            
class ParallelBeamGeometry:
    def __init__(self, eid=True, detector_file=None, detector_std_electronic=0, h_iso=1.0,
                 SID=50.0, SDD=100.0, N_channels=360, beam_width=50.0, 
                 N_proj=1000, theta_tot=2*np.pi):
        
        self.geometry = 'parallel_beam'
        self.noise = True  # set False for testing
        self.eid = eid  
        if eid: 
            self.det_mode = 'eid'  # energy integrating
        else:
            self.det_mode = 'pcd'  # photon counting
            
        self.std_e = detector_std_electronic  # electronic noise standard dev
        if (detector_file is None) or (detector_file == 'ideal'):  # ideal detector? 2 data points in case we need to interp
            self.det_E = np.array([1.0, 1000.0], dtype=np.float32)
            self.det_eta_E = np.array([1.0, 1.0], dtype=np.float32)
        else:
            data = np.fromfile(detector_file, dtype=np.float32)
            N_det_energy = len(data)//2
            self.det_E = data[:N_det_energy]      # 1st half is energies
            self.det_eta_E = data[N_det_energy:]  # 2nd half is detective efficiencies
         
        # source-isocenter and source-detector distances
        self.SID = SID
        self.SDD = SDD
        
        # multi-channel parallel-beam detector 
        self.N_channels = N_channels
        self.beam_width = beam_width
        self.s = beam_width / N_channels # pixel width, i.e. sampling distance at detector
        self.channels = np.arange(-beam_width/2, beam_width/2, self.s) + self.s/2  ## might need to offset ?
        
        # detector pixel dimensions
        self.s_iso = self.s  # equivalent for parallel beam
        self.h_iso = h_iso  # equivalent without cone
        self.A_iso = self.s_iso*self.h_iso   # area [cm2]
        
        # projection views (angles `theta`)
        self.N_proj = N_proj
        self.theta_tot = theta_tot
        self.dtheta = theta_tot/N_proj
        self.thetas = np.arange(0, theta_tot, self.dtheta )
        
        # just a check
        if len(self.thetas) > N_proj:
            self.thetas = self.thetas[:N_proj]

        if len(self.channels) > N_channels:
            self.channels = self.channels[:N_channels]


    
class xRaySpectrum:
    def __init__(self, filename, name, mono_E=None):
            
        # Effective mu_water and mu_air dictionaries for HU conversions,
        # found by simulating noiseless images of water phantom with each 
        # spectrum and measuring the mu value in its center (150 projections, 
        # 100 detector channels, 1 mGy dose).
        # These might be affected by beam hardening.
        u_water_dict = {
            '6MV':       0.04268331080675125  ,
            'detunedMV': 0.05338745564222336  ,
            '80kV':      0.24212932586669922  ,
            '120kV':     0.21030768752098083  ,
            '140kV':     0.2016972303390503  }
        u_air_dict = {
            '6MV':       0.00024707260308787227  ,
            'detunedMV': 0.00031386411865241826  ,
            '80kV':      0.002364289714023471  ,
            '120kV':     0.0016269732732325792  ,
            '140kV':     0.0014648198848590255  }
        E_eff_dict = {  # linear interp of u_water_dict with NIST curve [keV]
            '6MV':       2692.36  ,
            'detunedMV': 1753.73  ,
            '80kV':      46.32  ,
            '120kV':     57.91, 
            '140kV':     63.79    }
        
        self.filename = filename 
        self.name = name
        try:  
            data = np.fromfile(filename, dtype=np.float32)
            self.E, self.I0 = data.reshape([2, data.size//2])
            self.u_water = u_water_dict[name]
            self.u_air = u_air_dict[name]
            self.E_eff = E_eff_dict[name]
        except:
            print(f"Failed to open spectrum filename {filename}, failed to initialize.")
        
        # For debugging, can use a monoenergetic x-ray beam.
        if mono_E is not None:
            print(f'Debugging! Monoenergetic on! {mono_E} keV')
            self.E = np.array([mono_E])
            self.I0_raw = np.array([1.0e8]) # arbitrary counts
            self.name = f'mono{mono_E:04}keV'

    def get_counts(self):
        return np.trapz(self.I0, x=self.E)
    
    def rescale_counts(self, scale, verbose=False):
        if verbose:
            print(f'rescaled counts : {self.get_counts():.2e} -> {scale*self.get_counts():.2e}')
        self.I0 = self.I0 * scale


## for reading in the params
def make_combos(var_list, combos=None, this_col=None):
    """
    Recursive function to generate a list of mixed-type lists, all possible 
    combinations of the sub-list variables in var_list. Used for creating 
    combinations of looping variables. Example:
        
    >>> make_combos([[True, False], [100,200], ['a','b','c']])
    [[True, 100, 'a'], [True, 100, 'b'], [True, 100, 'c'], [True, 200, 'a'],  
     [True, 200, 'b'], [True, 200, 'c'], [False, 100, 'a'], [False, 100, 'b'],  
     [False, 100, 'c'], [False, 200, 'a'], [False, 200, 'b'], [False, 200, 'c']]
    """

    def place_col(M, col, x):
        for i in range(len(x)):
            M[i][col] = x[i]

    def place_M(M, mini_M, i, j):
        for x in range(len(mini_M[0])):
            for y in range(len(mini_M)):
                M[j+y][i+x] = mini_M[y][x]
                
    # make everything a list
    for i in range(len(var_list)):
        if not isinstance(var_list[i], list):
            var_list[i] = [var_list[i]]  # assuming this is a single value...

    N = np.prod([len(v) for v in var_list])  # total num of combination vectors
    if combos is None:  # initialize things that update recursively later
        combos = [['foo' for i in range(len(var_list))] for j in range(N)]  
        this_col = 0  

    m = int(N / len(var_list[0])) # number of times to repeat first vector
    place_col(combos, 0, np.repeat(var_list[0], m))

    if var_list[1:]: # if there are more variables, loop over them
        this_col += 1
        sub_combo = make_combos(var_list[1:], combos=[v[1:] for v in combos][:m], this_col=this_col)
        for i_opt in range(len(var_list[0])):
            place_M(combos, sub_combo, 1, i_opt*m)

    return combos


def read_parameter_file(filename):
    """
    Read a json-formatted parameter file.

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    all_params : TYPE
        DESCRIPTION.

    """
    
    with open(filename) as f:
        all_parameters = json.load(f)
    
    # Make dictionaries for each set of parameter combinations
    param_keys = list(all_parameters.keys())
    param_value_combos = make_combos(list(all_parameters.values()))
    param_dicts = [dict(zip(param_keys, values)) for values in param_value_combos]
    
    ## Package the parameters into objects for each run
    all_params = []
    for p in param_dicts:  # might want to package into dictionaries later
        
        ## 1 : simulations to run
        run_id = p['RUN_ID']
        do_forward_projection = p['forward_project']
        do_back_projection = p['back_project']
        these_params = [run_id, do_forward_projection, do_back_projection]  # init

        ## 2 : scanner geometry
        eid = p['detector_mode'] == 'eid'  # convert to bool
        if 'spectral_threshold' in p:  # spectral PCD mode?
            p_spectral_thresh = p['spectral_threshold']
        else:
            p_spectral_thresh = None
        if p['scanner_geometry'] == 'fan_beam':
            ct = FanBeamGeometry(N_channels=p['N_channels'], 
                                 N_proj=p['N_projections'],
                                 gamma_fan=p['fan_angle_total'],  # only difference with ParallelBeam params!
                                 theta_tot=p['rotation_angle_total'],
                                 SID=p['SID'], 
                                 SDD=p['SDD'], 
                                 eid=eid, 
                                 h_iso=p['detector_px_height'],
                                 detector_file=p['detector_filename'],
                                 detector_std_electronic=p['detector_std_electronic'],
                                 spectral_threshold=p_spectral_thresh)
        elif p['scanner_geometry'] == 'parallel_beam':
            ct = ParallelBeamGeometry(N_channels=p['N_channels'], 
                                      N_proj=p['N_projections'],
                                      beam_width=p['beam_width'],  # only difference with FanBeam params!
                                      theta_tot=p['rotation_angle_total'],
                                      SID=p['SID'], 
                                      SDD=p['SDD'], 
                                      eid=eid, 
                                      h_iso=p['detector_px_height'],
                                      detector_file=p['detector_filename'],
                                      detector_std_electronic=p['detector_std_electronic'])
        these_params.append(ct)
        
        ## 3 : phantom
        if p['phantom_type'] == 'voxel':
            phantom = VoxelPhantom(name=p['phantom_id'],
                                   phantom_filename=p['phantom_filename'],
                                   matcomp_filename=p['matcomp_filename'],
                                   Nx=p['Nx'], Ny=p['Ny'], Nz=p['Nz'],
                                   dx=p['dx'], dy=p['dy'], dz=p['dz'],
                                   z_index=p['z_index'])
        these_params.append(phantom)

        ## 4 : x-ray energy spectrum
        try:
            spectrum = xRaySpectrum(name=p['spectrum_id'],
                                    filename=p['spectrum_filename'])
            # Scale photon counts delivered to each detector channel (at isocenter) per projection.
            counts_raw = spectrum.get_counts()
            counts_per_proj_per_channel = p['N_photons_per_cm2_per_scan'] * ct.A_iso / ct.N_proj
            spectrum.rescale_counts(counts_per_proj_per_channel / counts_raw)
            these_params.append(spectrum)
        except:
            these_params.append('NO_SPECTRUM_ASSIGNED')

        if do_back_projection:
            these_params.append(p['N_recon_matrix'])  # N_matrix
            these_params.append(p['FOV_recon'])  # FOV
            these_params.append(p['ramp_filter_percent_Nyquist'])  # ramp
        
        if 'N_repeats' in p:
            for i in range(p['N_repeats']):
                these_params[0] = p['RUN_ID'] + f'/repeat{i:03}'
                all_params.append(these_params)
        else:
            all_params.append(these_params)

    return all_params


    
