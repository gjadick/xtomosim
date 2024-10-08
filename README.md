# xtomosim

A Python-based x-ray tomographic imaging simulator. Xtomosim is capable of:

1. Axial computed tomography (CT) forward projecting with either fan-beam or parallel-beam geometry.
2. Filtered back-projection recontruction with your desired level of noise suppression.
3. Beam-hardening correction (either water or water-and-bone technique)


## Quick start

First, I recommend creating a virtual environment in the main xtomosim directory
for managing packages. 

```
python -m venv xtomosim-env
source xtomosim-env/bin/activate
python -m pip install --upgrade pip
```

Once you are in the desired environment, install the required packages:
```
python -m pip install -r requirements.txt
```

Now you are ready to simulate a CT scan. An example parameter file **input/parameters_example.txt**
and corresponding phantom, spectrum, and detector files have been provided for a test run:
```
python main.py
```
The resulting sinograms and reconstructed images will be saved in a new **output/** directory.
 
 
 
## Simulations

### Parameter files
 
For your own simulations, in **main.py**, assign the *param_file* variable to 
a string with the path to an xtomosim parameter file. This is a json file with 
the following required fields:

| **Parameter name** | **Type** | **Description** |
| ---------------- | --------| ----- |
| *RUN_ID* | str | Unique identifier for the simulation. This will become the directory any outputs. If it is not changed for different simulations, old data will be overwritten. |
| *forward_project* | bool | Whether to simulate sinograms. |
| *back_project* | bool | Whether to reconstruct sinograms. If *forward_project* is false, then the sinogram files must already exist in the **{main_output_dir}/{RUN_ID}/** directory.
| *phantom_type* | str | "voxel" (other analytical options coming soon) |
| *phantom_id* | str | Unique identifier for the phantom. | 
| *phantom_filename* | str | Path to the raw uint8 file with the phantom material indices. Should be raveled from the shape [Nx, Ny, Nz]. |
| *matcomp_filename* | str | Path to the csv file with the density and atomic composition corresponding to each material index in the phantom file. (see example) |
| *Nx* | int >= 1 | Number of pixels in the phantom x-direction (lateral). |
| *Ny* | int >= 1| Number of pixels in the phantom y-direction (anterio-posterior). |
| *Nz* | int >= 1| Number of pixels in the phantom z-direction (axial). |
| *dx* | float | Size of phantom pixels in x-direction [cm] |
| *dy* | float | Size of phantom pixels in y-direction [cm] |
| *dz* | float | Size of phantom pixels in z-direction [cm] |
| *z_index* | int >= 0 | Index of the axial phantom slice for the simulation. Only one "row" may be simulated at a time. |
| *scanner_geometry* | str, "parallel_beam" OR "fan_beam" | Simulation geometry. Note that the geometry affects required params below. |
| *SID* | float > 0 | source-to-isocenter distance [cm] |
| *SDD* | float > *SID* | source-to-detector distance [cm] |
| *N_channels* | int >= 1 | Number of detector channels. These will be equally spaced within *beam_width* (parallel beam) OR *fan_angle_total* (fan beam). |
| *N_projections* | int >= 1| Number of projection views. These will be equally spaced within *rotation_angle_total*. |
| *beam_width* -- ONLY for parallel-beam sims | float > 0 | Total detector width [cm]. |
| *fan_angle_total* -- ONLY for fan-beam sims | float > 0 | Total detector arc angle [rad]. Note that this is NOT the angle of half the detector. |
| *rotation_angle_total* | float > 0 | Total rotation angle of the x-ray source. |
| *detector_px_height* | float > 0 | Height of detector pixels [cm] (used for determining total x-ray flux from the spectrum file). |
| *detector_mode* | str, "eid" OR "pcd" | Detection scheme, either energy-integrating or photon-counting. |
| *detector_std_electronic* | int | Standard deviation of additive Poisson noise in units of photon counts. This emulates electronic noise. |
| *detector_filename* | "ideal" OR str | Path to raw float32 file with the detector efficiency-vs-energy [keV] data. Set to "ideal" to simply use efficiency = 1.0 for all energies. |
| *spectrum_id* | str | Unique identifier for the x-ray spectrum. |
| *spectrum_filename* | str | Path to the raw float32 file with the counts-vs-energy [keV] data of the incident x-ray spectrum. |
| *N_photons_per_cm2_per_scan* | int | Incident x-ray spectrum flux [photon counts per cm^2]. This affects image noise. |
| *N_recon_matrix* | int | Number of pixels in the x- and y-direction of the reconstructed image. |
| *FOV_recon* | float | Field-of-view of the reconstructed image [cm]. |
| *ramp_filter_percent_Nyquist* | float > 0.0 AND <= 1.0 | Reconstruction filter cutoff frequency. |
    
Note that certain parameters are linked to others. For example, the final three 
*N_recon_matrix, FOV_recon, ramp_filter_percent_Nyquist* are only required if *back_project* is true.
Also, *scanner_geometry* == "parallel_beam" requires assigning *beam_width*,
whereas *scanner_geometry* == "fan_beam" requires assigning *fan_angle_total* instead.
 
See **input/** for examples of valid inputs for *phantom_filename, matcomp_filename, detector_filename, spectrum_filename*.
 
 
 
### Other parameters

Within **main.py**, there are four other variables that can be assigned:

| *Variable* | *Type* |*Description* |
| -----------| ------ | ------------ |
| *main_output_dir* | str | Path to where the simulated sinograms and images should be stored. |
| *show_imgs* | bool | Whether to show intermediate images while **main.py** is running. |
| *do_bhc_water* | bool | Whether to perform a water-based polynomial curve fit beam-hardening correction. |
| *do_bhc_bone* | bool | Whether to perform a two-material (water and bone) beam-hardening correction. This may require manual tuning in the *bhc_bone()* function of **bhc.py**. |

(Coming soon -- the *do_bhc_water* and *do_bhc_bone* vars will be moved to the parameter file)



## Misc.

Tested in Python 3.12.1.
















