'''
Braden Nowicki with Dr. Massimo Ricotti
University of Maryland, College Park Astronomy Department

Script to visualize RAMSES-RT Simulations of high-redshift galaxies in a variety of metal lines. 
Ionization Parameter, Number Density, and Temperature for each pixel are input into an interpolator 
for each line; the interpolator is created via the module 'emission.py'. 'emission.py' currently 
uses the 'linelist.dat' datatable to build interpolators; this can be adjusted to work with other 
tables from Cloudy runs. 
'''

'''
Setup fields in yt
'''

# importing packages
import numpy as np
import os
#import matplotlib.pyplot as plt
import emission
#import astropy
import yt
#from yt.units import dimensions
import copy
#from scipy.special import voigt_profile
#from astropy.cosmology import FlatLambdaCDM
#from matplotlib.colors import LogNorm
import sys
import galaxy_visualization

#f1 = "/Users/bnowicki/Documents/Research/Ricotti/output_00273"
filename = sys.argv[1]

# TODO user input which fields to plot, width, etc.

# Cloudy Grid Run Bounds (log values)
# Umin, Umax, Ustep: -6.0 1.0 0.5
# Nmin, Nmax, Nstep: -1.0 6.0 0.5 
# Tmin, Tmax, Tstop: 3.0 6.0 0.1

lines=["H1_6562.80A","O1_1304.86A","O1_6300.30A","O2_3728.80A","O2_3726.10A","O3_1660.81A",
       "O3_1666.15A","O3_4363.21A","O3_4958.91A","O3_5006.84A", "He2_1640.41A","C2_1335.66A",
       "C3_1906.68A","C3_1908.73A","C4_1549.00A","Mg2_2795.53A","Mg2_2802.71A","Ne3_3868.76A",
       "Ne3_3967.47A","N5_1238.82A","N5_1242.80A","N4_1486.50A","N3_1749.67A","S2_6716.44A","S2_6730.82A"]

wavelengths=[6562.80, 1304.86, 6300.30, 3728.80, 3726.10, 1660.81, 1666.15, \
             4363.21, 4958.91, 5006.84, 1640.41, 1335.66, \
             1906.68, 1908.73, 1549.00, 2795.53, 2802.71, 3868.76, \
             3967.47, 1238.82, 1242.80, 1486.50, 1749.67, 6716.44, 6730.82]

'''
cell_fields = [
    "Density",
    "x-velocity",
    "y-velocity",
    "z-velocity",
    "Pressure",
    "Metallicity",
    # "dark_matter_density",
    "xHI",
    "xHII",
    "xHeII",
    "xHeIII",
]
'''
epf = [
    ("particle_family", "b"),
    ("particle_tag", "b"),
    ("particle_birth_epoch", "d"),
    ("particle_metallicity", "d"),
]
             
# Ionization Parameter Field
# Based on photon densities in bins 2-4
# Don't include bin 1 -> Lyman Werner non-ionizing
def _ion_param(field, data): 
    from yt.frontends.ramses.field_handlers import RTFieldFileHandler
    p = RTFieldFileHandler.get_rt_parameters(ds).copy()
    p.update(ds.parameters)

    cgs_c = 2.99792458e10     #light velocity
    pd_2 = data['ramses-rt','Photon_density_2']*p["unit_pf"]/cgs_c #physical photon number density in cm-3
    pd_3 = data['ramses-rt','Photon_density_3']*p["unit_pf"]/cgs_c
    pd_4 = data['ramses-rt','Photon_density_4']*p["unit_pf"]/cgs_c

    photon = pd_2 + pd_3 + pd_4

    return photon/data['gas', 'number_density']  

# Luminosity field
# Cloudy Intensity obtained assuming height = 1cm
# Return intensity values erg/s/cm**2
# Multiply intensity at each pixel by volume of pixel -> luminosity
def get_luminosity(line):
   def _luminosity(field, data):
      return data['gas', 'intensity_' + line]*data['gas', 'volume']
   return copy.deepcopy(_luminosity)

yt.add_field(
    ('gas', 'ion-param'), 
    function=_ion_param, 
    sampling_type="cell", 
    units="cm**3", 
    force_override=True
)

# True divides emissions by density squared in interpolator
dens_normalized = True
if dens_normalized: 
    units = '1/cm**6'
else:
    units = '1'

# Add intensity and luminosity fields for all lines in the list
for i in range(len(lines)):
    yt.add_field(
        ('gas', 'intensity_' + lines[i]),
        function=emission.get_line_emission(i, dens_normalized),
        sampling_type='cell',
        units=units,
        force_override=True
    )
    
    yt.add_field(
        ('gas', 'luminosity_' + lines[i]),
        function=get_luminosity(lines[i]),
        sampling_type='cell',
        units='1/cm**3',
        force_override=True
    )

# Load Simulation Data
ds = yt.load(filename, extra_particle_fields=epf)
ad = ds.all_data()

sim_run = filename.split('/')[-1]

'''
Line Luminosities
'''
luminosities=[]

for line in lines:
    luminosity=ad['gas', 'luminosity_' + line].sum()
    luminosities.append(luminosity.value)

directory = 'analysis/' + sim_run + '_analysis'

if not os.path.exists(directory):
    os.makedirs(directory)

# Save the data to the new directory
file_path = os.path.join(directory, "line_luminosity.txt")
np.savetxt(file_path, luminosities, delimiter=',')

'''
Create figures
'''

# Dictionary of manual limits for each line for animating
# galaxies over time slices (fix colorbar limits for visual consistency)
lims_00273 = {
    'Ionization Parameter': [10e-7, 10e-1],
    'Number Density': [10e-2, 10e5],
    'Mass Density': [10e-26, 10e-19],
    'Temperature': [10e1, 10e6],
    'Metallicity': [10e-2, 10e1],
    "H1_6562.80A": [10e-7, 10e2],
    "O1_1304.86A": [10e-9, 10e1],
    "O1_6300.30A": [10e-8, 10e-3],
    "O2_3728.80A": [10e-7, 10e-2],
    "O2_3726.10A": [10e-7, 10e-2],
    "O3_1660.81A": [10e-8, 10e-4],
    "O3_1666.15A": [10e-8, 10e-3],
    "O3_4363.21A": [10e-9, 10e-4],
    "O3_4958.91A": [10e-8, 10e-3],
    "O3_5006.84A": [10e-8, 10e-3], 
    "He2_1640.41A": [10e-10, 10e-3],
    "C2_1335.66A": [10e-8, 10e2],
    "C3_1906.68A": [10e-7, 10e-2],
    "C3_1908.73A": [10e-7, 10e-2],
    "C4_1549.00A": [10e-16, 10e-9],
    "Mg2_2795.53A": [10e-8, 10e2],
    "Mg2_2802.71A": [10e-8, 10e2],
    "Ne3_3868.76A": [10e-9, 10e-4],
    "Ne3_3967.47A": [10e-9, 10e-5],
    "N5_1238.82A": [10e-14, 10e-4],
    "N5_1242.80A": [10e-14, 10e-4],
    "N4_1486.50A": [10e-10, 10e-4],
    "N3_1749.67A": [10e-8, 10e-4],
    "S2_6716.44A": [10e-8, 10e-2],
    "S2_6730.82A": [10e-8, 10e-3]
}

def sim_diagnostics(ds, data_file):
    star_ctr=galaxy_visualization.star_center(ad)
    ctr_den=ad.quantities.max_location(("gas", "number_density"))
    val, x_pos, y_pos, z_pos = ctr_den
    ctr = [x_pos.value, y_pos.value, z_pos.value]
    
    # For projections in a spherical region
    sp = ds.sphere(star_ctr, (3000, "pc"))
    width = 1500

    galaxy_visualization.plot_diagnostics(ds, sp, data_file, star_ctr, width)
    galaxy_visualization.plot_intensities(ds, sp, data_file, star_ctr, width)

    #galaxy_visualization.plot_diagnostics(ds, sp, data_file, star_ctr, width, lims_00273)
    #galaxy_visualization.plot_intensities(ds, sp, data_file, star_ctr, width, lims_00273)
    galaxy_visualization.spectra_driver(ds, luminosities, data_file)

sim_diagnostics(ds, sim_run)