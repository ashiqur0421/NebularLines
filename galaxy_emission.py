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
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import emission
import astropy
import yt
from yt.units import dimensions
import copy
from scipy.special import voigt_profile
from astropy.cosmology import FlatLambdaCDM

# TODO change to take input file(s) as input from cmd line or text file
f1 = "/Users/bnowicki/Documents/Research/Ricotti/output_00273"

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
        #('gas', 'intensity_' + lines[i] + '_[erg_cm^{-2}_s^{-1}]'),
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

ds = yt.load(f1)
ad = ds.all_data()

# For projections in a spherical region
sp = ds.sphere([0.49118094, 0.49275361, 0.49473726], (2000, "pc"))

'''
Projection Plots of Ionization Parameter, Number Density, 
Mass Density, Temperature, Metallicity
'''

# center derived from max density pixel
# Alternatively, use
# center=("max", ("gas", "[field]")),
# in projection plot
center_max=[0.49118094, 0.49275361, 0.49473726]

# Projection Plot Driver 
# Simplify making consistent plots of different fields
# field specified as a tuple
# width specified as a tuple wtih number and units, e.g (700, 'pc')
# Alternatively, can plot width=0.0001 - portion of box in code units
# weight_field can be specified or None for no weight field
def proj_plot(width, center, field, weight_field):
    if weight_field == None:
        p = yt.ProjectionPlot(ds, "z", field,
                      width=width,
                      data_source=sp,
                      buff_size=(1000, 1000),
                      center=center)
    else:
        p = yt.ProjectionPlot(ds, "z", field,
                      width=width,
                      weight_field=weight_field,
                      data_source=sp,
                      buff_size=(1000, 1000),
                      center=center)
    return p

# Slice Plot Driver
def slc_plot(width, center, field):
    slc = yt.SlicePlot(
                    ds, "z", field,
                    center=center,
                    width=width,
                    buff_size=(1000, 1000))

    return slc

# Wrapper for making diagnostic plots of given fields
def plot_diagnostics():
    # Ionization Parameter
    proj_ion_param = proj_plot((400, 'pc'), center_max, ('gas', 'ion-param'), ('gas', 'number_density'))
    #proj_ion_param.set_unit(('gas', 'ion-param'), '1')
    proj_ion_param.save('400pc_')

    # Number Density
    proj_num_density = proj_plot((400, 'pc'), center_max, ('gas', 'number_density'), ('gas', 'number_density'))
    proj_num_density.save('400pc')

    # Mass Density
    proj_density = proj_plot((400, 'pc'), center_max, ('gas', 'density'), ('gas', 'number_density'))
    proj_density.save('400pc_')

    proj_density_wide = proj_plot((1500, 'pc'), center_max, ('gas', 'density'), ('gas', 'number_density'))
    proj_density_wide.save('1500pc_')

    # Temperature
    proj_temp = proj_plot((400, 'pc'), center_max, ('gas', 'temperature'), ('gas', 'number_density'))
    proj_temp.save('400pc_')

    # Metallicity
    proj_metallicity = proj_plot((400, 'pc'), center_max, ('gas', 'metallicity'), ('gas', 'number_density'))
    proj_metallicity.save('400pc_')

#plot_diagnostics()

'''
Visualizing Line Intensities
'''

proj_halpha = proj_plot((400, 'pc'), center_max, ('gas', 'intensity_H1_6562.80A'), None)
proj_halpha.save('400pc_')
proj_halpha.set_width((2, 'kpc'))
proj_halpha.save('2000pc_')

proj_halpha_l = proj_plot((400, 'pc'), center_max, ('gas', 'luminosity_H1_6562.80A'), None)
proj_halpha_l.save('400pc_')

proj_oiii = proj_plot((400, 'pc'), center_max, ('gas', 'intensity_O3_5006.84A'), None)
proj_oiii.save('400pc_')

slc_halpha = slc_plot((400, 'pc'), center_max, ('gas', 'intensity_H1_6562.80A'))
slc_halpha.save('400pc_')


'''
Plot Spectra at simulated wavelengths
'''
# TODO flags to do normal version (plotting points), beautified
# (voigt functions summed over noise_level), wavelengths as is or redshifted
# flux or luminosity

luminosities=[]

for line in lines:
    luminosity=ad['gas', 'luminosity_' + line].sum()
    luminosities.append(luminosity.value)

# TODO use sim redshift
#z = 6.145
z = ds.current_redshift
omega_matter = ds.omega_matter
omega_lambda = ds.omega_lambda
cosmo = FlatLambdaCDM(H0=70, Om0=omega_matter)#, Om0=0.3)
d_l = cosmo.luminosity_distance(z)*3.086e24 # convert Mpc to cm
flux_arr = luminosities/(4*np.pi*d_l**2)#/100

# resolving power
# R = lambda/delta_lambda
R = 1000
line_widths = wavelengths/R # Angstrom

def plot_spectra(wavelengths, luminosities, flux_arr, z, noise_lvl, line_widths, \
                 sim_spectra=False, redshift_wavelengths=False):

    (ax1, ax2), fig = plt.subplots(2, sharex=True)

    # Display spectra at redshifted wavelengths
    # lambda_obs = (1+z)*lambda_rest
    if redshift_wavelengths:
        wavelengths = (1+z)*wavelengths

    if sim_spectra:
        x_range, y_vals_l = plot_voigts(wavelengths, luminosities, line_widths, 0.0, noise_lvl)
        x_range, y_vals_f = plot_voigts(wavelengths, flux_arr, line_widths, 0.0, noise_lvl)
        ax1.plot(x_range, np.log10(y_vals_l.value), 'o')
        ax2.plot(x_range, np.log10(y_vals_f), 'o')
        ax2.set_xlabel(r'Wavelength [$\AA$]')
        ax1.set_ylabel(r'Log(Flux) [$erg s^{-1] cm^{-2}$]')
        ax2.set_ylabel(r'Log(Luminosity) [$erg s^{-1}$]')
    else:
        ax1.plot(wavelengths, np.log10(flux_arr.value), 'o')
        ax2.plot(wavelengths, np.log10(luminosities), 'o')
        ax2.set_xlabel(r'Wavelength [$\AA$]')
        ax1.set_ylabel(r'Log(Flux) [$erg s^{-1] cm^{-2}$]')
        ax2.set_ylabel(r'Log(Luminosity) [$erg s^{-1}$]')

# Plot voigt profiles for spectral lines over a noise level
# sigma - stdev of normal dist
# gamma - FWHM of cauchy dist
def plot_voigts(centers, amplitudes, sigmas, gammas, noise_lvl):

    x_range = np.linspace(min(centers) - 5, max(centers) + 5, 1000)
    y_vals = np.zeros_like(x_range)+noise_lvl

    for amp, center, sigma, gamma in zip(amplitudes, centers, sigmas, gammas):
        if amp > noise_lvl:
            y_vals += (amp-noise_lvl)*voigt_profile(x_range - center, sigma, gamma)

    return x_range, y_vals

# TODO mult metallicity by 4?

plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, sim_spectra=False, \
             redshift_wavelengths=False)

plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, sim_spectra=True, \
             redshift_wavelengths=False)

plt.show()