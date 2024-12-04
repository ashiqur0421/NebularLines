'''
Braden Nowicki

Visualization routines for simulation datasets.
'''

'''
Projection, Slice Plot Routines
'''

# importing packages
#import pandas as pd
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
from matplotlib.colors import LogNorm
import sys

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


# TODO
# Locate simulation center as the center of mass
# of star particles
#def star_center(ad):


# Projection Plot Driver 
# Simplify making consistent plots of different fields
# field specified as a tuple
# width specified as a tuple wtih number and units, e.g (700, 'pc')
# Alternatively, can plot width=0.0001 - portion of box in code units
# weight_field can be specified or None for no weight field
def proj_plot(ds, sp, width, center, field, weight_field):
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
def slc_plot(ds, width, center, field):
    slc = yt.SlicePlot(
                    ds, "z", field,
                    center=center,
                    width=width,
                    buff_size=(1000, 1000))

    return slc

# yt_plot projection or slice plot, field as a string, lbox in pc
# plot_type = 'slc' or 'proj', data_file - string of input data being read
# Title String with Units
def convert_to_plt(data_file, yt_plot, plot_type, field, lbox, title):
    directory = data_file + '_analysis'

    if not os.path.exists(directory):
        os.makedirs(directory)

    fname = os.path.join(directory, str(lbox) + 'pc_' + field.replace('.', ',') + '_' + plot_type)

    plot_frb = yt_plot.frb
    p_img = np.array(plot_frb['gas', field])
    extent_dens = [-lbox/2, lbox/2, -lbox/2, lbox/2]
    dens_norm = LogNorm(np.min(p_img), np.max(p_img))
    fig = plt.figure()
    plt.imshow(p_img, norm=dens_norm, extent=extent_dens, origin='lower', aspect='auto')
    plt.xlabel("X (pc)")
    plt.ylabel("Y (pc)")
    plt.title(title)
    plt.xlim(-lbox/2, lbox/2)
    plt.ylim(-lbox/2, lbox/2)
    plt.colorbar()
    plt.savefig(fname=fname)

'''
Projection Plots of Ionization Parameter, Number Density, 
Mass Density, Temperature, Metallicity
'''

# Wrapper for making diagnostic plots of given fields
def plot_diagnostics(ds, sp, data_file, center, width):
    # Ionization Parameter
    proj_ion_param = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'ion-param'), ('gas', 'number_density'))
    #proj_ion_param.set_unit(('gas', 'ion-param'), '1')
    convert_to_plt(data_file, proj_ion_param, 'proj', 'ion-param', width, 'Ionization Parameter')
    #proj_ion_param.save(str(width) + 'pc')

    # Number Density
    proj_num_density = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'number_density'), ('gas', 'number_density'))
    #proj_num_density.save(str(width) + 'pc')
    convert_to_plt(data_file, proj_num_density, 'proj', 'number_density', width, r'Number Density [$cm^{-3}$]')

    # Mass Density
    proj_density = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'density'), ('gas', 'number_density'))
    #proj_density.save(str(width) + 'pc')
    convert_to_plt(data_file, proj_density, 'proj', 'density', width, r'Density [$g\: cm^{-3}$]')

    proj_density_wide = proj_plot(ds, sp, (1500, 'pc'), center, ('gas', 'density'), ('gas', 'number_density'))
    #proj_density_wide.save('1500pc')
    convert_to_plt(data_file, proj_density_wide, 'proj', 'density', 1500, r'Density [$g\: cm^{-3}$]')

    # Temperature
    proj_temp = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'temperature'), ('gas', 'number_density'))
    #proj_temp.save(str(width) + 'pc')
    convert_to_plt(data_file, proj_temp, 'proj', 'temperature', width, 'Temperature [K]')

    # Metallicity
    proj_metallicity = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'metallicity'), ('gas', 'number_density'))
    #proj_metallicity.save(str(width) + 'pc_')
    convert_to_plt(data_file, proj_metallicity, 'proj', 'metallicity', width, 'Metallicity')

'''
Visualizing Line Intensities
'''

# TODO expand to more lines
def plot_intensities(ds, sp, data_file, center, width):
    proj_halpha = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'intensity_H1_6562.80A'), None)
    #proj_halpha.save(str(width) + 'pc_')
    convert_to_plt(data_file, proj_halpha, 'proj', 'intensity_H1_6562.80A', width, r'Projected H1 6562.80A Intensity [$erg\: s^{-1}\: cm^{-2}$]')

    proj_halpha.set_width((2000, 'pc'))
    #proj_halpha.save('2000pc_')
    convert_to_plt(data_file, proj_halpha, 'proj', 'intensity_H1_6562.80A', 2000, r'Project H1 6562.80A Intensity [$erg\: s^{-1}\: cm^{-2}$]')

    proj_halpha_l = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'luminosity_H1_6562.80A'), None)
    #proj_halpha_l.save(str(width) + 'pc_')
    convert_to_plt(data_file, proj_halpha_l, 'proj', 'luminosity_H1_6562.80A', width, r'Projected H1 6562.80A Luminosity [$erg\: s^{-1}$]')

    proj_oiii = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'intensity_O3_5006.84A'), None)
    #proj_oiii.save(str(width) + 'pc_')
    convert_to_plt(data_file, proj_oiii, 'proj', 'intensity_O3_5006.84A', width, r'Projected O3 5006.84A Intensity [$erg\: s^{-1}\: cm^{-2}$]')

    slc_halpha = slc_plot(ds, (width, 'pc'), center, ('gas', 'intensity_H1_6562.80A'))
    #slc_halpha.save(str(width) + 'pc_')
    convert_to_plt(data_file, slc_halpha, 'slc', 'intensity_H1_6562.80A', width, r'H1 6562.80A Intensity [$erg\: s^{-1}\: cm^{-2}$]')


'''
Plot Spectra at simulated wavelengths
'''

def spectra_driver(ds, luminosities, data_file):
    z = ds.current_redshift
    omega_matter = ds.omega_matter
    omega_lambda = ds.omega_lambda
    cosmo = FlatLambdaCDM(H0=70, Om0=omega_matter)#, Om0=0.3)
    d_l = cosmo.luminosity_distance(z)*3.086e24 # convert Mpc to cm
    flux_arr = luminosities/(4*np.pi*d_l**2)#/100
    flux_arr = flux_arr.value

    # resolving power
    # R = lambda/delta_lambda
    # thermal width - 10km/s
    # delv = sqrt(avg v^2) = sqrt(kT/mH mu - molecular weight)
    # not seen - need stronger resolution
    # Milky Way bulk velocity 200km/s avg v^2 = GMdark matter halo/Rvirial radius
    # winds - gas inside galaxy
    # feedback - explosion, gas expelled from galaxy, 1000km/s
    # broad line spectra- gas orbiting BH close
    R = 1000

    directory = data_file + '_analysis'

    if not os.path.exists(directory):
        os.makedirs(directory)

    plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname=os.path.join(directory, "raw_spectra.png"), \
             sim_spectra=False, redshift_wavelengths=False)

    plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname=os.path.join(directory, "sim_spectra.png"), \
             sim_spectra=True, redshift_wavelengths=False)

    plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname=os.path.join(directory, "sim_spectra_redshifted.png"), \
             sim_spectra=True, redshift_wavelengths=True)
    

def plot_spectra(wavelengths, luminosities, flux_arr, z, noise_lvl, R, \
                 fname, sim_spectra=False, redshift_wavelengths=False):

    # Display spectra at redshifted wavelengths
    # lambda_obs = (1+z)*lambda_rest
    if redshift_wavelengths:
        wavelengths = (1+z)*np.array(wavelengths)

    # TODO calculate line widths before redshifting?
    line_widths = np.array(wavelengths)/R # Angstrom

    if sim_spectra:
        fig, ax1 = plt.subplots(1)

        #x_range, y_vals_l = plot_voigts(wavelengths, luminosities, line_widths, [0.0]*len(wavelengths), noise_lvl)
        x_range, y_vals_f = plot_voigts(wavelengths, flux_arr, line_widths, [0.0]*len(wavelengths), noise_lvl)
        ax1.plot(x_range, np.log10(y_vals_f), color='black')
        #ax2.plot(x_range, np.log10(y_vals_l))
        ax1.set_xlabel(r'Wavelength [$\AA$]')
        ax1.set_ylabel(r'Log(Flux) [$erg\: s^{-1}\: cm^{-2}$]')
        #ax2.set_ylabel(r'Log(Luminosity) [$erg s^{-1}$]')
    else:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)

        ax1.plot(wavelengths, np.log10(flux_arr), 'o')
        ax2.plot(wavelengths, np.log10(luminosities), 'o')
        ax2.set_xlabel(r'Wavelength [$\AA$]')
        ax1.set_ylabel(r'Log(Flux) [$erg\: s^{-1}\: cm^{-2}$]')
        ax2.set_ylabel(r'Log(Luminosity) [$erg\: s^{-1}$]')

    plt.savefig(fname)

# Plot voigt profiles for spectral lines over a noise level
# sigma - stdev of normal dist
# gamma - FWHM of cauchy dist
def plot_voigts(centers, amplitudes, sigmas, gammas, noise_lvl):

    x_range = np.linspace(min(centers) - 20, max(centers) + 20, 1000)
    y_vals = np.zeros_like(x_range)+noise_lvl

    for amp, center, sigma, gamma in zip(amplitudes, centers, sigmas, gammas):
        if amp > noise_lvl:
            y_vals += (amp-noise_lvl)*voigt_profile(x_range - center, sigma, gamma)

    return x_range, y_vals

# TODO mult metallicity by 4?

#plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname='raw_spectra.png', \
#             sim_spectra=False, redshift_wavelengths=False)

#plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname='sim_spectra.png', \
#             sim_spectra=True, redshift_wavelengths=False)

#plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname='sim_spectra_redshifted.png', \
#             sim_spectra=True, redshift_wavelengths=True)

#plt.show()

# TODO - intrinsic width of line - temperature, bulk velocity of grid points, sum voigts
# assuming unresolved - width doesnt matter - R = 1000
# resolve actual lines - each line contributed mostly by certain cells
# where most of signal coming from - order cells by luminosity
# full calculation for brightest
# or sum cells within temp range - treat all as having same width. bulk velocity problem
