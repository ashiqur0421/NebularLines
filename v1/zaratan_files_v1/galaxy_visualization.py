'''
Braden Nowicki

Visualization routines for simulation datasets.
'''

'''
Projection, Slice Plot Routines
'''

# importing packages
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
from scipy.ndimage import gaussian_filter

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


# Find center of mass of star particles
def star_center(ad):
    x_pos = np.array(ad["star", "particle_position_x"])
    y_pos = np.array(ad["star", "particle_position_y"])
    z_pos = np.array(ad["star", "particle_position_z"])
    x_center = np.mean(x_pos)
    y_center = np.mean(y_pos)
    z_center = np.mean(z_pos)
    x_pos = x_pos - x_center
    y_pos = y_pos - y_center
    z_pos = z_pos - z_center
    ctr_at_code = np.array([x_center, y_center, z_center])
    return ctr_at_code

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
    directory = 'analysis/' + data_file + '_analysis'

    if not os.path.exists(directory):
        os.makedirs(directory)

    fname = os.path.join(directory, str(lbox) + 'pc_' + field.replace('.', ',') + '_' + plot_type)

    plot_frb = yt_plot.frb
    p_img = np.array(plot_frb['gas', field])

    if np.min(p_img) <= 0:
        print("Warning: Data contains non-positive values. Adjusting for LogNorm.")
        p_img = np.clip(p_img, a_min=1e-10, a_max=None)  # Clip values below 1e-10 to avoid log of zero

    extent_dens = [-lbox/2, lbox/2, -lbox/2, lbox/2]
    dens_norm = LogNorm(np.min(p_img), np.max(p_img))
    fig = plt.figure()
    # TODO fix imshow issues
    im = plt.imshow(p_img, norm=dens_norm, extent=extent_dens, origin='lower', aspect='auto', interpolation='nearest')
    plt.xlabel("X (pc)")
    plt.ylabel("Y (pc)")
    plt.title(title)
    plt.xlim(-lbox/2, lbox/2)
    plt.ylim(-lbox/2, lbox/2)
    plt.colorbar(im)
    plt.savefig(fname=fname)
    plt.close()

def convert_to_plt_2(data_file, yt_plot, plot_type, field, lbox, title, lims=None):
    '''
    lims = [vmin, vmax] fixed limits on colorbar values for image if desired; else None
    '''

    directory = 'analysis/' + data_file + '_analysis'

    if not os.path.exists(directory):
        os.makedirs(directory)

    fname = os.path.join(directory, data_file + '_' + str(lbox) + 'pc_' + field.replace('.', ',') + '_' + plot_type)
    if lims != None:
        fname = fname + '_lims'

    plot_frb = yt_plot.frb
    p_img = np.array(plot_frb['gas', field])

    # Clip non-positive values to avoid log of zero or negative numbers
    if np.min(p_img) <= 0:
        print("Warning: Data contains non-positive values. Adjusting for LogNorm.")
        p_img = np.clip(p_img, a_min=1e-10, a_max=None)  # Clip values below 1e-10

    # Optionally smooth the data to reduce checkerboard patterns
    #p_img = gaussian_filter(p_img, sigma=1)

    # Set the extent of the plot (this assumes 'lbox' is the length of the box in parsecs)
    extent_dens = [-lbox / 2, lbox / 2, -lbox / 2, lbox / 2]

    # Define the color normalization based on the range of the data
    if lims == None:
        dens_norm = LogNorm(vmin=np.min(p_img), vmax=np.max(p_img))
    else:
        # Set fixed color normalization limits (vmin and vmax for the colorbar)
        dens_norm = LogNorm(vmin=lims[0], vmax=lims[1])

    # Create the figure
    fig = plt.figure(figsize=(8, 6))
    
    # Create the image plot
    im = plt.imshow(p_img, norm=dens_norm, extent=extent_dens, origin='lower', 
                    aspect='auto', interpolation='bilinear', cmap='viridis')
    
    #cmap viridis, inferno, magma
    
    # Add labels and title
    plt.xlabel("X (pc)", fontsize=12)
    plt.ylabel("Y (pc)", fontsize=12)
    plt.title(title, fontsize=14)
    
    # Set axis limits
    plt.xlim(-lbox / 2, lbox / 2)
    plt.ylim(-lbox / 2, lbox / 2)

    # Add color bar
    cbar = plt.colorbar(im)

    #if lims != None:
        # Set ticks at the colorbar limits
        #cbar.set_ticks([lims[0], lims[1]])  
    #cbar.set_label(field, fontsize=12)

    # Save the figure
    plt.savefig(fname, dpi=300)
    plt.close()

'''
Projection Plots of Ionization Parameter, Number Density, 
Mass Density, Temperature, Metallicity
'''

# TODO update convert to plt to do both versions of plots with and without limits

# Wrapper for making diagnostic plots of given fields
def plot_diagnostics(ds, sp, data_file, center, width, lims_dict=None):
    # Ionization Parameter
    proj_ion_param = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'ion-param'), ('gas', 'number_density'))
    #proj_ion_param.set_unit(('gas', 'ion-param'), '1')
    if lims_dict == None:
        convert_to_plt_2(data_file, proj_ion_param, 'proj', 'ion-param', width, 'Ionization Parameter')
    else:
        convert_to_plt_2(data_file, proj_ion_param, 'proj', 'ion-param', width, 'Ionization Parameter', lims_dict['Ionization Parameter'])
    #proj_ion_param.save(str(width) + 'pc')

    # Number Density
    proj_num_density = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'number_density'), ('gas', 'number_density'))
    #proj_num_density.save(str(width) + 'pc')
    if lims_dict == None:
        convert_to_plt_2(data_file, proj_num_density, 'proj', 'number_density', width, r'Number Density [$cm^{-3}$]')
    else:
        convert_to_plt_2(data_file, proj_num_density, 'proj', 'number_density', width, r'Number Density [$cm^{-3}$]', lims_dict['Number Density'])

    # Mass Density
    proj_density = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'density'), ('gas', 'number_density'))
    #proj_density.save(str(width) + 'pc')
    if lims_dict == None: 
        convert_to_plt_2(data_file, proj_density, 'proj', 'density', width, r'Density [$g\: cm^{-3}$]')
    else:
        convert_to_plt_2(data_file, proj_density, 'proj', 'density', width, r'Density [$g\: cm^{-3}$]', lims_dict['Mass Density'])

    #proj_density_wide = proj_plot(ds, sp, (1500, 'pc'), center, ('gas', 'density'), ('gas', 'number_density'))
    #proj_density_wide.save('1500pc')
    #if lims_dict == None:
    #    convert_to_plt_2(data_file, proj_density_wide, 'proj', 'density', 1500, r'Density [$g\: cm^{-3}$]')
    #else:
    #    convert_to_plt_2(data_file, proj_density_wide, 'proj', 'density', 1500, r'Density [$g\: cm^{-3}$]')

    # Temperature
    proj_temp = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'temperature'), ('gas', 'number_density'))
    #proj_temp.save(str(width) + 'pc')
    if lims_dict == None:
        convert_to_plt_2(data_file, proj_temp, 'proj', 'temperature', width, 'Temperature [K]')
    else:
        convert_to_plt_2(data_file, proj_temp, 'proj', 'temperature', width, 'Temperature [K]', lims_dict['Temperature'])

    # Metallicity
    proj_metallicity = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'metallicity'), ('gas', 'number_density'))
    #proj_metallicity.save(str(width) + 'pc_')
    if lims_dict == None:
        convert_to_plt_2(data_file, proj_metallicity, 'proj', 'metallicity', width, 'Metallicity')
    else:
        convert_to_plt_2(data_file, proj_metallicity, 'proj', 'metallicity', width, 'Metallicity', lims_dict['Metallicity'])

'''
Visualizing Line Intensities
'''

def plot_intensities(ds, sp, data_file, center, width, lims_dict=None):
    for line in lines:
        proj = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'intensity_' + line), None)

        if lims_dict == None:
            convert_to_plt_2(data_file, proj, 'proj', 'intensity_' + line, width, \
                        'Projected ' + line.replace('_', ' ') + r' Intensity [$erg\: s^{-1}\: cm^{-2}$]')
        else:
            convert_to_plt_2(data_file, proj, 'proj', 'intensity_' + line, width, \
                        'Projected ' + line.replace('_', ' ') + r' Intensity [$erg\: s^{-1}\: cm^{-2}$]', lims_dict[line])

    #proj_halpha = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'intensity_H1_6562.80A'), None)
    #convert_to_plt_2(data_file, proj_halpha, 'proj', 'intensity_H1_6562.80A', width, r'Projected H1 6562.80A Intensity [$erg\: s^{-1}\: cm^{-2}$]')

    #proj_halpha.set_width((2000, 'pc'))
    #convert_to_plt_2(data_file, proj_halpha, 'proj', 'intensity_H1_6562.80A', 2000, r'Projected H1 6562.80A Intensity [$erg\: s^{-1}\: cm^{-2}$]')

    proj_halpha_l = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'luminosity_H1_6562.80A'), None)
    convert_to_plt_2(data_file, proj_halpha_l, 'proj', 'luminosity_H1_6562.80A', width, r'Projected H1 6562.80A Luminosity [$erg\: s^{-1}$]')

    slc_halpha = slc_plot(ds, (width, 'pc'), center, ('gas', 'intensity_H1_6562.80A'))
    convert_to_plt_2(data_file, slc_halpha, 'slc', 'intensity_H1_6562.80A', width, r'H1 6562.80A Intensity [$erg\: s^{-1}\: cm^{-2}$]')

'''
Plot Spectra at simulated wavelengths
'''

def spectra_driver(ds, luminosities, data_file):
    z = ds.current_redshift
    omega_matter = ds.omega_matter
    omega_lambda = ds.omega_lambda
    cosmo = FlatLambdaCDM(H0=70, Om0=omega_matter)#, Om0=0.3)
    d_l = cosmo.luminosity_distance(z)*3.086e24 # convert Mpc to cm
    flux_arr = luminosities/(4*np.pi*d_l**2)
    flux_arr = flux_arr.value

    # resolving power
    # R = lambda/delta_lambda
    R = 1000

    directory = 'analysis/' + data_file + '_analysis'

    if not os.path.exists(directory):
        os.makedirs(directory)

    plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname=os.path.join(directory, data_file + "_raw_spectra"), \
             sim_spectra=False, redshift_wavelengths=False)

    plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname=os.path.join(directory, data_file + "_sim_spectra"), \
             sim_spectra=True, redshift_wavelengths=False)

    plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname=os.path.join(directory, data_file + "_sim_spectra_redshifted"), \
             sim_spectra=True, redshift_wavelengths=True)
    

def plot_spectra(wavelengths, luminosities, flux_arr, z, noise_lvl, R, \
                 fname, sim_spectra=False, redshift_wavelengths=False):

    pad = 1000

    # Display spectra at redshifted wavelengths
    # lambda_obs = (1+z)*lambda_rest
    if redshift_wavelengths:
        wavelengths = (1+z)*np.array(wavelengths)
        pad = 5000

    line_widths = np.array(wavelengths)/R # Angstrom

    if sim_spectra:
        fig, ax1 = plt.subplots(1)
        x_range, y_vals_f = plot_voigts(wavelengths, flux_arr, line_widths, [0.0]*len(wavelengths), noise_lvl, pad)
        ax1.plot(x_range, np.log10(y_vals_f), color='black')
        ax1.set_xlabel(r'Wavelength [$\AA$]')
        ax1.set_ylabel(r'Log(Flux) [$erg\: s^{-1}\: cm^{-2} \: \AA]$')
        plt.savefig(fname)
        plt.close()

        fig, ax1 = plt.subplots(1)
        x_range, y_vals_l = plot_voigts(wavelengths, luminosities, line_widths, [0.0]*len(wavelengths), noise_lvl, pad)
        ax1.plot(x_range, np.log10(y_vals_l), color='black')
        ax1.set_ylim([32, 44])
        ax1.set_xlabel(r'Wavelength [$\AA$]')
        ax1.set_ylabel(r'Log(Luminosity) [$erg\: s^{-1}$]')
        lum_fname = fname + '_lum'
        plt.savefig(lum_fname)
        plt.close()
    else:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(wavelengths, np.log10(flux_arr), 'o')
        ax2.plot(wavelengths, np.log10(luminosities), 'o')
        ax2.set_xlabel(r'Wavelength [$\AA$]')
        ax1.set_ylabel(r'Log(Flux) [$erg\: s^{-1}\: cm^{-2} \: \AA]$')
        ax2.set_ylabel(r'Log(Luminosity) [$erg\: s^{-1}$]')
        plt.savefig(fname)
        plt.close()

# Plot voigt profiles for spectral lines over a noise level
# sigma - stdev of normal dist
# gamma - FWHM of cauchy dist
# TODO noiseless profile, Poisson Noise
def plot_voigts(centers, amplitudes, sigmas, gammas, noise_lvl, pad):

    x_range = np.linspace(min(centers) - pad, max(centers) + pad, 1000)
    y_vals = np.zeros_like(x_range)+noise_lvl

    for amp, center, sigma, gamma in zip(amplitudes, centers, sigmas, gammas):
        y_vals += (amp)*voigt_profile(x_range - center, sigma, gamma)
        #if amp > noise_lvl:
            #y_vals += (amp-noise_lvl)*voigt_profile(x_range - center, sigma, gamma) # - noise after no sub

    #y_vals += noise_lvl

    return x_range, y_vals
