# importing packages
import numpy as np
import shutil
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

'''
galaxy_visualization.py

Author: Braden Nowicki

Visualization and analysis routines for RAMSES-RT Simulations.

'''

'''
Projection, Slice Plot Routines
'''

# TODO docstrings

# Cloudy Grid Run Bounds (log values)
# Umin, Umax, Ustep: -6.0 1.0 0.5
# Nmin, Nmax, Nstep: -1.0 6.0 0.5 
# Tmin, Tmax, Tstop: 3.0 6.0 0.1

'''
lines=["H1_6562.80A","O1_1304.86A","O1_6300.30A","O2_3728.80A","O2_3726.10A",
       "O3_1660.81A","O3_1666.15A","O3_4363.21A","O3_4958.91A","O3_5006.84A", 
       "He2_1640.41A","C2_1335.66A","C3_1906.68A","C3_1908.73A","C4_1549.00A",
       "Mg2_2795.53A","Mg2_2802.71A","Ne3_3868.76A","Ne3_3967.47A","N5_1238.82A",
       "N5_1242.80A","N4_1486.50A","N3_1749.67A","S2_6716.44A","S2_6730.82A"]

wavelengths=[6562.80, 1304.86, 6300.30, 3728.80, 3726.10, 
             1660.81, 1666.15, 4363.21, 4958.91, 5006.84, 
             1640.41, 1335.66, 1906.68, 1908.73, 1549.00, 
             2795.53, 2802.71, 3868.76, 3967.47, 1238.82, 
             1242.80, 1486.50, 1749.67, 6716.44, 6730.82]
'''

class VisualizationManager:

    def __init__(self, filename, lines, wavelengths):
        '''
        
        Parameters:
        filename (str): filepath to the RAMSES-RT output_*/info_*.txt file
        lines (List, strings): List of nebular emission lines
        wavelengths (List, floats): List of corresponding wavelengths

        file_dir (str): filepath to output directory
        output_file (str): output folder, e.g. output_00273
        sim_run (str): Time slice number for simulation eg. 00273
        info_file (str): Filename with info file appended '/info_00273.txt'
        directory (str): analysis output directory
        '''

        
        self.filename = filename
        self.file_dir = os.path.dirname(self.filename)
        self.lines = lines
        self.wavelengths = wavelengths
        self.output_file = self.file_dir.split('/')[-1]
        self.sim_run = self.output_file.split('_')[1]
        #self.info_file = f'{self.file_dir}/info_{self.sim_run}.txt'

        # Analysis directory for saving
        self.directory = f'analysis/{self.output_file}_analysis'

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        print(f'Filename = {self.filename}')
        print(f'File Directory = {self.file_dir}')
        print(f'Output File = {self.output_file}')
        print(f'Simulation Run = {self.sim_run}')
        print(f'Analysis Directory = {self.directory}')

    
    # Find center of mass of star particles
    # TODO ds and ad as attributes of class?
    def star_center(self, ad):
        '''
        Locate the center of mass of star particles in code units.

        Parameters:
        ad: data object from RAMSES-RT output loaded into yt-project

        Returns:
        ctr_at_code (List, float): Coordinates (code units) of center of mass

        '''

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


    def proj_plot(self, ds, sp, width, center, field, weight_field):
        '''
        Projection Plot Driver.

        Parameters:
        ds: loaded RAMSES-RT data set
        sp: sphere data object to project within
        width (tuple, int and str): width in code units or formatted with 
            units, e.g. (1500, 'pc')
        center (List, float): center (array of 3 values) in code units
        field (tuple, str): field to project, e.g. ('gas', 'temperature')
        weight_field (tuple, str): field to weight project (or None if 
            unweighted)

        Returns:
        Projection Plot Object
        '''

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


    def slc_plot(self, ds, width, center, field):
        '''
        Slice Plot Driver.

        Parameters:
        ds: load RAMSES-RT data set
        width (tuple, int and str): width in code units or formatted with 
            units, e.g. (1500, 'pc')
        center (List, float): center (array of 3 values) in code units
        field (tuple, str): field to project, e.g. ('gas', 'temperature')
        
        Returns:
        Slice Plot Object
        '''

        slc = yt.SlicePlot(
                        ds, "z", field,
                        center=center,
                        width=width,
                        buff_size=(1000, 1000))

        return slc
    

    # TODO change redshift to self
    def convert_to_plt(self, yt_plot, plot_type, field, width, 
                       redshift, title, lims=None):
        '''
        Convert a yt projection or slice plot to matplotlib.

        Parameters:
        yt_plot: Projection or Slice Plot Object
        plot_type (str): Type of plot (for filename) - 'proj' or 'slc'
        field (tuple, str): field to plot, e.g. ('gas', 'temperature')
        width (tuple, int and str): width in code units or formatted with 
            units, e.g. (1500, 'pc')
        redshift (float): redshift of current time slice
        title (str): Plot title
        lims (None or List): [vmin, vmax] fixed limits on colorbar values
            for image if desired; otherwise None

        Returns:
        NA

        Saves desired figures with usable file naming scheme.
        '''

        lbox = width[0]
        length_unit = width[1]
        field_comma = field[1].replace('.', ',')

        plot_title = f'{self.output_file}_{lbox}{length_unit}_' + \
            f'{field_comma}_{plot_type}'

        fname = os.path.join(self.directory, plot_title)
        if lims != None:
            fname = fname + '_lims'

        plot_frb = yt_plot.frb
        # TODO check below
        #p_img = np.array(plot_frb['gas', field])
        p_img = np.array(plot_frb[field[0], field[1]])

        # Clip non-positive values to avoid log of zero or negative numbers
        if np.min(p_img) <= 0:
            print('Warning: Data contains non-positive values. Adjusting ' +
                  'for LogNorm.')
            
            # Clip values below 1e-10
            p_img = np.clip(p_img, a_min=1e-10, a_max=None)

        # Replace NaN with 0 and Inf with finite numbers
        if np.any(np.isnan(p_img)) or np.any(np.isinf(p_img)):
            print('Warning: Data contains NaN or Inf values. ' +
                  'Replacing with 0.')
            p_img = np.nan_to_num(p_img)

        # TODO
        #p_img = gaussian_filter(p_img, sigma=1)

        # Set the extent of the plot
        extent_dens = [-lbox / 2, lbox / 2, -lbox / 2, lbox / 2]

        # Define the color normalization based on the range of the data
        if lims == None:
            dens_norm = LogNorm(vmin=np.min(p_img), vmax=np.max(p_img))
        else:
            # Set fixed color normalization limits
            dens_norm = LogNorm(vmin=lims[0], vmax=lims[1])


        # Viridis, Inferno, Magma maps work - perceptually uniform
        fig = plt.figure(figsize=(8, 6))
        #im = plt.imshow(p_img, norm=dens_norm, extent=extent_dens, 
        #                origin='lower', aspect='auto', 
        #                interpolation='bilinear', cmap='viridis')

        im = plt.imshow(p_img, norm=dens_norm, extent=extent_dens, 
                        origin='lower', aspect='equal', 
                        interpolation='nearest', cmap='viridis')

        plt.xlabel(f'X [{length_unit}]', fontsize=12)
        plt.ylabel(f'Y [{length_unit}]', fontsize=12)
        plt.title(title, fontsize=14)

        plt.xlim(-lbox / 2, lbox / 2)
        plt.ylim(-lbox / 2, lbox / 2)

        cbar = plt.colorbar(im)

        # Add redshift
        plt.text(0.05, 0.05, f'z = {redshift:.5f}', color='white', fontsize=9,
                 ha='left', va='bottom', transform=plt.gca().transAxes)
        # TODO font

        # TODO print

        # Save the figure
        plt.savefig(fname, dpi=300)
        plt.close()


    def plot_wrapper(self, ds, sp, width, center, field_list,
                     weight_field_list, title_list, proj=True, slc=True,
                     lims_dict=None, lims_titles=None):
        '''
        Wrapper for plotting a variety of fields simultaneously.

        Parameters:
        ds: loaded RAMSES-RT data set
        sp: sphere data object to project within
        center (List, float): center (array of 3 values) in code units
        width (tuple, int and str): width in code units or formatted with 
            units, e.g. (1500, 'pc')
        field_list (List of tuple, str): list of fields to plot, e.g. 
            ('gas', 'temperature')
        weight_field_list (List of tuple, str): list of fields to weight 
            projections (or None if unweighted)
        title_list (List of str): list of titles associated with plots
        lims_dict (None or Dict): dictionary of [vmin, vmax] fixed limits on
            colorbar values for image if desired; otherwise None
        lims_titles (List, str): titles associated with lims_dict for
            corresponding fields
        '''

        redshift = ds.current_redshift

        for i, field in enumerate(field_list):
            if proj:
                p = self.proj_plot(ds, sp, width, center, field, 
                                   weight_field_list[i])
                
                if lims_dict == None:
                    self.convert_to_plt(p, 'proj', field, width, redshift,
                                        'Projected ' + title_list[i])
                else:
                    self.convert_to_plt(p, 'proj', field, width, redshift,
                                        'Projected ' + title_list[i], 
                                        lims_dict[lims_titles[i]])

            if slc:
                p = self.slc_plot(ds, width, center, field)
                
                if lims_dict == None:
                    self.convert_to_plt(p, 'slc', field, width, redshift,
                                        title_list[i])
                else:
                    self.convert_to_plt(p, 'slc', field, width, redshift,
                                        title_list[i], 
                                        lims_dict[lims_titles[i]])
                    

    def phase_plot(self, ds, sp, x_field, y_field):
        extrema = {("gas", "number_density"): (1e-4, 1e4), ("gas", "temperature"): (1e3, 1e8)}

        profile = yt.create_profile(
            sp,
            [("gas", "temperature"), ("gas", "number_density")],
            #n_bins=[128, 128],
            fields=[("gas", "flux_H1_6562.80A")],
            weight_field=None,
            #units=units,
            extrema=extrema,
        )

        plot = yt.PhasePlot.from_profile(profile)

        plot_frb = yt_plot.frb
        # TODO check below
        #p_img = np.array(plot_frb['gas', field])
        p_img = np.array(plot_frb[field[0], field[1]])
        


    
    def save_array_with_headers(self, filename, array, headers, delimiter=','):
        '''
        Saves a NumPy array to a text file, including column headers.

        Parameters:
        filename (str): The name of the file to save to.
        array (np.ndarray): The NumPy array to save.
        headers (List): A list of strings representing the column headers.
        delimiter (str, optional): The delimiter to use between values. 
            Defaults to ','.
        '''
        with open(filename, 'w') as file:
            file.write(delimiter.join(headers) + '\n')
            np.savetxt(file, array, delimiter=delimiter, fmt='%s')


    def calc_luminosities(self, sp):
        '''
        Agreggate luminosities for each emission line in sphere sp.

        Parameters:
        sp: Data sphere.

        Returns:
        luminosities (List, float): array of luminosities for corresponding
            emission lines.
        '''

        lum_file_path = os.path.join(self.directory, 
                                     f'{self.output_file}_line_luminosity.txt')

        luminosities = []

        for line in self.lines:
            luminosity=sp.quantities.total_quantity(
                ('gas', 'luminosity_' + line)
            )
            luminosities.append(luminosity.value)
            print(f'{line} Luminosity = {luminosity} erg/s')

        # TODO
        #emission_line_str = ", ".join(self.lines)
        #np.savetxt(lum_file_path, luminosities, delimeter=',', 
        #           header=emission_line_str)

        self.save_array_with_headers(lum_file_path, luminosities, self.lines)
        
        self.luminosities = luminosities

        return luminosities
    

    def save_sim_info(self, ds):
        '''
        Save simulation parameters/information.

        Parameters:
        ds: RAMSES data loaded into yt.
        '''

        self.current_time = ds.current_time
        self.domain_dimensions = ds.domain_dimensions
        self.domain_left_edge = ds.domain_left_edge
        self.domain_right_edge = ds.domain_right_edge
        self.cosmological_simulation = ds.cosmological_simulation
        self.current_redshift = ds.current_redshift
        self.omega_lambda = ds.omega_lambda
        self.omega_matter = ds.omega_matter
        self.omega_radiation = ds.omega_radiation
        self.hubble_constant = ds.hubble_constant


        file_path = os.path.join(self.directory, 
                                f'{self.output_file}_sim_info.txt')
        
        with open(file_path, 'w') as file:
            file.write(f'current_time: {self.current_time}\n')
            file.write(f'domain_dimensions: {self.domain_dimensions}\n')
            file.write(f'domain_left_edge: {self.domain_left_edge}\n')
            file.write(f'domain_right_edge: {self.domain_right_edge}\n')
            file.write(f'cosmological_simulation: ' +
                       f'{self.cosmological_simulation}\n')
            file.write(f'current_redshift: {self.current_redshift}\n')
            file.write(f'omega_lambda: {self.omega_lambda}\n')
            file.write(f'omega_matter: {self.omega_matter}\n')
            file.write(f'omega_radiation: {self.omega_radiation}\n')
            file.write(f'hubble_constant: {self.hubble_constant}\n')


        # TODO
        '''
        column_headers = ['current_time', 'domain_dimensions',
                          'domain_left_edge', 'domain_right_edge',
                          'cosmological_simulation', 'current_redshift',
                          'omega_lambda', 'omega_matter',
                          'omega_radiation', 'hubble_constant']
        
        info_arr = [self.current_time, self.domain_dimensions,
                    self.domain_left_edge, self.domain_right_edge,
                    self.cosmological_simulation, self.current_redshift,
                    self.omega_lambda, self.omega_matter,
                    self.omega_radiation, self.hubble_constant]

        file_path = os.path.join(self.directory, 
                                f'{self.output_file}_sim_info.txt')

        self.save_array_with_headers(file_path, info_arr, self.lines)
        '''

        # Copy information files from data folder to analysis
        sim_info_files = [
            os.path.join(self.file_dir, f'header_{self.sim_run}.txt'),
            os.path.join(self.file_dir, 'hydro_file_descriptor.txt'),
            os.path.join(self.file_dir, f'info_{self.sim_run}.txt'),
            os.path.join(self.file_dir, f'info_rt_{self.sim_run}.txt'),
            os.path.join(self.file_dir, 'namelist.txt')
        ]

        for sim_info_file in sim_info_files:
            shutil.copy2(sim_info_file, self.directory)


    def save_sim_field_info(self, ds, sp):
        '''
        Save min, max, mean, and aggregate of each field in fields array.

        Parameters:
        ds: RAMSES data loaded into yt.
        TODO ad
        '''

        fields = [
            ('gas', 'temperature'),
            ('gas', 'density'),
            #('gas', 'number_density'),
            ('gas', 'my_H_nuclei_density'),
            ('gas', 'my_temperature'),
            ('gas', 'ion_param'),
            ('gas', 'metallicity')
        ]

        for line in self.lines:
            fields.append(('gas', 'flux_'  + line))
            fields.append(('gas', 'luminosity_'  + line))

        # Calculate desired quantities for each field
        field_info = []

        for field in fields:
            min = sp.min(field)
            max = sp.max(field)
            mean = sp.mean(field)
            agg = sp.quantities.total_quantity(field)

            field_info.append((min, max, mean, agg))

        # Save data to a file
        file_path = os.path.join(self.directory, 
                                f'{self.output_file}_field_info.txt')
        
        with open(file_path, 'w') as file:
            for i, field in enumerate(fields):
                file.write(f'{field}_min: {fields[i][0]}\n')
                file.write(f'{field}_max: {fields[i][1]}\n')
                file.write(f'{field}_mean: {fields[i][2]}\n')
                file.write(f'{field}_agg: {fields[i][3]}\n')

        '''
        Reading the data file example:

        Regex Pattern for float: r'[-+]?\d*\.\d+([eE][-+]?\d+)?'
        Scientific Notation Possible

        import re

        with open('data.txt', 'r') as file:
            file_content = file.read()

        temp_min_pattern = fr'{field}_min: [-+]?\d*\.\d+([eE][-+]?\d+)?'

        temp_min = float(re.search(temp_min_pattern, file_content).group(1)) 
        '''





'''
Projection Plots of Ionization Parameter, Number Density, 
Mass Density, Temperature, Metallicity
'''

'''
# TODO update convert to plt to do both versions of plots with and without limits

# Wrapper for making diagnostic plots of given fields
def plot_diagnostics(ds, sp, data_file, center, width, lims_dict=None):
    redshift = ds.current_redshift

    # Ionization Parameter
    proj_ion_param = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'ion-param'), ('gas', 'number_density'))
    if lims_dict == None:
        convert_to_plt_2(data_file, proj_ion_param, 'proj', 'ion-param', width, redshift, 'Ionization Parameter')
    else:
        convert_to_plt_2(data_file, proj_ion_param, 'proj', 'ion-param', width, redshift, 'Ionization Parameter', lims_dict['Ionization Parameter'])

    # Number Density
    proj_num_density = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'number_density'), ('gas', 'number_density'))
    if lims_dict == None:
        convert_to_plt_2(data_file, proj_num_density, 'proj', 'number_density', width, redshift, r'Number Density [$cm^{-3}$]')
    else:
        convert_to_plt_2(data_file, proj_num_density, 'proj', 'number_density', width, redshift, r'Number Density [$cm^{-3}$]', lims_dict['Number Density'])

    # Mass Density
    proj_density = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'density'), ('gas', 'number_density'))
    if lims_dict == None: 
        convert_to_plt_2(data_file, proj_density, 'proj', 'density', width, redshift, r'Density [$g\: cm^{-3}$]')
    else:
        convert_to_plt_2(data_file, proj_density, 'proj', 'density', width, redshift, r'Density [$g\: cm^{-3}$]', lims_dict['Mass Density'])

    # Temperature
    proj_temp = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'temperature'), ('gas', 'number_density'))
    if lims_dict == None:
        convert_to_plt_2(data_file, proj_temp, 'proj', 'temperature', width, redshift, 'Temperature [K]')
    else:
        convert_to_plt_2(data_file, proj_temp, 'proj', 'temperature', width, redshift, 'Temperature [K]', lims_dict['Temperature'])

    # Metallicity
    proj_metallicity = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'metallicity'), ('gas', 'number_density'))
    if lims_dict == None:
        convert_to_plt_2(data_file, proj_metallicity, 'proj', 'metallicity', width, redshift, 'Metallicity')
    else:
        convert_to_plt_2(data_file, proj_metallicity, 'proj', 'metallicity', width, redshift, 'Metallicity', lims_dict['Metallicity'])


Visualizing Line Intensities


def plot_intensities(ds, sp, data_file, center, width, lims_dict=None):
    redshift = ds.current_redshift

    for line in lines:
        proj = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'intensity_' + line), None)

        if line == 'H1_6562.80A':
            line_title = r'H$\alpha$_6562.80A'
        else:
            line_title = line

        if lims_dict == None:
            convert_to_plt_2(data_file, proj, 'proj', 'intensity_' + line, width, redshift, \
                        'Projected ' + line_title.replace('_', ' ') + r' Flux [$erg\: s^{-1}\: cm^{-2}$]')
        else:
            convert_to_plt_2(data_file, proj, 'proj', 'intensity_' + line, width, redshift, \
                        'Projected ' + line_title.replace('_', ' ') + r' Flux [$erg\: s^{-1}\: cm^{-2}$]', lims_dict[line])

    proj_halpha_l = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'luminosity_H1_6562.80A'), None)
    convert_to_plt_2(data_file, proj_halpha_l, 'proj', 'luminosity_H1_6562.80A', width, redshift, r'Projected H$\alpha$ 6562.80A Luminosity [$erg\: s^{-1}$]')

    slc_halpha = slc_plot(ds, (width, 'pc'), center, ('gas', 'intensity_H1_6562.80A'))
    convert_to_plt_2(data_file, slc_halpha, 'slc', 'intensity_H1_6562.80A', width, redshift, r'H$\alpha$ 6562.80A Flux [$erg\: s^{-1}\: cm^{-2}$]')
'''





# TODO
'''
Plot Spectra at simulated wavelengths
'''

'''
def spectra_driver(ds, luminosities, data_file, linear=False):
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

    # Raw spectra values
    plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname=os.path.join(directory, data_file + "_raw_spectra"), \
             sim_spectra=False, redshift_wavelengths=False)

    # Sim spectra, not redshifted
    plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname=os.path.join(directory, data_file + "_sim_spectra"), \
             sim_spectra=True, redshift_wavelengths=False)

    # Sim spectra, redshifted
    plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname=os.path.join(directory, data_file + "_sim_spectra_redshifted"), \
             sim_spectra=True, redshift_wavelengths=True)
    
    # With Limits for animation
    # Sim spectra, not redshifted
    plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname=os.path.join(directory, data_file + "_sim_spectra_lims"), \
             sim_spectra=True, redshift_wavelengths=False, lum_lims=[32, 44], flux_lims=[-24, -19])

    # Sim spectra, redshifted
    plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname=os.path.join(directory, data_file + "_sim_spectra_redshifted_lims"), \
             sim_spectra=True, redshift_wavelengths=True, lum_lims=[32, 44], flux_lims=[-24, -19])
    
    # Linear Scale
    # Sim spectra, not redshifted
    #plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname=os.path.join(directory, data_file + "_sim_spectra_lims_linear"), \
    #         sim_spectra=True, redshift_wavelengths=False, lum_lims=[32, 38], flux_lims=[-24, -20], linear=True)

    # Sim spectra, redshifted
    #plot_spectra(wavelengths, luminosities, flux_arr, z, 10e-25, R, fname=os.path.join(directory, data_file + "_sim_spectra_redshifted_lims_linear"), \
    #         sim_spectra=True, redshift_wavelengths=True, lum_lims=[32, 38], flux_lims=[-24, -20], linear=True)

    

def plot_spectra(wavelengths, luminosities, flux_arr, z, noise_lvl, R, \
                 fname, sim_spectra=False, redshift_wavelengths=False, lum_lims=None, flux_lims=None, linear=False):

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
        if linear == False:
            ax1.plot(x_range, np.log10(y_vals_f), color='black')
        else:
            ax1.plot(x_range, y_vals_f, color='black')
        if flux_lims != None:
            if linear == False:
                ax1.set_ylim(flux_lims)
            else:
                ax1.set_ylim([10**flux_lims[0], 10**flux_lims[1]])
        ax1.set_xlabel(r'Wavelength [$\AA$]')
        if linear == False:
            ax1.set_ylabel(r'Log(Flux) [$erg\: s^{-1}\: cm^{-2} \: \AA^{-1}]$')
        else:
            ax1.set_ylabel(r'Flux [$erg\: s^{-1}\: cm^{-2} \: \AA^{-1}]$')
        plt.savefig(fname)
        plt.close()

        fig, ax1 = plt.subplots(1)
        x_range, y_vals_l = plot_voigts(wavelengths, luminosities, line_widths, [0.0]*len(wavelengths), noise_lvl, pad)
        if linear == False:
            ax1.plot(x_range, np.log10(y_vals_l), color='black')
        else:
            ax1.plot(x_range, y_vals_l, color='black')
        if lum_lims != None:
            if linear == False:
                ax1.set_ylim(lum_lims)
            else:
                ax1.set_ylim([10**lum_lims[0], 10**lum_lims[1]])
        ax1.set_xlabel(r'Wavelength [$\AA$]')
        if linear == False:
            ax1.set_ylabel(r'Log(Luminosity) [$erg\: s^{-1} \: \AA^{-1}]$')
        else:
            ax1.set_ylabel(r'Luminosity [$erg\: s^{-1} \: \AA^{-1}]$')
        lum_fname = fname + '_lum'
        plt.savefig(lum_fname)
        plt.close()
    else:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(wavelengths, np.log10(flux_arr), 'o')
        ax2.plot(wavelengths, np.log10(luminosities), 'o')
        ax2.set_xlabel(r'Wavelength [$\AA$]')
        ax1.set_ylabel(r'Log(Flux) [$erg\: s^{-1}\: cm^{-2}$]')
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



Star + Gas Plot
Adapted from work by Sarunyapat Phoompuang


def star_gas_overlay(ds, ad, sp, data_file, center, width, field, lims_dict=None):
    redshift = ds.current_redshift

    lims = lims_dict["H1_6562.80A"] # TODO

    directory = 'analysis/' + data_file + '_analysis'

    if not os.path.exists(directory):
        os.makedirs(directory)

    fname = os.path.join(directory, data_file + '_' + str(width) + 'pc_stellar_dist')

    # Finding center of the data
    x_pos = np.array(ad["star", "particle_position_x"])
    y_pos = np.array(ad["star", "particle_position_y"])
    z_pos = np.array(ad["star", "particle_position_z"])
    x_center = np.mean(x_pos)
    y_center = np.mean(y_pos)
    z_center = np.mean(z_pos)
    x_pos = x_pos - x_center
    y_pos = y_pos - y_center
    z_pos = z_pos - z_center

    # Create a ProjectionPlot
    #p = proj_plot(ds, sp, (width, 'pc'), center, ('gas', 'intensity_H1_6562.80A'), None)
    p = yt.ProjectionPlot(ds, "z", field,
                      width=(width, 'pc'),
                      data_source=sp,
                      buff_size=(2000, 2000),
                      center=center)

    p_frb = p.frb  # Fixed-Resolution Buffer from the projection
    p_img = np.array(p_frb["gas", 'intensity_H1_6562.80A'])
    star_bins = 2000
    star_mass = np.ones_like(x_pos) * 10
    pop2_xyz = np.array(ds.arr(np.vstack([x_pos, y_pos, z_pos]), "code_length").to("pc")).T
    extent_dens = [-width/2, width/2, -width/2, width/2]
    #gas_range = (20, 2e4)
    norm1 = LogNorm(vmin=lims[0], vmax=lims[1])
    #norm1 = LogNorm(vmin = gas_range[0], vmax = gas_range[1])
    #plt.figure(figsize = (12,8))
    #plt.imshow(p_img, norm = norm1, extent = extent_dens, origin = 'lower', aspect = 'auto', cmap = 'inferno')
    #plt.colorbar(label = r"Number Density [1/cm$^3$]")
    # plt.scatter(pop2_xyz[:, 0], pop2_xyz[:, 1], s = 5, marker = '.', color = 'red')
    #plt.xlabel("X (pc)")
    #plt.ylabel("Y (pc)")
    #plt.title("Gas Density and Particle Positions")
    #plt.xlim(-width/2, width/2)
    #plt.ylim(-width/2, width/2)
    #plt.show()
	
    stellar_mass_dens, _, _ = np.histogram2d(pop2_xyz[:, 0],
                                             pop2_xyz[:, 1],
                                             bins = star_bins,
                                             weights = star_mass,
                                             range = [[-width / 2, width / 2],
                                                      [-width / 2, width / 2],
                                                     ],
    )
    stellar_mass_dens = stellar_mass_dens.T
    stellar_mass_dens = np.where(stellar_mass_dens <= 1, 0, stellar_mass_dens)
    stellar_range = [1, 1200]
    norm2 = LogNorm(vmin = stellar_range[0], vmax = stellar_range[1])
    plt.figure(figsize = (12, 8))
    lumcmap = "cmr.amethyst"
    plt.imshow(stellar_mass_dens, norm = norm2, extent = extent_dens, origin = 'lower', aspect = 'auto', cmap = 'winter_r')
    plt.colorbar(label = "Stellar Mass Density")
    plt.xlabel("X (pc)")
    plt.ylabel("Y (pc)")
    plt.title("Stellar Mass Density Distribution")
    plt.savefig(fname=fname)
    plt.close()
    	
    #print(np.min(p_img), np.max(p_img))  # Check for min/max values of p_img
    #print(np.min(stellar_mass_dens), np.max(stellar_mass_dens))  # Check for min/max values of stellar_mass_dens

    overlay_fname = fname + '_H1' #+ field
    fig, ax = plt.subplots(figsize = (12, 8))
    alpha_star = stellar_mass_dens
    alpha_star = np.where(stellar_mass_dens <= 1, 0.0, 1)

    print(alpha_star.shape)
    print(p_img.shape)

    img1 = ax.imshow(p_img, norm = norm1, extent = extent_dens, origin = 'lower', aspect = 'auto', cmap = 'inferno', alpha = 1, interpolation='bilinear')
    cbar1 = fig.colorbar(img1, ax = ax, orientation = 'vertical', pad = 0.02)
    cbar1.set_label('Projected ' + r'H$\alpha$ Flux [$erg\: s^{-1}\: cm^{-2}$]') # TODO
    img2 = ax.imshow(stellar_mass_dens, norm = norm2, extent = extent_dens, origin = 'lower', aspect = 'auto', cmap = 'winter_r', interpolation='bilinear')

    # Make sure alpha_star matches the image shape (it must be the same size as the image)
    if img2.get_array().shape != alpha_star.shape:
        print(f"Shape mismatch: Image shape {img2.get_array().shape} vs. alpha_star shape {alpha_star.shape}")
    
    img2.set_alpha(alpha_star)  # Apply alpha mask after plotting

    cbar2 = fig.colorbar(img2, ax = ax, orientation = 'vertical', pad = 0.02)
    cbar2.set_label("Stellar Mass Density")
    # ax.scatter(pop2_xyz[:, 0], pop2_xyz[:, 1], s=5, marker='.', color='black')
    ax.set_xlabel("X (pc)")
    ax.set_ylabel("Y (pc)")
    ax.set_title(r'H$\alpha$ Flux' + ' and Stellar Mass Density Distribution')
    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(-width / 2, width / 2)

    plt.text(0.05, 0.05, f'z = {redshift:.5f}', color='white', fontsize=9, ha='left', va='bottom', \
             transform=plt.gca().transAxes)

    plt.savefig(fname=overlay_fname)
    plt.close()

'''

# sp.quantities.center_of_mass(use_gas=False, use_particles=True, particle_type="star")


# TODO 1D Profile Plots, Phase Plots
# Save header information to a text file
# Save lines and wavelengths
# Save mins, maxs, means
# TODO cloudy run