import copy
import numpy as np
from scipy.interpolate import RegularGridInterpolator

'''
emission.py

Author: Braden Nowicki

Generate line emission fields from a Cloudy-generated linelist table.

Class structure allows for modularity with emission from multiple
Cloudy-generated tables. Instantiate an instance of the
EmissionLineInterpolator class with the file path for a Cloudy-generated
emission table (a data cube), example:

    filename='linelist.dat'
    emission_interpolator = EmissionLineInterpolator(filename)

Upon instantiation, the EmissionLineInterpolator object reads the data table,
reconfigures it into a data cube for usage, and creates interpolators
for all emission lines. The interpolators take Log(Ionization Parameter U),
Log(Number Density), Log(Temperature) and return corresponding fluxes
in specified nebular recombination emission lines. The function 
get_line_emission() can then be used to get the function necessary for a
derived field given the index of the emission line in a line list.
These derived fields are used in yt to contain information about
the emission in each cell. Each line has two interpolators: one returning
the actual flux value, and one returning the flux value normalized
by the density squared. This is how to get interpolated emission for a specific
line (e.g. line 0 = HII, density normalized):

    line_emission_function = \
        emission_interpolator.get_line_emission(0, dens_normalized=True)

'''

# TODO docstrings

class EmissionLineInterpolator:
    def __init__(self, filename, lines):
        '''
        Initializes the interpolator with line list loaded from the given 
        filename/filepath.
        
        Parameters:
        filename (str): The name/path of the file to load the line emission 
        data from. This contains a text table. A commented (unread) header
        displays the lines in the table. The next line (read) gives
        information about the parameter space:

            minU, maxU, stepU, minN, maxN, stepN, minT, maxT, stepT

        Each column contains the flux [erg s^-1 cm^-2] for the line.
        Each row is for a specific configuration of parameters.
        The parameters iterate in a known fashion, allowing each column
        to be reconfigured into a data cube (flux at each U, N, T point).

        '''

        self.filename = filename
        self.lines = lines
        self._load_data()
        self._reconfigure_data_cube()
        self._create_interpolators()


    def _load_data(self):
        '''
        Load line emission data from the file.
        '''

        # Read line emission data (line list, run params)
        minU, maxU, stepU, minN, maxN, stepN, minT, maxT, stepT = \
            np.loadtxt(self.filename, unpack=True, dtype=float, max_rows=1, 
                       skiprows=5)
        self.minU, self.maxU, self.stepU = minU, maxU, stepU
        self.minN, self.maxN, self.stepN = minN, maxN, stepN
        self.minT, self.maxT, self.stepT = minT, maxT, stepT
        print(f'minU={self.minU}, maxU={self.maxU}, stepU={self.stepU}, ' +
              f'minN={self.minN}, maxN={self.maxN}, stepN={self.stepN}, ' +
              f'minT={self.minT}, maxT={self.maxT}, stepT={self.stepT}')
        
        self.ll = np.loadtxt(self.filename, unpack=True, dtype=float, 
                             skiprows=7)
        print(f'Line List Shape = {self.ll.shape}')


    def _reconfigure_data_cube(self):
        '''
        Reconfigure the linelist into a data cube.
        '''
        
        # Number of emission lines
        self.ncols = len(self.lines)

        # Calculate the grid dimensions
        self.dimU = int((self.maxU - self.minU) / self.stepU) + 1
        self.dimT = int((self.maxT - self.minT) / self.stepT) + 1
        self.dimN = int((self.maxN - self.minN) / self.stepN) + 1
        print(self.dimU, self.dimN, self.dimT)

        # The log values of U, N, T in the run/grid
        self.logU = self.minU + np.arange(self.dimU) * self.stepU
        self.logN = self.minN + np.arange(self.dimN) * self.stepN
        self.logT = self.minT + np.arange(self.dimT) * self.stepT

        # (Ionization Parameter, Density, Temperature)
        # (U, density, T)
        # d defines the cube dimensions
        # 4D cube with ncols line strengths at each U, N, T coordinate
        # cub[i] is the cube for a single emission line
        # reshape the 1D array ll[i, :] of a certain line's strengths
        # to U, N, T grid
        # Initialize the 4D data cube for each emission line
        d = (self.dimU, self.dimN, self.dimT)
        self.cub = np.zeros((self.ncols, self.dimU, self.dimN, self.dimT))

        for i in range(self.ncols):
            self.cub[i] = np.reshape(self.ll[i, :], d)


    def _create_interpolators(self):
        '''
        Create interpolators for each emission line.
        '''

        self.interpolator = [None] * self.ncols
        for i in np.arange(self.ncols):
            self.interpolator[i] = RegularGridInterpolator(
                (self.logU, self.logN, self.logT), self.cub[i]
            )

        # Normalize by the density squared
        self.dens_normalized_cub = self.cub.copy()
        for i in np.arange(self.dimN):
            self.dens_normalized_cub[:, :, i, :] = \
                self.dens_normalized_cub[:, :, i, :] / 10 ** (2 * self.logN[i])

        # Create density squared normalized interpolators
        self.dens_normalized_interpolator = [None] * self.ncols
        for i in np.arange(self.ncols):
            self.dens_normalized_interpolator[i] = RegularGridInterpolator(
                (self.logU, self.logN, self.logT), self.dens_normalized_cub[i]
            )


    def get_interpolator(self, lineidx, dens_normalized):
        '''
        Returns the interpolator for the specified line and normalization 
        option.
        
        Parameters:
        lineidx (int): Index of the emission line.
        dens_normalized (bool): Flag - whether to use the density squared 
        normalized interpolator.
        
        Returns:
        RegularGridInterpolator: The corresponding interpolator object.
        '''

        if dens_normalized:
            return self.dens_normalized_interpolator[lineidx]
        return self.interpolator[lineidx]


    def get_line_emission(self, idx, dens_normalized):
        '''
        Returns a function for line emission of index idx.
        Allows for the batch creation of flux derived fields for various lines.
        
        Parameters:
        idx (int): The index of the emission line.
        dens_normalized (bool): Flag - whether to use the density squared 
        normalized version.
        
        Returns:
        function: A function that calculates the emission as a derived field.
        The data parameter represents simulation data loaded into yt.
        '''

        def _line_emission(field, data):
            interpolator = self.get_interpolator(idx, dens_normalized)

            # Change to log values
            U_val = data['gas', 'ion_param'].value
            #N_val = data['gas', 'number_density'].value
            N_val = data['gas', 'my_H_nuclei_density'].value
            #T_val = data['gas', 'temperature'].value
            T_val = data['gas', 'my_temperature'].value

            # Truncate negative temperatures
            # Temperature is a derived/calculated field; there are some
            # cases in which it is close to or equal to 0.0 K.
            T_val = np.where(T_val < 0.0, 1e-4, T_val)

            U = np.log10(U_val)
            N = np.log10(N_val)
            T = np.log10(T_val)

            # Adjust log values to within bounds supported by interpolation 
            # #table
            Uadj = np.where(U < self.minU, self.minU, U)
            Uadj = np.where(Uadj > self.maxU, self.maxU, Uadj)

            Nadj = np.where(N < self.minN, self.minN, N)
            Nadj = np.where(Nadj > self.maxN, self.maxN, Nadj)

            # TODO set to 0 below
            Tadj = np.where(T < self.minT, self.minT, T)
            Tadj = np.where(Tadj > self.maxT, self.maxT, Tadj)

            tup = np.stack((Uadj, Nadj, Tadj), axis=-1)

            size = Nadj.size

            # Return interpolated values weighted by metallicity for 
            # non-Hydrogen and Helium lines
            interp_val = interpolator(tup)
            interp_val[np.where(Tadj <= self.minT)] = 0

            # TODO check metallicity: mult by 4? solar metallicity?
            if idx not in [0, 10]:
                interp_val = interp_val * data['gas', 'metallicity']

            if dens_normalized:
                interp_val = interp_val * data['gas', 'my_H_nuclei_density'] ** 2
            else:
                interp_val = interp_val * data['gas', 'my_H_nuclei_density'] / \
                    data['gas', 'my_H_nuclei_density']

            return interp_val
        return copy.deepcopy(_line_emission)
    

    def get_luminosity(self, line):
        '''
        Return function for derived luminosity field of each line.
        The Cloudy flux is obtained assuming a gas cloud of height = 1 cm,
        Returns flux values erg s^-1 c^-2.
        Multiply the flux at each cell by the volume of the cell
        to obtain the intrinsic luminosity.

        line (str): desired emission line from field
        '''

        def _luminosity(field, data):
            return data['gas', 'flux_' + line]*data['gas', 'volume']
        return copy.deepcopy(_luminosity)

