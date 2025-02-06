'''
Line Emission from a cloudy output table
'''

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import copy

class EmissionLineInterpolator:
    def __init__(self, filename):
        '''
        Initializes the interpolator with data loaded from the given filename.
        
        Parameters:
        filename (str): The name of the file to load the line emission data from.
        '''
        self.filename = filename
        self._load_data()
        self._reconfigure_data_cube()
        self._create_interpolators()

    def _load_data(self):
        '''Load line emission data from the file.'''
        # Read line emission data (line list, run params)
        minU, maxU, stepU, minN, maxN, stepN, minT, maxT, stepT = np.loadtxt(self.filename, unpack=True, dtype=float, max_rows=1, skiprows=5)
        self.minU, self.maxU, self.stepU = minU, maxU, stepU
        self.minN, self.maxN, self.stepN = minN, maxN, stepN
        self.minT, self.maxT, self.stepT = minT, maxT, stepT
        print(self.minU, self.maxU, self.stepU, self.minN, self.maxN, self.stepN, self.minT, self.maxT, self.stepT)
        
        self.ll = np.loadtxt(self.filename, unpack=True, dtype=float, skiprows=7)
        print(self.ll.shape)

    def _reconfigure_data_cube(self):
        '''Reconfigure the linelist into a 4D data cube.'''

        titls = [
            'H1_6562.80A','O1_1304.86A','O1_6300.30A','O2_3728.80A','O2_3726.10A','O3_1660.81A',
            'O3_1666.15A','O3_4363.21A','O3_4958.91A','O3_5006.84A', 'He2_1640.41A','C2_1335.66A',
            'C3_1906.68A','C3_1908.73A','C4_1549.00A','Mg2_2795.53A','Mg2_2802.71A','Ne3_3868.76A',
            'Ne3_3967.47A','N5_1238.82A','N5_1242.80A','N4_1486.50A','N3_1749.67A','S2_6716.44A','S2_6730.82A'
        ]
        
        # Number of emission lines
        self.ncols = len(titls)

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
        '''Create interpolators for each emission line.'''
        self.interpolator = [None] * self.ncols
        for i in np.arange(self.ncols):
            self.interpolator[i] = RegularGridInterpolator((self.logU, self.logN, self.logT), self.cub[i])

        # Normalize by the density squared
        self.dens_normalized_cub = self.cub.copy()
        for i in np.arange(self.dimN):
            self.dens_normalized_cub[:, :, i, :] = self.dens_normalized_cub[:, :, i, :] / 10 ** (2 * self.logN[i])

        # Create density squared normalized interpolators
        self.dens_normalized_interpolator = [None] * self.ncols
        for i in np.arange(self.ncols):
            self.dens_normalized_interpolator[i] = RegularGridInterpolator((self.logU, self.logN, self.logT), self.dens_normalized_cub[i])

    def get_interpolator(self, lineidx, dens_normalized):
        '''
        Returns the interpolator for the specified line and normalization option.
        
        Parameters:
        lineidx (int): Index of the emission line.
        dens_normalized (bool): Whether to use the density squared normalized interpolator.
        
        Returns:
        RegularGridInterpolator: The corresponding interpolator.
        '''
        if dens_normalized:
            return self.dens_normalized_interpolator[lineidx]
        return self.interpolator[lineidx]

    def get_line_emission(self, idx, dens_normalized):
        '''
        Returns a function for line emission of index idx.
        Allows for the batch creation of intensity fields for various lines.
        
        Parameters:
        idx (int): The index of the emission line.
        dens_normalized (bool): Whether to use the density squared normalized version.
        
        Returns:
        function: A function that calculates the emission for a given field and data.
        '''
        def _line_emission(field, data):
            interpolator = self.get_interpolator(idx, dens_normalized)

            # Change to log values
            U_val = data['gas', 'ion-param'].value
            N_val = data['gas', 'number_density'].value
            T_val = data['gas', 'temperature'].value

            # Cut off negative temperatures
            T_val = np.where(T_val < 0.0, 1e-4, T_val)

            U = np.log10(U_val)
            N = np.log10(N_val)
            T = np.log10(T_val)

            # Adjust log values to within bounds supported by interpolation table
            Uadj = np.where(U < self.minU, self.minU, U)
            Uadj = np.where(Uadj > self.maxU, self.maxU, Uadj)

            Nadj = np.where(N < self.minN, self.minN, N)
            Nadj = np.where(Nadj > self.maxN, self.maxN, Nadj)

            Tadj = np.where(T < self.minT, self.minT, T)
            Tadj = np.where(Tadj > self.maxT, self.maxT, Tadj)

            tup = np.stack((Uadj, Nadj, Tadj), axis=-1)

            size = Nadj.size

            # Return interpolated values weighted by metallicity for non-Hydrogen and Helium lines
            interp_val = interpolator(tup)

            if idx not in [0, 10]:
                interp_val = interp_val * data['gas', 'metallicity']  # TODO: check *4

            if dens_normalized:
                interp_val = interp_val * data['gas', 'number_density'] ** 2
            else:
                interp_val = interp_val * data['gas', 'number_density'] / data['gas', 'number_density']

            return interp_val
        return copy.deepcopy(_line_emission)

# Usage example
filename = 'linelist.dat'
emission_interpolator = EmissionLineInterpolator(filename)

# Get interpolated emission for a specific line (e.g., line 0, normalized by density)
line_emission_function = emission_interpolator.get_line_emission(0, dens_normalized=True)
