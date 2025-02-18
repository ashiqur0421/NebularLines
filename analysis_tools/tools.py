'''
Suite of functions for performing analysis on data
output from many time slices.
'''

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# TODO read in lines avail
lines=["H1_6562.80A","O1_1304.86A","O1_6300.30A","O2_3728.80A","O2_3726.10A","O3_1660.81A",
       "O3_1666.15A","O3_4363.21A","O3_4958.91A","O3_5006.84A", "He2_1640.41A","C2_1335.66A",
       "C3_1906.68A","C3_1908.73A","C4_1549.00A","Mg2_2795.53A","Mg2_2802.71A","Ne3_3868.76A",
       "Ne3_3967.47A","N5_1238.82A","N5_1242.80A","N4_1486.50A","N3_1749.67A","S2_6716.44A","S2_6730.82A"]

def check_file_pattern(folder_path, pattern):
    '''Checks if a file matching the pattern exists in the folder.'''
    files = glob.glob(f"{folder_path}/{pattern}")
    return len(files) > 0


def lvz(data_path:str, sim_titl:str, log) -> None:
    '''
    Luminosity vs. Redshift
    Visualize line luminosity evolution over time

    folder_path: path to folder with line_luminosity files
    sim_titl: str referring to simulation
    '''

    directory = f'{sim_titl}_post_analysis'

    if not os.path.exists(directory):
        os.makedirs(directory)

    pattern = 'output_*_line_luminosity.txt'

    files = sorted(glob.glob(f"{data_path}/{pattern}"))

    num_lines = len(lines)

    if len(files) > 0:
        num_slices = len(files)
        print(num_slices)
        line_lums = [0]*num_slices
        n = 0

        for file in files:
            line_lums[n] = np.loadtxt(file, dtype='float')
            n += 1

    line_lums = np.array(line_lums)

    print(line_lums.shape)

    # TODO - will get actual redshift values from each sim on a rerun
    Z = np.linspace(9.952258, 8.031847, num_slices)

    y_label = 'Luminosity [$erg\: s^{-1}$]'


    redshift, Mstar = np.loadtxt('/Users/bnowicki/Documents/Github/NebularLines/analysis_tools/logSFC', usecols=[2, 7], unpack=True)
    #redshift = SFC[:2]
    Mstar = np.cumsum(Mstar)
    #print(redshift)
    # deriv cumulative

    # TODO one line and ratios wrt that line
    # two responses
    # time shift and ratio (delayed response?)
    # evolutionary sequence of gas phase following burst
    # TODO - redshift 1/aexp in info_output...txt

    plt.figure(figsize=(10, 6))

    # Plot each line in 'line_lums' against 'Z'
    #for i in range(line_lums.shape[1]):
    for i in [0, 3]:
        if log == True:
            plt.plot(Z, np.log10(line_lums[:, i]), label=lines[i])
        else:
            plt.plot(Z, line_lums[:, i], label=lines[i])

    plt.plot(redshift, np.log(Mstar)+30, label=lines[i])

    # Label the axes
    plt.xlabel('Redshift')
    plt.ylabel(r'Log(Luminosity) [$erg\: s^{-1}$]')

    # TODO O2 ratio

    # Add a legend
    plt.legend(title=sim_titl, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax = plt.gca()
    ax.invert_xaxis()
    ax.set_xlim(10.0, 8.0)

    # Display the plot
    plt.tight_layout()
    #plt.savefig(f'{directory}/{sim_titl}_lvz_log={log}')
    plt.show()

lvz('/Users/bnowicki/Documents/Scratch/movie_dir_2', 'CC Fiducial', True)
lvz('/Users/bnowicki/Documents/Scratch/movie_dir_2', 'CC Fiducial', False)

# TODO cloudy run class

# TODO
class Simulation_Analysis:
    '''
    Class containing post-processing functions for a simulation
    '''

    main_table = pd.DataFrame()

    def __init__(self, sim_titl:str, data_path:str):
        '''
        folder_path: path to folder with line_luminosity files
        sim_titl: str referring to simulation
        '''

        self.sim_titl = sim_titl
        self.data_path = data_path
        self.lines = lines

        self.make_dir()


    def make_dir(self):
        directory = f'{self.sim_titl}_post_analysis'

        if not os.path.exists(directory):
            os.makedirs(directory)

    
    def populate_table(self):
        '''
        Populate main_table across many time slices.
        '''




    def lvz(self):
        '''
        Luminosity vs. Redshift
        '''



    

