import os
import re
import ffmpeg

def make_animation(image_pattern, output_file, framerate=2, resolution=(1920, 1080)):
    '''
    Creates video from a sequence of images
    '''

    (
        ffmpeg
        .input(image_pattern, pattern_type='glob', framerate=framerate)
        .output(output_file, video_bitrate='5000k',
        s=f'{resolution[0]}x{resolution[1]}',  # Set the resolution
        pix_fmt='yuv420p')  # Ensure compatibility with most players
        .run()
    )

#create_movies_for_sequences(image_folder, frame_rate, resolution)

lines=["H1_6562.80A","O1_1304.86A","O1_6300.30A","O2_3728.80A","O2_3726.10A","O3_1660.81A",
       "O3_1666.15A","O3_4363.21A","O3_4958.91A","O3_5006.84A", "He2_1640.41A","C2_1335.66A",
       "C3_1906.68A","C3_1908.73A","C4_1549.00A","Mg2_2795.53A","Mg2_2802.71A","Ne3_3868.76A",
       "Ne3_3967.47A","N5_1238.82A","N5_1242.80A","N4_1486.50A","N3_1749.67A","S2_6716.44A","S2_6730.82A"]

image_dir = '/Users/bnowicki/Documents/Research/Ricotti/analysis/movie_dir/'
image_patterns = ['output_*_1500pc_density_proj', 'output_*_1500pc_density_proj_lims',
                  'output_*_1500pc_ion-param_proj', 'output_*_1500pc_ion-param_proj_lims',
                  'output_*_1500pc_luminosity_H1_6562,80A_proj', 'output_*_1500pc_intensity_H1_6562,80A_slc',
                  'output_*_1500pc_metallicity_proj', 'output_*_1500pc_metallicity_proj_lims',
                  'output_*_1500pc_number_density_proj', 'output_*_1500pc_number_density_proj_lims',
                  'output_*_1500pc_temperature_proj', 'output_*_1500pc_temperature_proj_lims',
                  'output_*_raw_spectra',
                  'output_*_sim_spectra', 'output_*_sim_spectra_lum',
                  'output_*_sim_spectra_redshifted', 'output_*_sim_spectra_redshifted_lum']

for line in lines:
    line_pattern = line.replace('.', ',')
    pattern =  'output_*_1500pc_intensity_' + line_pattern + '_proj'
    pattern_lims = 'output_*_1500pc_intensity_' + line_pattern + '_proj_lims'
    image_patterns.append(pattern)
    image_patterns.append(pattern_lims)

for image_pattern in image_patterns:
    make_animation(image_dir + image_pattern + '.png', image_pattern + '.mp4', framerate=framerate, resolution=resolution)