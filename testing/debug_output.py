import yt
#filename = "/Users/bnowicki/Documents/Research/Ricotti/output_00273/info_00273.txt"
filename = "/Users/bnowicki/Documents/Research/Ricotti/output_00273/info_00273.txt"
#filename = "/Users/bnowicki/Documents/Research/Ricotti/output_00353/info_00353.txt"

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
    "refinement-param"
]


epf = [
    ("particle_family", "b"),
    ("particle_tag", "b"),
    ("particle_birth_epoch", "d"),
    ("particle_metallicity", "d"),
]

#yt.mylog.setLevel(10)

#ds2 = yt.load(filename, fields=cell_fields, extra_particle_fields=epf)
ds2 = yt.load(filename, extra_particle_fields=epf)
ad2 = ds2.all_data()
#star_ctr=galaxy_visualization.star_center(ad)
#sp = ds.sphere(star_ctr, (3000, "pc"))
#sp_lum = ds.sphere(star_ctr, (10, 'kpc'))
#width = 1500


sim_run = filename.split('/')[-1]

print(ds2.field_list)
#display(ds2.fields)