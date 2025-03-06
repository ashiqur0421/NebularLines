import numpy as np
import sys
import glob
import os
import argparse
import matplotlib.pyplot as plt
import time
import yt
import shutil


FIELDS = ["Density",
          "x-velocity", "y-velocity", "z-velocity",
          "Pressure",
          "Metallicity",
          "xHI", "xHII", "xHeII", "xHeIII"]
#粒子変数の内容（追加分）
EPF= [('particle_family', 'b'),      #byte size
      ('particle_tag', 'b'),         #byte size
      ('particle_birth_epoch', 'd'), #double size
      ('particle_metallicity', 'd')] #double size

#number density of hydrogen atoms
def _my_H_nuclei_density(field, data):
    dn=data["ramses","Density"].in_cgs()
    XH_RAMSES=0.76 #defined by RAMSES in cooling_module.f90
    YHE_RAMSES=0.24 #defined by RAMSES in cooling_module.f90
    mH_RAMSES=yt.YTArray(1.6600000e-24,"g") #defined by RAMSES in cooling_module.f90

    return dn*XH_RAMSES/mH_RAMSES


#平均分子量を考慮した温度の定義
def _my_temperature(field, data):
    #y(i): abundance per hydrogen atom
    XH_RAMSES=0.76 #defined by RAMSES in cooling_module.f90
    YHE_RAMSES=0.24 #defined by RAMSES in cooling_module.f90
    mH_RAMSES=yt.YTArray(1.6600000e-24,"g") #defined by RAMSES in cooling_module.f90
    kB_RAMSES=yt.YTArray(1.3806200e-16,"erg/K") #defined by RAMSES in cooling_module.f90

    dn=data["ramses","Density"].in_cgs()
    pr=data["ramses","Pressure"].in_cgs()
    yHI=data["ramses","xHI"]
    yHII=data["ramses","xHII"]
    yHe = YHE_RAMSES*0.25/XH_RAMSES
    yHeII=data["ramses","xHeII"]*yHe
    yHeIII=data["ramses","xHeIII"]*yHe
    yH2=1.-yHI-yHII
    yel=yHII+yHeII+2*yHeIII
    mu=(yHI+yHII+2.*yH2 + 4.*yHe) / (yHI+yHII+yH2 + yHe + yel)
    return pr/dn * mu * mH_RAMSES / kB_RAMSES

#マッハ数の定義
def _Mach(field, data):
    dn=data["ramses","Density"]
    pr=data["ramses","Pressure"]
    cI2 = pr/dn #等温音速
    vx=data["ramses","x-velocity"]
    vy=data["ramses","y-velocity"]
    vz=data["ramses","z-velocity"]
    v2 = vx**2 + vy**2 + vz**2
    mach = np.sqrt(v2/cI2)
    return mach

#金属度 [Z_solar]の定義
def _Zmetal_in_Zsun(field, data):
    Zmetal=data["ramses","Metallicity"]
    Zmetal_in_Zsun = Zmetal/0.02 #devided by Zsun = 0.02
    return Zmetal_in_Zsun

#フロア付き (phase diagramに利用)
def _Zmetal_in_Zsun_floor(field, data):
    floor_value = 1e-7
    Zmetal=data["ramses","Metallicity"]
    Zmetal_in_Zsun = Zmetal/0.02 #devided by Zsun = 0.02
    Zmetal_in_Zsun[Zmetal_in_Zsun < floor_value] = floor_value #フロアを課す
    return Zmetal_in_Zsun

#化学組成比 y(i) = n(i)/nH
#HI
def _yHI(field, data):
    yHI=data["ramses","xHI"]
    return yHI
#HII
def _yHII(field, data):
    yHII=data["ramses","xHII"]
    return yHII
#H2
def _yH2(field, data):
    yHI=data["ramses","xHI"]
    yHII=data["ramses","xHII"]
    yH2=(1.-yHI-yHII)*0.5
    return yH2
#フロア付き (phase diagramに利用)
def _yH2_floor(field, data):
    floor_value = 1e-8
    yHI=data["ramses","xHI"]
    yHII=data["ramses","xHII"]
    yH2=(1.-yHI-yHII)*0.5
    yH2[yH2 < floor_value] = floor_value #フロアを課す
    return yH2

## *** WARNING *** WARNING *** WARNING *** WARNING *** WARNING *** WARNING ***##
## the unit conversion in the following variable seems wrong
## *** WARNING *** WARNING *** WARNING *** WARNING *** WARNING *** WARNING ***##
#J21 (LW intensity in unit of 1e-21 erg s-1 cm-2 Hz-1 sr-1)
#conversion from photon density nLW to J21
# n_nu = unu/<hnu> = 4pi/c Jnu/<hnu>
# -> J21 = 1e21 * c[cm/s]* nLW[cm-3]/4pi  * <hnu> / Delta nu = (where <hnu> = 12.4 eV, Delta nu = 2.4 eV/h)
# see yt/frontends/ramses/fields.py for the definition of data["rt","photon_density_1"] in yt
def _J21(field, data):
    if 'photon_density_1' in dir(ds.fields.rt):
        from yt.frontends.ramses.field_handlers import RTFieldFileHandler
        p = RTFieldFileHandler.get_rt_parameters(ds).copy()
        p.update(ds.parameters)
        cgs_c =  2.99792458e10     #light velocity
        cgs_h = 6.62606876e-27     #planck constant
        hnu_ave = 12.4e0           #average energy of LW photons in eV
        Delta_nu = 2.4e0/cgs_h           #energy range of LW photons in eV/h
        NpLW=data['ramses-rt','Photon_density_1']*p["unit_pf"]/cgs_c #physical photon number density in cm-3
        J21 = yt.YTArray(1e21 * cgs_c / (4*np.pi) * hnu_ave/Delta_nu * NpLW, '1')    #KS DEBUG
        return J21


#physical LW photon density (independent of rt_c_frac in steady states)
def _NpLW(field, data):
    if 'photon_density_1' in dir(ds.fields.rt):
        from yt.frontends.ramses.field_handlers import RTFieldFileHandler
        p = RTFieldFileHandler.get_rt_parameters(ds).copy()
        p.update(ds.parameters)
        cgs_c =  2.99792458e10     #light velocity
        print ("ds.parameters", ds.parameters)
        NpLW=data["ramses-rt","Photon_density_1"]*p["unit_pf"]/cgs_c
        return NpLW

#physical HI ionizing photon density (independent of rt_c_frac in steady states)
def _NpHI(field, data):
    if 'photon_density_2' in dir(ds.fields.rt):
        from yt.frontends.ramses.field_handlers import RTFieldFileHandler
        p = RTFieldFileHandler.get_rt_parameters(ds).copy()
        p.update(ds.parameters)
        cgs_c =  2.99792458e10     #light velocity
        print ("ds.parameters", ds.parameters)
        NpHI=data["ramses-rt","Photon_density_2"]*p["unit_pf"]/cgs_c
        return NpHI


###########################
# データの読み込み
###########################
#yt.funcs.mylog.setLevel(1) #log表示レベルの設定
infofile_fp = "/Users/bnowicki/Documents/Research/Ricotti/output_00273/info_00273.txt"
#ds = yt.load(infofile_fp, fields=FIELDS, extra_particle_fields=EPF)
ds = yt.load(infofile_fp, extra_particle_fields=EPF)

#if not args.DM_only:
ds.add_field(("gas","my_H_nuclei_density"),function=_my_H_nuclei_density, units="cm**-3", sampling_type="cell")
ds.add_field(("gas","my_temperature"),function=_my_temperature, units="K",sampling_type="cell")
ds.add_field(("gas","Mach"),function=_Mach, units="1",sampling_type="cell")
ds.add_field(("gas","Zmetal_in_Zsun"),function=_Zmetal_in_Zsun, units="1",sampling_type="cell")
ds.add_field(("gas","Zmetal_in_Zsun_floor"),function=_Zmetal_in_Zsun_floor, units="1",sampling_type="cell")
ds.add_field(("gas","yHI"),function=_yHI, units="1",sampling_type="cell")
ds.add_field(("gas","yHII"),function=_yHII, units="1",sampling_type="cell")    
ds.add_field(("gas","yH2"),function=_yH2, units="1",sampling_type="cell")
ds.add_field(("gas","yH2_floor"),function=_yH2_floor, units="1",sampling_type="cell")
ds.add_field(("gas","J21"),function=_J21, units="1",sampling_type="cell")
ds.add_field(("gas","NpLW"),function=_NpLW, units="1",sampling_type="cell")
ds.add_field(("gas","NpHI"),function=_NpHI, units="1",sampling_type="cell")        

#meta data
reg = ds.unit_registry
code_length = reg.lut['code_length'][0]                #code_length
current_redshift = ds.current_redshift                 #redshift
hubble_constant  = ds.hubble_constant                  #hubble_constant h [100 km/s/Mpc]
omega_m = ds.omega_matter                              #omega matter
omega_l = ds.omega_lambda                              #omega lambda
boxsize_in_cMpc_h = ds.domain_width.in_units('Mpccm/h')[0] #boxsize in [cMpc/h]
boxsize_in_kpc = ds.domain_width.in_units('kpc')[0] #boxsize in [kpc]


print(ds.field_list)


slc = yt.SlicePlot(
                    ds, "z", ('ramses', 'xHI'),
                    center=[0.49118094, 0.49275361, 0.49473726],
                    width=(1500, 'pc'),
                    buff_size=(1000, 1000))
slc.save()

slc = yt.SlicePlot(
                    ds, "z", ('gas', 'my_temperature'),
                    center=[0.49118094, 0.49275361, 0.49473726],
                    width=(1500, 'pc'),
                    buff_size=(1000, 1000))
slc.save()

slc = yt.SlicePlot(
                    ds, "z", ('gas', 'temperature'),
                    center=[0.49118094, 0.49275361, 0.49473726],
                    width=(1500, 'pc'),
                    buff_size=(1000, 1000))
slc.save()

slc = yt.SlicePlot(
                    ds, "z", ('ramses-rt', 'Photon_density_2'),
                    center=[0.49118094, 0.49275361, 0.49473726],
                    width=(1500, 'pc'),
                    buff_size=(1000, 1000))
slc.save()