import numpy as np

filenames=['nebula.linesPredictionO', 'nebula.linesPredictionCN']
titls=[["H1 6562.80A","O1 1304.86A","O1 6300.30A","O2 3728.80A","O2 3726.10A","O3 1660.81A","O3 1666.15A","O3 4363.21A","O3 4958.91A","O3 5006.84A"], ["He2 1640.41A","C2 1335.66A","C3 1906.68A","C3 1908.73A","C4 1549.00A","Mg2 2795.53A","Mg2 2802.71A","Ne3 3868.76A","Ne3 3967.47A","N5 1238.82A","N5 1242.80A","N4 1486.50A","N3 1749.67A","S2 6716.44A","S2 6730.82A"]]

ncols=[len(titls[0]), len(titls[1])]
cols=[np.arange(ncols[0])+2, np.arange(ncols[1])+2]
         
minU,maxU,stepU,minN,maxN,stepN,minT,maxT,stepT=np.loadtxt(filenames[0],unpack=True,dtype=float, max_rows=1)
linelists=[np.transpose(np.loadtxt(filenames[0],unpack=True,dtype=float,usecols=cols[0],skiprows=1)),np.transpose(np.loadtxt(filenames[1],unpack=True,dtype=float,usecols=cols[1],skiprows=1))]

#print(linelists[0].shape)
#print(linelists[1].shape)

linelist=np.hstack((linelists[0], linelists[1]))

titlarr=np.concatenate((titls[0], titls[1]))
#print(titlarr)
#print(titlarr.shape)

#linelist=np.vstack((titlarr, linelist))
#print(linelist)
#print(linelist.shape)


# Savetxt, add line with parameter mins, maxs, steps -> combined table for interpolation
with open(filenames[0], 'r') as file:
    runparams=file.readline()

np.savetxt('linelist.dat', linelist, header=np.array2string(titlarr) + '\n' + runparams)
