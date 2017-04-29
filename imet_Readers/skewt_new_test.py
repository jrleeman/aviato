
#from matplotlib.axes import Axes
#from matplotlib.lines import Line2D
#from matplotlib.collections import LineCollection
#from matplotlib.ticker import FixedLocator, AutoLocator, ScalarFormatter
#import matplotlib.transforms as transforms
#import matplotlib.axis as maxis
#import matplotlib.artist as artist
#from matplotlib.projections import register_projection

import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metpy.plots import SkewT, Hodograph
from metpy.units import units
from metpy.calc import get_wind_components,thermo
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from scipy import interpolate
import sharppy
import sharppy.sharptab.profile as profile
import sharppy.sharptab.interp as interp
import sharppy.sharptab.winds as winds
import sharppy.sharptab.utils as utils
import sharppy.sharptab.params as params
import sharppy.sharptab.thermo as thermo
import warnings
warnings.filterwarnings('ignore')

def var_read(type):
    real_data=True
    need_orig_array=True
    while(real_data):
        #data=re.split('/|\s+|\r\n',type.readline())
        data=re.findall(r"[-+]?\d*\.\d+|\d+",type.readline())
        data = [float(i) for i in data]
        if len(data)<1:
            real_data=False
        else:
            if need_orig_array:
                data_array=np.asarray(data)
                need_orig_array=False
            else:
                temp_data_array=np.asarray(data)
                data_array = np.vstack((data_array,temp_data_array))
    return(data_array)

#Read the imetOS2 data
def read_imet(lid,ascent):
    #if (lid < 100):
    #   slid = "0"+str(lid)
    #else:
    #   slid = str(lid)
    slid    = str(lid)
    sascent = str(ascent)
    stdf = "/var/www/html/soundings/launches/"+slid+"_"+sascent+"/"+slid+"_"+sascent+"_STDLVLS.txt"
    sigf = "/var/www/html/soundings/launches/"+slid+"_"+sascent+"/"+slid+"_"+sascent+"_SIGLVLS.txt"
    sumf = "/var/www/html/soundings/launches/"+slid+"_"+sascent+"/"+slid+"_"+sascent+"_SUMMARY.txt"

    #READ STANDARD LEVEL DATA
    std = open(stdf, 'r')
    for i in range(4): #Skip first 4 lines
        std.readline()
    data_std = var_read(std)

    #READ SIGNIFICANT TEMPERATURE/HUMIDITY LEVELS
    sig = open(sigf, 'r')
    test=0
    temp=True
    while temp:
        test=test+1
        text= sig.readline()
        if "TEMPERATURE" in text:
            break
        if test==200:
            break
    for i in range(5): #Skip first 4 lines
        sig.readline()
    #print sig.readline()
    data_sig_temp = var_read(sig)
    sig.close()

    #READ SIGNIFICANT WIND LEVELS
    sig = open(sigf, 'r')
    test=0
    wind=True
    while wind:
        test=test+1
        text= sig.readline()
        if "SIGNIFICANT WIND" in text:
            break
        if test==200:
            break
    for i in range(4): #Skip first 4 lines
        sig.readline()
    #print sig.readline()
    data_sig_wind = var_read(sig)
    sig.close()

    #READ SUMMARY FILE
    f = open(sumf, 'r')
    summary=True
    while summary:
        test=test+1
        text= f.readline()
        if "Launched" in text:
            break
        if test==200:
            break
    data_time = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    f.close()
    
    data_sig = np.vstack((data_sig_wind,data_sig_temp))
    data_sig = data_sig[data_sig[:,0].argsort()][::-1]

    return(data_std, data_sig, data_time)

#Remove duplicate rows from array
def unique(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return a[ui]

def thetas(theta, presvals):
    return ((theta + thermo.ZEROCNK) / (np.power((1000. / presvals),thermo.ROCP))) - thermo.ZEROCNK


launch = sys.argv[1]
ascent = sys.argv[2]

#p,h,T,Td,direc,spd,tdata = read_imet(launch,ascent)
data_std, data_sig, tdata = read_imet(launch,ascent)
data_all = np.vstack((data_sig,data_std))
data_all = data_all[data_all[:,0].argsort()][::-1]
p = data_all[:,0]
p_std = data_std[:,0]
T = data_all[:,1]
Td = data_all[:,2]
RH = data_all[:,3]
#Td = 243.04*(np.log(RH/100)+((17.625*T)/(243.04+T)))/(17.625-np.log(RH/100)-((17.625*T)/(243.04+T)))
h = data_all[:,4]
h = [i-np.min(h) for i in h] #reduce to ground level
spd = data_all[:,5]
spd_std = data_std[:,5]
direc = data_all[:,6]
direc_std = data_std[:,6]

prof = profile.create_profile(profile='default', pres=p, hght=h, tmpc=T,dwpc=Td, wspd=spd,
wdir=direc,strictQC=False)

#interpolate pressure to important height levels
h_new = [1000]+[3000,6000,9000,12000,15000]
for i in range(len(h_new)):
    if np.max(h)>h_new[i]:
        index=i
h_new_labels = ['1 km','3 km','6 km','9 km','12 km','15 km']
h_new_labels = h_new_labels[0:index+1]
p_interped_func = interpolate.interp1d(h, p)
p_interped = p_interped_func(h_new[0:index+1])

# Add units to the data arrays
p = p * units.mbar
p_std = p_std * units.mbar
T = T * units.degC
Td = Td * units.degC
spd = spd * units.knot
spd_std = spd_std * units.knot
direc = direc * units.deg
direc_std = direc_std * units.deg

# Convert wind speed and direction to components
u, v = get_wind_components(spd, direc)
u_std, v_std = get_wind_components(spd_std, direc_std)

#PARCEL CALCULATIONS with sharppy
sfcpcl = params.parcelx( prof, flag=1 ) # Surface Parcel
fcstpcl = params.parcelx( prof, flag=2 ) # Forecast Parcel
mupcl = params.parcelx( prof, flag=3 ) # Most-Unstable Parcel
mlpcl = params.parcelx( prof, flag=4 ) # 100 mb Mean Layer Parcel

sfc = prof.pres[prof.sfc]
p3km = interp.pres(prof, interp.to_msl(prof, 3000.))
p6km = interp.pres(prof, interp.to_msl(prof, 6000.))
p1km = interp.pres(prof, interp.to_msl(prof, 1000.))
mean_3km = winds.mean_wind(prof, pbot=sfc, ptop=p3km)
sfc_6km_shear = winds.wind_shear(prof, pbot=sfc, ptop=p6km)
sfc_3km_shear = winds.wind_shear(prof, pbot=sfc, ptop=p3km)
sfc_1km_shear = winds.wind_shear(prof, pbot=sfc, ptop=p1km)
srwind = params.bunkers_storm_motion(prof)
srh3km = winds.helicity(prof, 0, 3000., stu = srwind[0], stv = srwind[1])
srh1km = winds.helicity(prof, 0, 1000., stu = srwind[0], stv = srwind[1])

stp_fixed = params.stp_fixed(sfcpcl.bplus, sfcpcl.lclhght, srh1km[0], utils.comp2vec(sfc_6km_shear[0], sfc_6km_shear[1])[1])
ship = params.ship(prof)
eff_inflow = params.effective_inflow_layer(prof)
ebot_hght = interp.to_agl(prof, interp.hght(prof, eff_inflow[0]))
etop_hght = interp.to_agl(prof, interp.hght(prof, eff_inflow[1]))
effective_srh = winds.helicity(prof, ebot_hght, etop_hght, stu = srwind[0], stv = srwind[1])
ebwd = winds.wind_shear(prof, pbot=eff_inflow[0], ptop=eff_inflow[1])
ebwspd = utils.mag( ebwd[0], ebwd[1] )
scp = params.scp(mupcl.bplus, effective_srh[0], ebwspd)
stp_cin = params.stp_cin(mlpcl.bplus, effective_srh[0], ebwspd, mlpcl.lclhght, mlpcl.bminus)

indices = {'SBCAPE': [int(sfcpcl.bplus), 'J/kg'],\
           'SBCIN': [int(sfcpcl.bminus), 'J/kg'],\
           'SBLCL': [int(sfcpcl.lclhght), 'm AGL'],\
           'SBLFC': [int(sfcpcl.lfchght), 'm AGL'],\
           'SBEL': [int(sfcpcl.elhght), 'm AGL'],\
           'SBLI': [int(sfcpcl.li5), 'C'],\
           'MLCAPE': [int(mlpcl.bplus), 'J/kg'],\
           'MLCIN': [int(mlpcl.bminus), 'J/kg'],\
           'MLLCL': [int(mlpcl.lclhght), 'm AGL'],\
           'MLLFC': [int(mlpcl.lfchght), 'm AGL'],\
           'MLEL': [int(mlpcl.elhght), 'm AGL'],\
           'MLLI': [int(mlpcl.li5), 'C'],\
           'MUCAPE': [int(mupcl.bplus), 'J/kg'],\
           'MUCIN': [int(mupcl.bminus), 'J/kg'],\
           'MULCL': [int(mupcl.lclhght), 'm AGL'],\
           'MULFC': [int(mupcl.lfchght), 'm AGL'],\
           'MUEL': [int(mupcl.elhght), 'm AGL'],\
           'MULI': [int(mupcl.li5), 'C'],\
           '0-1 km SRH': [int(srh1km[0]), 'm2/s2'],\
           '0-1 km Shear': [int(utils.comp2vec(sfc_1km_shear[0], sfc_1km_shear[1])[1]), 'kts'],\
           '0-3 km SRH': [int(srh3km[0]), 'm2/s2'],\
           'Eff. SRH': [int(effective_srh[0]), 'm2/s2'],\
           'EBWD': [int(ebwspd), 'kts'],\
           'PWV': [round(params.precip_water(prof), 2), 'inch'],\
           'K-index': [int(params.k_index(prof)), ''],\
           'STP(fix)': [round(stp_fixed, 1), ''],\
           'SHIP': [round(ship, 1), ''],\
           'SCP': [round(scp, 1), ''],\
           'STP(cin)': [round(stp_cin, 1), '']}

# Set the parcel trace to be plotted as the Most-Unstable parcel.
pcl = mupcl

# Create a new figure. The dimensions here give a good aspect ratio
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='skewx')
ax.grid(True)

pmax = 1000
pmin = 10
dp = -10
presvals = np.arange(int(pmax), int(pmin)+dp, dp)

# plot the moist-adiabats
for t in np.arange(-10,45,5):
    tw = []
    for p in presvals:
        tw.append(thermo.wetlift(1000., t, p))
    ax.semilogy(tw, presvals, 'k-', alpha=.2)

def thetas(theta, presvals):
    return ((theta + thermo.ZEROCNK) / (np.power((1000. / presvals),thermo.ROCP))) - thermo.ZEROCNK

# plot the dry adiabats
for t in np.arange(-50,110,10):
    ax.semilogy(thetas(t, presvals), presvals, 'r-', alpha=.2)

plt.title(' OAX 140616/1900 (Observed)', fontsize=12, loc='left')
# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dicatated by the typical meteorological plot
ax.semilogy(prof.tmpc, prof.pres, 'r', lw=2) # Plot the temperature profile
ax.semilogy(prof.wetbulb, prof.pres, 'c-') # Plot the wetbulb profile
ax.semilogy(prof.dwpc, prof.pres, 'g', lw=2) # plot the dewpoint profile
ax.semilogy(pcl.ttrace, pcl.ptrace, 'k-.', lw=2) # plot the parcel trace 
# An example of a slanted line at constant X
l = ax.axvline(0, color='b', linestyle='--')
l = ax.axvline(-20, color='b', linestyle='--')

# Plot the effective inflow layer using blue horizontal lines
ax.axhline(eff_inflow[0], color='b')
ax.axhline(eff_inflow[1], color='b')

#plt.barbs(10*np.ones(len(prof.pres)), prof.pres, prof.u, prof.v)
# Disables the log-formatting that comes with semilogy
ax.yaxis.set_major_formatter(plt.ScalarFormatter())
ax.set_yticks(np.linspace(100,1000,10))
ax.set_ylim(1050,100)
ax.xaxis.set_major_locator(plt.MultipleLocator(10))
ax.set_xlim(-50,50)

# List the indices within the indices dictionary on the side of the plot.
string = ''
for key in np.sort(indices.keys()):
    string = string + key + ': ' + str(indices[key][0]) + ' ' + indices[key][1] + '\n'
plt.text(1.02, 1, string, verticalalignment='top', transform=plt.gca().transAxes)

# Draw the hodograph on the Skew-T.
# TAS 2015-4-16: hodograph doesn't plot for some reason ...
ax2 = plt.axes([.625,.625,.25,.25])
below_12km = np.where(interp.to_agl(prof, prof.hght) < 12000)[0]
u_prof = prof.u[below_12km]
v_prof = prof.v[below_12km]
ax2.plot(u_prof[~u_prof.mask], v_prof[~u_prof.mask], 'k-', lw=2)
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
for i in range(10,90,10):
    # Draw the range rings around the hodograph.
    circle = plt.Circle((0,0),i,color='k',alpha=.3, fill=False)
    ax2.add_artist(circle)
ax2.plot(srwind[0], srwind[1], 'ro') # Plot Bunker's Storm motion right mover as a red dot
ax2.plot(srwind[2], srwind[3], 'bo') # Plot Bunker's Storm motion left mover as a blue dot

ax2.set_xlim(-60,60)
ax2.set_ylim(-60,60)
ax2.axhline(y=0, color='k')
ax2.axvline(x=0, color='k')

plt.savefig('/var/www/html/soundings/launches/'+launch+'_'+ascent+'/'+launch+'_'+ascent+'_KVUM.png',dpi=150)
#plt.show()
