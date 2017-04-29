
#from matplotlib.axes import Axes
#from matplotlib.lines import Line2D
#from matplotlib.collections import LineCollection
#from matplotlib.ticker import FixedLocator, AutoLocator, ScalarFormatter
#import matplotlib.transforms as transforms
#import matplotlib.axis as maxis
#import matplotlib.artist as artist
#from matplotlib.projections import register_projection

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

def var_read(type,p,tm,h,T,rh,Td,dirc,sped):
    dread = True
    while(dread):
        dummy = type.readline().split()
        if not dummy:
           dread = False
        else:
           p.append(dummy[0])
           tm.append(dummy[1])
           h.append(dummy[2])
           T.append(dummy[3])
           rh.append(dummy[4])
           Td.append(dummy[5])
           dirc.append(dummy[6])
           sped.append(dummy[7])
    return(p,tm,h,T,rh,Td,dirc,sped)

def read_imet(lid,ascent):
    import re
    #if (lid < 100):
    #   slid = "0"+str(lid)
    #else:
    #   slid = str(lid)
    slid    = str(lid)
    sascent = str(ascent)
    stdf = "/home/kgoebber/http/soundings/launches/"+slid+"_"+sascent+"/"+slid+"_"+sascent+".STD"
    sigf = "/home/kgoebber/http/soundings/launches/"+slid+"_"+sascent+"/"+slid+"_"+sascent+".SIG"
    #stdf = slid+"_00"+sascent+".STD"
    #sigf = slid+"_00"+sascent+".SIG"
    std = open(stdf, 'r')
    sig = open(sigf, 'r')

    # Empty arrays for sounding variables
    p    = []
    tm   = []
    h    = []
    T    = []
    rh   = []
    Td   = []
    dirc = []
    sped = []

    # Read Standard Level Data
    tdata = re.split('/|\s+|\r\n',std.readline())
    for i in range(19): #Skip next 19 lines
        std.readline()
    p,tm,h,T,rh,Td,dirc,sped = var_read(std,p,tm,h,T,rh,Td,dirc,sped)
    # Read Significant Level Data
    for i in range(22):
        sig.readline()
    p,tm,h,T,rh,Td,dirc,sped = var_read(sig,p,tm,h,T,rh,Td,dirc,sped)

    data = np.empty((len(p),6))
    data[:,0] = p
    data[:,1] = h
    data[:,2] = T
    data[:,3] = Td
    data[:,4] = dirc
    data[:,5] = sped
    data=data[data[:,0].argsort()][::-1]

    return(data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],tdata)

import sys

launch = sys.argv[1]
ascent = sys.argv[2]

p,h,T,Td,direc,spd,tdata = read_imet(launch,ascent)
h = [i-np.min(h) for i in h]
prof = profile.create_profile(profile='default', pres=p, hght=h, tmpc=T,dwpc=Td, wspd=spd, wdir=direc,strictQC=False)

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
T = T * units.degC
Td = Td * units.degC
spd = spd * units.knot
direc = direc * units.deg

# Convert wind speed and direction to components
u, v = get_wind_components(spd, direc)

# Create a new figure. The dimensions here give a good aspect ratio
fig = plt.figure(figsize=(6.5875, 6.2125))

# Grid for plots
skew = SkewT(fig)

#PARCEL CALCULATIONS with sharppy
sfcpcl = params.parcelx( prof, flag=1 ) # Surface Parcel
fcstpcl = params.parcelx( prof, flag=2 ) # Forecast Parcel
mupcl = params.parcelx( prof, flag=3 ) # Most-Unstable Parcel
mlpcl = params.parcelx( prof, flag=4 ) # 100 mb Mean Layer Parcel

# Set axis limits and color of major grid lines
skew.ax.set_xlim(-50,50)
skew.ax.set_ylim(1020,100)
#skew.ax.set_xlim(-30,35)
#skew.ax.set_ylim(1020,300)
skew.ax.yaxis.grid(b=True,which='major',color='k',linestyle='-',linewidth=0.5, alpha=0.5)
skew.ax.xaxis.grid(b=True,which='major',color='r',linestyle='--',linewidth=0.5, alpha=0.4)
plt.subplots_adjust(right=0.87)

# Plot important lines
dry_adiabat_temprange = np.arange(-80,200,10)
dry_adiabat_temprange = dry_adiabat_temprange * units.degC
moist_adiabat_presrange = np.linspace(1020,100,100)
moist_adiabat_presrange = moist_adiabat_presrange * units.mbar
skew.plot_dry_adiabats(t0=dry_adiabat_temprange,linestyle='-',colors='#D2691E', alpha=0.4)
skew.plot_moist_adiabats(p=moist_adiabat_presrange,linestyle='-',colors='g',alpha=0.4)
skew.plot_mixing_lines(colors='b',linewidth=0.75, alpha=0.5)
plt.axvline(0,color='r',linestyle='--',linewidth=0.75)
plt.axvline(-20,color='r',linestyle='--',linewidth=0.75)

# Set Title for SkewT image
plt.title('KVUM ', loc='left', fontsize=12)
plt.title(tdata[4]+'/'+tdata[3]+'/'+tdata[5], loc='center', fontsize=12)
plt.title(tdata[7]+' UTC', loc='right', fontsize=12)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r',linewidth=1.5)
skew.plot(p, Td, 'g',linewidth=1.5)
skew.plot(mupcl.ptrace,mupcl.ttrace, 'k-.')

#skew.ax.fill_betweenx(p,T,mupcl.ttrace)

ax2 = skew.ax.twinx()
plt.yscale('log', nonposy='clip')
ax2.set_xlim(-50,50)
ax2.set_ylim(1020,100)
plt.yticks(p_interped,h_new_labels,color='r', ha='left')
ax2.yaxis.tick_left()
ax2.tick_params(direction='in', pad=-5)

#PLOT PARCEL STUFF
plt.plot((37, 43), (mupcl.lfcpres,mupcl.lfcpres), 'r-',lw=1.5)
ax2.annotate('LFC', xy=(40, mupcl.lfcpres-10), xytext=(40, mupcl.lfcpres-10),ha='center', color='r')
plt.plot((37, 43), (mupcl.lclpres,mupcl.lclpres), 'g-',lw=1.5)
ax2.annotate('LCL', xy=(40, mupcl.lclpres+50), xytext=(40, mupcl.lclpres+50),ha='center', color='g')

# Create windbarbs and hodograph
skew.plot_barbs(p, u, v, xloc=1.1)
hgt_list = [0,3000,6000,9000,np.max(h)]
hodo_color = ['r','g','y','c']
hodo_label = ['0-3km','3-6km','6-9km','>9km']
ax_hod = inset_axes(skew.ax, '40%', '40%', loc=1)
for tick in ax_hod.xaxis.get_major_ticks():
    tick.label.set_fontsize(10)
    tick.label.set_rotation(45)
for tick in ax_hod.yaxis.get_major_ticks():
    tick.label.set_fontsize(10) 
hodo = Hodograph(ax_hod, component_range=80.)
hodo.add_grid(increment=20)
for k in range(len(hgt_list)-1):
    index1 = min(range(len(h)), key=lambda i: abs(h[i]-hgt_list[k]))
    index2 = min(range(len(h)), key=lambda i: abs(h[i]-hgt_list[k+1]))
    hodo.plot(u[index1:index2+1],v[index1:index2+1],c=hodo_color[k],linewidth=2.0,label=hodo_label[k])
ax_hod.legend(loc=2,prop={'size':8})

plt.savefig('/home/kgoebber/http/soundings/launches/'+launch+'_'+ascent+'/'+launch+'_'+ascent+'_KVUM.png',dpi=150)
#plt.show()
