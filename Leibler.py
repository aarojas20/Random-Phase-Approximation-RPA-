# Random-Phase-Approximation-RPA-
# This script performs a random phase approximation in python for x-ray scattering data
# the user should calculate the number of sites occupied by each polymeric domain in the block copolymer (Naref and Nbref), 
# the scattering length densities for each segment (Ba and Bb)
# this code will use a robust linear-squares algorithm to fit a Leibler function to data
#-------------------------------------------------------------------
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import numpy as np
import csv
import pprint
#-------------------------------------------------------------------------------------------------------
# read the file containing the data
file='C:\\Users\\Balsara Lab\\My Documents\\scattering_data_90C.csv'
infile=open(file,'r')
table=[]
table=[row for row in csv.reader(infile)]
infile.close()
qdata=[]
Idata=[]
#now transform string objects into float
for r in range(1,len(table)):
    for c in range(0,len(table[0])):
        table[r][c]=float(table[r][c])
    qdata.append(table[r][0])    # assign the q-values from the file to a variable list named qdata
    Idata.append(table[r][1])       # assign the intensity data from the file to a variable list named Idata
# convert the list to numpy arrays
qdata=np.array(qdata)
Idata=np.array(Idata)
#-----------------------------------------------------------------------------------------------------
# define a function for the background fit
def bkd(q,a,b,c):
    return a+b*q**(-1*c)
#---------------------------------------------------------------
# identify the region to fit for the background 
qdata_bkd=[]
Idata_bkd=[]
for i in range(0,70):
    qdata_bkd.append(qdata[i])
    Idata_bkd.append(Idata[i])
for i in range(700,1060):
    qdata_bkd.append(qdata[i])
    Idata_bkd.append(Idata[i])
qdata_bkd=np.array(qdata_bkd)
Idata_bkd=np.array(Idata_bkd)
#-----------------------------------------------------------------------------------------------------
# define the function computing residuals for least-squares minimization
# in the leibler fit
def func(x,q,I):
    # x is an array where x[0] = Rgsq, x[1] = Xab, x[2] = A, x[3] = B, and x[4] = C
    # where Rgsq is the radius of gyration squared, Xab is the flory hugins interaction paramter
    # and where A, B, C are constants in the background fit A+B*q**(-C)
    # definitions of some constants
    # let 'a' stand for PEO monomer, let 'b' stand for PS monomer
    #
    vref=10**(-22)                  # the reference volume in cm^3 
    Naref=78                        # the number of sites occupied by the PEO 5000 g/mol chain at 90 C
    Nbref=21                        # the number of sites occupied by the PSTFSIMg chain (2000 g/mol)
    Nref=Naref+Nbref                # the total number of sites
    Phia=Naref/Nref                 # the volume fraction of the "a" block (the PEO)
    Phib=Nbref/Nref                 # the volume fraction of the "b" block
    D=1.15                          # the dispersity of the block copolymer chain
    k=1/(D-1)                       # the dispersity index
    Ba=9.87*10**10                   # scattering length density of PEO 5000 g/mol
    Bb=1.36*10**11                  # scattering length density of PSTFSIMg (2000 g/mol)
    #laref=0.72                     # the statistical segment length of PEO monomer
    u=q**2*x[0]                     # calculate u
    #---------------------------------------------------------
    #define the form factors
    P1=(2/u**2)*(u-1+(k/(k+u))**k)
    Pa=(2/u**2)*(Phia*u-1+(k/(k+Phia*u))**k)
    Pb=(2/u**2)*(Phib*u-1+(k/(k+Phib*u))**k)
    # define the structure factors
    Saa=Nref*Pa
    Sbb=Nref*Pb
    Sab=Nref/2*(P1-Pa-Pb)
    # the leibler structure factor
    Saa_leibler=((Saa+Sbb+2*Sab)/(Saa*Sbb-Sab**2)-2*x[1])**(-1)
    # the instensity for the leibler fit
    I_leibler=((Ba-Bb)**2*vref*Saa_leibler)
#    Ibkd=x[2]+x[3]*np.exp(-1*x[4]*q) # you can use either an exponential or power law for the background
    Ibkd=x[2]+x[3]*q**(-1*x[4])
    Itot=I_leibler + Ibkd
    return Itot-I
#-----------------------------------------------------------------------------------------------------
# define a function to contain the leibler model
def mod(q,Rgsq,Xab,A,B,C):
    # definitions of some constants
    # let a stand for PEO monomer, let b stand for PS monomer
    #
    vref=10**(-22)                  # the reference volume in nm^3 
    Naref=78                       # the number of sites occupied by the PEO 
    Nbref=21                        # the number of sites occupied by the PSTFSI 
    Nref=Naref+Nbref
    Phia=Naref/Nref
    Phib=Nbref/Nref
    D=1.15                          # the dispersity of the block copolymer chain
    k=1/(D-1)                       # the dispersity index
    Ba=9.87*10**10                  # scattering length density of PEO
    Bb=1.36*10**11                 # scattering length density of PSTFSIMg 
    u=q**2*Rgsq                     # calculate u
    #---------------------------------------------------------
    #define the form factors
    P1=(2/u**2)*(u-1+(k/(k+u))**k)
    Pa=(2/u**2)*(Phia*u-1+(k/(k+Phia*u))**k)
    Pb=(2/u**2)*(Phib*u-1+(k/(k+Phib*u))**k)
    # define the structure factors
    Saa=Nref*Pa
    Sbb=Nref*Pb
    Sab=Nref/2*(P1-Pa-Pb)
    # the leibler structure factor
    Saa_leibler=((Saa+Sbb+2*Sab)/(Saa*Sbb-Sab**2)-2*Xab)**(-1)
    # the instensity for the leibler fit
    I_leibler=((Ba-Bb)**2*vref*Saa_leibler)
#    Ibkd=A+B*np.exp(-1*C*q)
    Ibkd=A+B*q**(-1*C)
    Itot=I_leibler + Ibkd
    return Itot
#-----------------------------------------------------------------------------------------------------
# define a function to contain the leibler model
def leib(q,Rgsq,Xab):
    # definitions of some constants
    # let a stand for PEO monomer, let b stand for PS monomer
    #
    vref=10**(-22)                  # the reference volume in nm^3 
    Naref=78                       # the number of sites occupied by the PEO 9500 g/mol chain at 90 C
    Nbref=21                        # the number of sites occupied by the PSTFSI chain (3600 g/mol)
    Nref=Naref+Nbref
    Phia=Naref/Nref
    Phib=Nbref/Nref
    D=1.15                         # the dispersity of the block copolymer chain
    k=1/(D-1)                       # the dispersity index
    Ba=9.87*10**10                  # scattering length density of PEO 9500 g/mol
    Bb=1.36*10**11                  # scattering length density of PSTFSIMg (36000 g/mol)
    u=q**2*Rgsq                     # calculate u
    #---------------------------------------------------------
    #define the form factors
    P1=(2/u**2)*(u-1+(k/(k+u))**k)
    Pa=(2/u**2)*(Phia*u-1+(k/(k+Phia*u))**k)
    Pb=(2/u**2)*(Phib*u-1+(k/(k+Phib*u))**k)
    # define the structure factors
    Saa=Nref*Pa
    Sbb=Nref*Pb
    Sab=Nref/2*(P1-Pa-Pb)
    # the leibler structure factor
    Saa_leibler=((Saa+Sbb+2*Sab)/(Saa*Sbb-Sab**2)-2*Xab)**(-1)
    # the instensity for the leibler fit
    I_leibler=((Ba-Bb)**2*vref*Saa_leibler)
    return I_leibler
#-----------------------------------------------------------------------------------
# identify the region to fit for the background 
#qdata_bkd=[]
#Idata_bkd=[]
#for i in range(5,30):
#    qdata_bkd.append(qdata[i])
#    Idata_bkd.append(Idata[i])
#for i in range(200,300):
#    qdata_bkd.append(qdata[i])
#    Idata_bkd.append(Idata[i])
#qdata_bkd=np.array(qdata_bkd)
#Idata_bkd=np.array(Idata_bkd)
#-----------------------------------------------------------------------------------------------------
plt.figure(figsize=(7,5),dpi=300)
#now graph the rest of the data
plt.loglog(qdata,Idata,'o',color='k',label='(5.0-2.0) (Mg$^{2+}$) diblock',markersize=4)              # this graphs the data
# identify the region to fit for leibler, ie, "just the hump"
qdata_leib=[]
Isub_leib=[]
for i in range(1,70):
    qdata_leib.append(qdata[i])
    Isub_leib.append(Idata[i])
qdata_leib=np.array(qdata_leib)
Isub_leib=np.array(Isub_leib)
#print(len(qdata_leib),len(Isub_leib))
#------------------------------------------------------------
#identify the initial guesses for x0
x0=[26,.18,.04,.0294,2.79]
bds=([0, 0.01, .001, .02, 2.128],[50,10,np.inf,np.inf,np.inf])  # this constrains the bounds for the fit parameters        
# run robust least squares with loss='soft_l1', or loss='arctan',set f_scale to 0.1 which means
# that inlier residuals are approximately lower than 0.1
res_robust=least_squares(func,x0,loss='soft_l1', f_scale=.1, bounds=bds,args=(qdata_leib,Isub_leib))
print('The results from the fit to the Leibler model')
print('Rgsq=', res_robust.x[0],', Xab=',res_robust.x[1])
print('The results from the fit to the background using the total model')
print('A=', res_robust.x[2],', B=',res_robust.x[3], ', C=',res_robust.x[4])
plt.semilogy(qdata,mod(qdata,*res_robust.x),color='darkviolet',label='Total Model')
#-----------------------------------------------------------------------------------------------------
#plt.plot(qdata_bkd,Idata_bkd,'yo',label='bkd')      # graphs the bkd region for fit
plt.plot(qdata_leib,Isub_leib,color='orange',label='data for total model fit')    # this graphs the region for leib fit
plt.plot(qdata,bkd(qdata,res_robust.x[2],res_robust.x[3],res_robust.x[4]),'r--',label='Power law fit')    # this graphs the background fit
Ileib=mod(qdata,*res_robust.x)-bkd(qdata,res_robust.x[2],res_robust.x[3],res_robust.x[4])
#plt.plot(qdata,Ileib,color='blue',label='Leibler model')
plt.plot(qdata,leib(qdata,43.98,.1997),color='gray',label='Leibler')
plt.xlabel('q (nm$^{-1}$)', fontsize=14)
plt.ylabel('Intensity (cm $^{-1}$)', fontsize=14)
plt.tick_params(labelsize=14)
plt.tick_params(which='major',right='on',direction='in',top='on',length=6)
plt.tick_params(which='minor',right='on',direction='in',top='on',length=3)
plt.axis([0.04,1.5,.01,200])
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
