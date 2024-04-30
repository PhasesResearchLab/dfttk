import sys, os
import numpy as np
import matplotlib.pyplot as plt
import copy 
import math
import csv
from scipy.optimize import fsolve
from scipy.optimize import leastsq
from scipy.optimize import fmin_cobyla
#from lib_shunlieos import eosparameter45, build_eos, murnaghan, vineteos, morseeos, pvbuildeos, pv2prop, pveosfit

'''
% -----------------------------------------------------------------------
%   3: 4-parameter                BM4   3 
% -----------------------------------------------------------------------
1. (energy.0) ==> lowest E0 ==> left side max 4(5) points; right side max 5(6) points 
2. min data points are 4 (5) 
3. delete points based on E-V points 

'''  
ipress =  1; ###% > 0, including the pressure data to compare
idata  =  2; #% 1: E-V data in one file ("ev.0"), 2: E-V data in 2 file (energy.0 and volume.0)
istrain=  1; #% > 0 plot the strain vs. volume relation
LL0    =  1; #% the no. of data point used as the reference state

ifigure = -9;      # > 0 plot figure
isave   = -9;      # > 0 save volume and fitted energy, pressure 
icluster = 9;     # in ACI cluster etc, no need to show figures plt.show() 
kkfig   =  9;      # > 0 four to five figures ONLY 
energy_select = "Y"

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
### functions  ###  up to line about 495  
def eosparameter45(xx, vzero, icase):
    chunit=160.2189;
    a=xx[0];  b=xx[1];  c=xx[2];  d=xx[3];  e=xx[4];
    global res1
##### to get the structure properties#####%%%%%%%

    if icase == 3:
        V =math.sqrt(-((4*c**3 - 9*b*c*d + math.sqrt((c**2 - 3*b*d)*(4*c**2 - 3*b*d)**2))/b**3));

    if icase == 4:
        func=lambda x:((8*e)/(3*x**(11/3)) + (2*d)/x**3 + (4*c)/(3*x**(7/3)) + (2*b)/(3*x**(5/3)))*chunit;
        V=fsolve(func,vzero);

    if icase == 3 or icase == 4:
        P=(8*e)/(3*V**(11/3)) + (2*d)/V**3 + (4*c)/(3*V**(7/3)) + (2*b)/(3*V**(5/3)); 
        P=P*chunit; 
        B =(2*(44*e + 27*d*V**(2/3) + 14*c*V**(4/3) + 5*b*V**2))/(9*V**(11/3));
        B = B*chunit;
        BP=(484*e + 243*d*V**(2/3) + 98*c*V**(4/3) + 25*b*V**2)/(132*e + 81*d*V**(2/3) + 42*c*V**(4/3) + 15*b*V**2);
        B2P =(4*V**(13/3)*(27*d*(22*e + 7*c*V**(4/3) + 10*b*V**2) + V**(2/3)*(990*b*e*V**(2/3) + 7*c*(176*e + 5*b*V**2))))/(44*e + 27*d*V**(2/3) + 14*c*V**(4/3) + 5*b*V**2)**3;
        B2P = B2P/chunit; 
        E0= a + e/V**(8/3) + d/V**2 + c/V**(4/3) + b/V**(2/3);
        res1=[V, E0, P, B, BP, B2P];

    return(res1)    





def energy_data():
    d=np.loadtxt('energy.0')
    min_indict=np.argmin(d)
    shape=len(d)
    st=0
    end=shape
    #print('min_indict',min_indict)
    if min_indict>5:
        st=min_indict-5
    if min_indict<shape-5:
        end=shape-5
    return st,end
def shortest_distance(point,para,V_range,ty):
    chunit=160.2189;
    a = para[0];
    b = para[1];
    c = para[2];
    d = para[3];
    e = 0.0;
    if ty=='ev':
        def func(x):
            return (a + b * (x) ** (-2 / 3) + c * (x) ** (-4 / 3) + d * (x) ** (-2) + e * (x) ** (-8 / 3));
    if ty=='pv':
        def func(x):
            return (((8 * e) / (3 * x ** (11 / 3)) + (2 * d) / x ** 3 + (4 * c) / (3 * x ** (7 / 3)) + (2 * b) / ( 3 * x ** (5 / 3))) * chunit);
    

    # def objective(X):
    #     x,y=X
    #     return np.sqrt((x-point[0])**2)
    # def c1(X):
    #     x,y=X
    #     return func(x)-y
    # def c2(X):
    #     x,y=X
    #     return y-func(x)
    # X=fmin_cobyla(objective,x0=V_range,cons=[c1,c2])    print('y-diff',point[1]-func(point[0]))

    return abs(point[1]-func(point[0]))
def data_select(ori_data,fzero,volume,data_tol,ty):
    fz=fzero[ori_data]
    vol=volume[ori_data]
    vol0=np.mean(volume)
    v_range=[min(vol)-1,max(vol)+2];
    bb = fz;
    AA = np.vstack((np.ones(np.shape(vol)), vol ** (-2 / 3), vol ** (-4 / 3), vol ** (-2)))
    AA = AA.T
    xx1 = np.linalg.pinv(AA);
    xx = xx1.dot(bb);  # (4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
    xx=np.append(xx,0)
    distance=[];
    data_res=eosparameter45(xx, vol0, 3)
    # print('data_res=',data_res[4])
    for i in range(len(vol)):
        distance.append(shortest_distance([vol[i],fz[i]],xx,v_range,ty))
        # print('distance=',distance)
    if len(distance)>=5:
        if data_res[4]<3 or data_res[4]>7 or max(distance)>data_tol:
            index=distance.index(max(distance))
            del ori_data[index]
            work_data=data_select(ori_data,fzero,volume,data_tol,ty)
    elif len(distance)<5:
        print('Not find 5 point in '+ty)
    return ori_data




#%%%%%%%%%%======================================================   
def build_eos(prop,L):
    #% By Shunli to find the fitted parameters a b c d
    #%  V0  E0  P0  B  BP  B2P  av_diff   max_diff                                                            
    #%  1   2   3   4  5   6     7        8                                                               
    global res2
    V    = prop[0];          #%  V0(A**3/atom) 
    E0   = prop[1];          #%  B0(GPa)   
    B    = prop[3];          #%  BP  
    bp   = prop[4];          #%  E0  (eV/atom) 
    b2p  = prop[5];          #%  b2p (1/GPa)
    changeunit = 1.0/160.21892;         #%  1 GPa = 1/160.22 eV/A**3 
    B    = B*changeunit;       #%  (nx1) B0 (eV/A**3) 
    b2p  = b2p/changeunit; #% A**3/eV

    
    if L==3 or L==4:
        a=(128*E0 + 3*B*(287 + 9*B*b2p - 87*bp + 9*bp**2)*V)/128;
        e=(3*B*(143 + 9*B*b2p - 63*bp + 9*bp**2)*V**(11/3))/128;
        d=(-3*B*(167 + 9*B*b2p - 69*bp + 9*bp**2)*V**3)/32;
        c=(9*B*(199 + 9*B*b2p - 75*bp + 9*bp**2)*V**(7/3))/64;
        b=(-3*B*(239 + 9*B*b2p - 81*bp + 9*bp**2)*V**(5/3))/32;
        if abs(e) < 1e-8:
            e=0;
        res2=[a, b, c, d, e];      

    
    return(res2)
    
## ==========================================
def pvbuildeos(prop, L):
#%  V0  P0  B  BP  B2P  av_diff   max_diff                                                            
#%  1   2   3   4  5     6           7
    V    = prop[0];          #%  V0[A]3/atom) 
    P0   = prop[1];          #%  P0[G]a)   
    B    = prop[2];          #%  B  GPa
    bp   = prop[3];          #%  BP  
    b2p  = prop[4];          #%  b2p  (1/GPa)


    if L==3 or L==4:
        e=(3*B*(143 + 9*B*b2p - 63*bp + 9*bp**2)*V**(11/3))/128;
        d=(-3*B*(167 + 9*B*b2p - 69*bp + 9*bp**2)*V**3)/32;
        c=(9*B*(199 + 9*B*b2p - 75*bp + 9*bp**2)*V**(7/3))/64;
        b=(-3*B*(239 + 9*B*b2p - 81*bp + 9*bp**2)*V**(5/3))/32;
        if abs(e) < 1e-8:
            e=0;     
        res1=[b, c, d, e];  
    return res1

## ==========================================
def pv2prop(xx, vzero, icase):
    chunit=160.2189;
    b=xx[0];  c=xx[1];  d=xx[2];  e=xx[3];
#%%%% to get the structure properties
#%%%%
    if icase == 3:
        V =np.sqrt(-((4*c**3 - 9*b*c*d + np.sqrt((c**2 - 3*b*d)*(4*c**2 - 3*b*d)**2))/b**3));

    if icase ==3 or icase == 4:
        P=(8*e)/(3*V**(11/3)) + (2*d)/V**3 + (4*c)/(3*V**(7/3)) + (2*b)/(3*V**(5/3)); 
        B =(2*(44*e + 27*d*V**(2/3) + 14*c*V**(4/3) + 5*b*V**2))/(9*V**(11/3));
        BP=(484*e + 243*d*V**(2/3) + 98*c*V**(4/3) + 25*b*V**2)/(132*e + 81*d*V**(2/3) + 42*c*V**(4/3) + 15*b*V**2);
        B2P =(4*V**(13/3)*(27*d*(22*e + 7*c*V**(4/3) + 10*b*V**2) + V**(2/3)*(990*b*e*V**(2/3) +  7*c*(176*e + 5*b*V**2))))/(44*e + 27*d*V**(2/3) + 14*c*V**(4/3) + 5*b*V**2)**3;
        res1=[V, P, B, BP, B2P];
        
    return res1

## ==========================================
def pveosfit(volume, fzero, pressure, vv, isave, ifigure, kkfig):#print
    chunit=160.2189;
    vzero=np.mean(volume); 
    res=[];  resdiff=[]; resxx=[]; respp=[];
    L=3

    qq = volume;
    bb = pressure;
    AA = [qq ** (-5 / 3) * (2 / 3), qq ** (-7 / 3) * (4 / 3), qq ** (-3) * 2];
    AA = np.array(AA)
    AA = AA.T
    xx1 = np.linalg.pinv(AA)
    xx = xx1.dot(bb);  # %(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
    b = xx[0];
    c = xx[1];
    d = xx[2];
    e = 0.0;
    xx = [b, c, d, e];
    x = vv;
    pp = (8 * e) / (3 * x ** (11 / 3)) + (2 * d) / x ** 3 + (4 * c) / (3 * x ** (7 / 3)) + (2 * b) / (3 * x ** (5 / 3));
    x = volume;
    pp0 = (8 * e) / (3 * x ** (11 / 3)) + (2 * d) / x ** 3 + (4 * c) / (3 * x ** (7 / 3)) + (2 * b) / (
                3 * x ** (5 / 3));
    if ifigure > 1:
        plt.plot(vv, pp, volume, pressure, 'o')
        plt.title('P-V FITTED curve, BM4, No. 3, GPa')
    diffp = pp0 - pressure;
    prop = pv2prop(xx, vzero, L);  # % [V0, P, B, BP, B2P];
    res = prop

    #print('res=',res)
    return res

#### end of functions ###
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


##########################################

if idata==1:
    data=np.loadtxt('ev.0'); 
    volume=data[:,1]; 
    fzero=data[:,2];

if idata ==2: 
    fzero   =np.loadtxt('energy.0');    #eV/UC
    volume  =np.loadtxt('volume.0');    # A^3  
    volume123=copy.deepcopy(volume); 
##print(np.shape(volume))
if ipress > 0:
    press0  =np.loadtxt('pressure.0');  # kBar
    pressure=press0/10;           # GPa

if ipress < 0:
    pressure=np.zeros(np.shape(volume));

nn1,nn2=energy_data()
datarange=list(range(nn1,nn2));
iselect = energy_select
data_tol=0.03*(max(fzero)-min(fzero))

if iselect == 'Y' or iselect == 'y':
    datarange1=copy.deepcopy(datarange)
    datarange2=copy.deepcopy(datarange)
    datarange =data_select(datarange1,fzero,volume,data_tol,'ev')   #%%%%%  <<<<<<<<<
    datarange_ev=datarange
    # print('datarange=',datarange)
    if len(datarange)<5:
        # print('datarange=',datarange2)
        datarange=data_select(datarange2,fzero,volume,10000,'pv')
        if len(datarange)<5:
            datarange=datarange_ev
w=np.savetxt('data_selection.txt',datarange,fmt='%i')

fzero=fzero[datarange]; 
volume=volume[datarange];
pressure=pressure[datarange];
vv=np.arange(min(volume)-1,max(volume)+2,0.05); # volume for the final-fited E-V

if istrain > 0:
    scales =np.loadtxt('scales.0');
    vectors=np.loadtxt('vectors.0');
    sss=[];
    L=LL0;
    mm0=3*(L-1)+1; 
    mm1=3*L;
    asd0=scales[L-1]*vectors[mm0-1:mm1,:];
    #print('asd0=',np.linalg.inv(asd0))
    #print()

for i in range(1,len(scales)+1):
    mm0=3*(i-1)+1; 
    mm1=3*(i);
    aa=scales[i-1]*vectors[mm0-1:mm1,:];
    zz1=np.linalg.inv(asd0);
    #print('zz1=', zz1*asd0)
    #zz=np.dot(zz1,aa);
    zz=zz1@aa
    #print('i, zz=', i, zz)
    #print()
    zz=np.reshape(zz, 9);  
    if len(sss)==0:
        sss=zz
    else:
        sss=np.vstack((sss, zz));
#print(volume123)
#print('sss=',sss)

if kkfig > 0 and istrain > 0: 
    fig1 = plt.figure('fig1')
    plt.plot(volume123, sss[:,0], '-o'); 
    plt.plot(volume123, sss[:,1], '-o');
    plt.plot(volume123, sss[:,2], '-o');
    plt.plot(volume123, sss[:,3], '-o');
    plt.plot(volume123, sss[:,4], '-o');
    plt.plot(volume123, sss[:,5], '-o');
    plt.plot(volume123, sss[:,6], '-o');
    plt.title('strain relation vs volume');  # of 'if istrain > 0'

#%%%%%%%%%%%%%%%%%
chunit=160.2189;
vzero=np.mean(volume); 
res=[];  
resdiff=[]; 
resee=[]; 
resxx=[]; 
resdiffpp=[]; 
respp=[];
#####################################
### BM4 only
L=3

bb = fzero;
AA = np.vstack((np.ones(np.shape(volume)), volume ** (-2 / 3), volume ** (-4 / 3), volume ** (-2)))
AA = AA.T
xx1 = np.linalg.pinv(AA);
xx = xx1.dot(bb);  # (4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
a = xx[0];
b = xx[1];
c = xx[2];
d = xx[3];
e = 0.0;
xx = [a, b, c, d, e];
# print('ori_pa',xx)
x = vv;
pp = ((8 * e) / (3 * x ** (11 / 3)) + (2 * d) / x ** 3 + (4 * c) / (3 * x ** (7 / 3)) + (2 * b) / ( 3 * x ** (5 / 3))) * chunit;
x = volume;
pp0 = ((8 * e) / (3 * x ** (11 / 3)) + (2 * d) / x ** 3 + (4 * c) / (3 * x ** (7 / 3)) + (2 * b) / ( 3 * x ** (5 / 3))) * chunit;
###############
ifigure=3

if ifigure > 1:
    fig4 = plt.figure('fig4')
    plt.plot(vv, pp, volume, pressure, 'o')
    plt.title('P-V curve, BM4, No. 3, GPa'),
diffp = pp0 - pressure;
#resdiffpp = np.vstack((resdiffpp, diffp));
ee = a + b * (vv) ** (-2 / 3) + c * (vv) ** (-4 / 3) + d * (vv) ** (-2) + e * (vv) ** (-8 / 3);

ifigure = 2
if ifigure > 1:
    fig5 = plt.figure('fig5')
    plt.plot(vv, ee, volume, bb, 'o')
    plt.title('E-V curve, BM4, No. 3')
curve = a + b * (volume) ** (-2 / 3) + c * (volume) ** (-4 / 3) + d * (volume) ** (-2) + e * (volume) ** (-8 / 3);
diff = curve - bb;
prop = eosparameter45(xx, vzero, L);  # [V0, E0, P, B, BP, B2P];
# print('prop =', prop)
newxx = build_eos(prop, L)
xxnp = np.array(xx)

#plt.show()

if ipress > 0:
    pvfit_res=pveosfit(volume, fzero, pressure, vv, isave, ifigure, kkfig);
    #print('pvfit_res=', pvfit_res)
