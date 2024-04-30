import sys, os
import numpy as np
import matplotlib.pyplot as plt
import copy 
import math
from scipy.optimize import fsolve
from scipy.optimize import leastsq
import argparse
#from lib_shunlieos import eosparameter45, build_eos, murnaghan, vineteos, morseeos, pvbuildeos, pv2prop, pveosfit

'''
% -----------------------------------------------------------------------
% EOS fittings by Shunli Shang at PSU, 06-Aug-2008
% translated from Shunli's matlab code, 3-15-2019
%
%  1:   4-parameter (Teter-Shang) mBM4  1 
%  2:   5-parameter (Teter-Shang) mBM5  2
%   3: 4-parameter                BM4   3 
%   4: 5-parameter                BM5   4 
%  5: 4-parameter Natural         Log4  5
%  6: 5-parameter Natural         Log5  6
%  7:  4-parameter Murnaghan      Mur   7
%  8:  4-parameter Vinet          Vinet 8 
%  9:  4-parameter Morse          Morse 9 
% get_0kres.m, includes function: eosfit.m
% -----------------------------------------------------------------------
'''  
ipress =  1; ###% > 0, including the pressure data to compare
idata  =  2; #% 1: E-V data in one file ("ev.0"), 2: E-V data in 2 file (energy.0 and volume.0)
istrain=  1; #% > 0 plot the strain vs. volume relation
LL0    =  1; #% the no. of data point used as the reference state
ifigure = -9;      # > 0 plot lots of figures; always set a negative value
isave   = -9;      # > 0 save volume and fitted energy, pressure

kkfig   =  9;      # > 0: to SAVE files to dask
#kkshow  =  9;      # > 0: to show figures on screen

parser = argparse.ArgumentParser()
parser.add_argument("-ishow", help="default < 0: donot show figures on screen", type=float, default=-9)
args = parser.parse_args()
kkshow = args.ishow
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

nn1=0;         #%%%%%%%%%%% <<<<<<<<<<<<<<<<<<<<<<<
nn2=999;       #%%%%%%%%%%% <<<<<<<<<<<<<<<<<<<<<<<
nn3=len(volume)
if nn3 < nn2:
    nn2=nn3;
    datarange=range(nn1,nn2);
iselect = -1;  #%%%%%%%%%% <<<<<<<<<<<<<<<<<<<<<<<
if iselect > 0:
    datarange =[2,  4, 5, 6, 7],   #%%%%%  <<<<<<<<<
if 'data_selection.txt' in os.listdir():
    datarange=np.loadtxt('data_selection.txt',dtype=int)

fzero=fzero[datarange]; 
volume=volume[datarange]; 
pressure=pressure[datarange];

vv=np.arange(min(volume)-1,max(volume)+2,0.05); # volume for the final-fited E-V
numbereos=9;        # 6 ( linear fittings);  9: all fittings
if numbereos==6:
    ieos=[1, 2, 3, 4, 5, 6]; 
if numbereos==9:
    ieos=[1, 2, 3, 4, 5, 6, 7, 8, 9];


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
### functions  ###  up to line about 495  

def eosparameter45(xx, vzero, icase):
    chunit=160.2189;
    a=xx[0];  b=xx[1];  c=xx[2];  d=xx[3];  e=xx[4];
    global res1
##### to get the structure properties#####%%%%%%%
    if icase == 1:
        V = 4*c**3 - 9*b*c*d + np.sqrt((c**2 - 3*b*d)*(4*c**2 -3*b*d)**2); V =-V/b**3;
		
    if icase == 2:
        func=lambda x:((4*e)/(3*x**(7/3)) + d/x**2 + (2*c)/(3*x**(5/3)) + b/(3*x**(4/3)))*chunit
        V=fsolve(func,vzero)

    if icase == 1 or icase == 2 :
        P=(4*e)/(3*V**(7/3)) + d/V**2 + (2*c)/(3*V**(5/3)) + b/(3*V**(4/3)); P=P*chunit; 
        B = ((28*e)/(9*V**(10/3)) + (2*d)/V**3 + (10*c)/(9*V**(8/3)) + (4*b)/(9*V**(7/3)))*V;  
        B = B*chunit;
        BP=(98*e + 54*d*V**(1/3) + 25*c*V**(2/3) + 8*b*V)/(42*e + 27*d*V**(1/3) + 15*c*V**(2/3) + 6*b*V); 
        B2P =(V**(8/3)*(9*d*(14*e + 5*c*V**(2/3) + 8*b*V) + 2*V**(1/3)*(126*b*e*V**(1/3) + 5*c*(28*e + b*V))))/(2*(14*e + 9*d*V**(1/3) + 5*c*V**(2/3) + 2*b*V)**3);
        B2P = B2P/chunit; 
        E0=a + b*V**(-1/3) + c*V**(-2/3)+ d*V**(-1) + e*V**(-4/3); 
        res1=[V, E0, P, B, BP, B2P];

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

    if icase == 5 or icase == 6:
        func=lambda x:(-((b + 2*c*math.log(x) + 3*d*math.log(x)**2 + 4*e*math.log(x)**3)/x))*chunit;
        V=fsolve(func,vzero);
        V=np.mean(V)
        P= -((b + 2*c*math.log(V) + 3*d*math.log(V)**2 + 4*e*math.log(V)**3)/V);  
        P=np.mean(P)
        P=P*chunit; 
        B =-((b - 2*c + 2*(c - 3*d)*math.log(V) + 3*(d - 4*e)*math.log(V)**2 + 4*e*math.log(V)**3)/V); 
        B=np.mean(B)
        B = B*chunit;
        BP=(b - 4*c + 6*d + 2*(c - 6*d + 12*e)*math.log(V) + 3*(d - 8*e)*math.log(V)**2 + 4*e*math.log(V)**3)/(b - 2*c + 2*(c - 3*d)*math.log(V) + 3*(d - 4*e)*math.log(V)**2 + 4*e*math.log(V)**3);
        B2P =(2*V*(2*c**2 - 3*b*d + 18*d**2 + 12*b*e - 6*c*(d + 4*e) + 6*(c*d - 3*d**2 - 2*b*e + 12*d*e)*math.log(V) + 9*(d - 4*e)**2*math.log(V)**2 + 24*(d - 4*e)*e*math.log(V)**3 + 24*e**2*math.log(V)**4))/(b - 2*c + 2*(c - 3*d)*math.log(V) + 3*(d - 4*e)*math.log(V)**2 + 4*e*math.log(V)**3)**3;
        B2P=np.mean(B2P)
        B2P = B2P/chunit; 
        E0= a + b*math.log(V) + c*math.log(V)**2 + d*math.log(V)**3 + e*math.log(V)**4;
        res1=[V, E0, P, B, BP, B2P];
    return(res1)    

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

    if L==1 or L==2:
        a=(8*E0 + 3*B*(122 + 9*B*b2p - 57*bp + 9*bp**2)*V)/8;
        e=(3*B*(74 + 9*B*b2p - 45*bp + 9*bp**2)*V**(7/3))/8;
        d=(-3*B*(83 + 9*B*b2p - 48*bp + 9*bp**2)*V**2)/2;
        c=(9*B*(94 + 9*B*b2p - 51*bp + 9*bp**2)*V**(5/3))/4;
        b=(-3*B*(107 + 9*B*b2p - 54*bp + 9*bp**2)*V**(4/3))/2;
        if abs(e) < 1e-8:
            e=0;
        res2=[a, b, c, d, e];      
    
    if L==3 or L==4:
        a=(128*E0 + 3*B*(287 + 9*B*b2p - 87*bp + 9*bp**2)*V)/128;
        e=(3*B*(143 + 9*B*b2p - 63*bp + 9*bp**2)*V**(11/3))/128;
        d=(-3*B*(167 + 9*B*b2p - 69*bp + 9*bp**2)*V**3)/32;
        c=(9*B*(199 + 9*B*b2p - 75*bp + 9*bp**2)*V**(7/3))/64;
        b=(-3*B*(239 + 9*B*b2p - 81*bp + 9*bp**2)*V**(5/3))/32;
        if abs(e) < 1e-8:
            e=0;
        res2=[a, b, c, d, e];      
        
    if L==5 or L==6:
        a=(24*E0 + 12*B*V*math.log(V)**2 + 4*B*(-2 + bp)*V*math.log(V)**3 + B*(3 + B*b2p - 3*bp + bp**2)*V*math.log(V)**4)/24;
        b=-(B*V*math.log(V)*(6 + 3*(-2 + bp)*math.log(V) + (3 + B*b2p - 3*bp + bp**2)*math.log(V)**2))/6;
        c=(B*V*(2 + 2*(-2 + bp)*math.log(V) + (3 + B*b2p - 3*bp + bp**2)*math.log(V)**2))/4;
        d=-(B*V*(-2 + bp + (3 + B*b2p - 3*bp + bp**2)*math.log(V)))/6;
        e=(B*(3 + B*b2p - 3*bp + bp**2)*V)/24;
        if abs(e) < 1e-8:
            e=0;    
        res2=[a, b, c, d, e]; 
    
    return(res2)
	
## ==========================================
def murnaghan(xini,Data):
#%  Equation of state: Murnaghan  
#%  #% V-1      E0-2       B-3        bp-4
#
    V  = xini[0];
    E0 = xini[1];
    B  = xini[2];
    bp = xini[3];
    x=Data[:,0];
    y=Data[:,1];
    eng = E0 - (B*V)/(-1 + bp)+(B*(1+(V/x)**bp/(-1+bp))*x)/bp; 
    return(eng-y);

## ==========================================	
def vineteos(xini,Data):
#%  Equation of state: Vinet 
#%  #% V-1      E0-2       B-3        bp-4
#%
    V  = xini[0];
    E0 = xini[1];
    B  = xini[2];
    bp = xini[3];
    x=Data[:,0];
    y=Data[:,1];
    eng =E0 + (4*B*V)/(-1 + bp)**2 - (4*B*V*(1 + (3*(-1 + bp)*(-1 + (x/V)**(1/3)))/2))/((-1 + bp)**2*np.exp((3*(-1 + bp)*(-1 + (x/V)**(1/3)))/2));
    return eng - y;
	
## ==========================================
def morseeos(xini,Data):
#%  Equation of state: Morse 
#%  #% V-1      E0-2       B-3        bp-4
#%
    V  = xini[0];
    E0 = xini[1];
    B  = xini[2];
    bp = xini[3];
#%
    a= E0 + (9*B*V)/(2*(-1 + bp)**2);
    b= (-9*B*np.exp(-1 + bp)*V)/(-1 + bp)**2;
    c= (9*B*np.exp(-2 + 2*bp)*V)/(2*(-1 + bp)**2);
    d= (1 - bp)/V**(1/3);
    x=Data[:,0];  #% volume
    y=Data[:,1];  #% energy
    eng = a + b*np.exp(d*x**(1/3)) + c*np.exp(2*d*x**(1/3)); 
    return eng - y;

## ==========================================
def pvbuildeos(prop, L):
#%  V0  P0  B  BP  B2P  av_diff   max_diff                                                            
#%  1   2   3   4  5     6           7
    V    = prop[0];          #%  V0[A]3/atom) 
    P0   = prop[1];          #%  P0[G]a)   
    B    = prop[2];          #%  B  GPa
    bp   = prop[3];          #%  BP  
    b2p  = prop[4];          #%  b2p  (1/GPa)

    if L==1 or L==2:
        e=(3*B*(74 + 9*B*b2p - 45*bp + 9*bp**2)*V**(7/3))/8;
        d=(-3*B*(83 + 9*B*b2p - 48*bp + 9*bp**2)*V**2)/2;
        c=(9*B*(94 + 9*B*b2p - 51*bp + 9*bp**2)*V**(5/3))/4;
        b=(-3*B*(107 + 9*B*b2p - 54*bp + 9*bp**2)*V**(4/3))/2;
        if abs(e) < 1e-8:
            e=0;     
        res1=[b, c, d, e];      

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
    if icase == 1: 
        V = 4*c**3 - 9*b*c*d + np.sqrt((c**2 - 3*b*d)*(4*c**2 -3*b*d)**2); V =-V/b**3;

    if icase == 2:
        fun=lambda x:((4*e)/(3*x**(7/3)) + d/x**2 + (2*c)/(3*x**(5/3)) + b/(3*x**(4/3)));
        V=fsolve(fun,vzero);

    if icase ==1 or icase == 2:
        P=(4*e)/(3*V**(7/3)) + d/V**2 + (2*c)/(3*V**(5/3)) + b/(3*V**(4/3));  
        B = ((28*e)/(9*V**(10/3)) + (2*d)/V**3 + (10*c)/(9*V**(8/3)) + (4*b)/(9*V**(7/3)))*V;  
        BP=(98*e + 54*d*V**(1/3) + 25*c*V**(2/3) + 8*b*V)/(42*e + 27*d*V**(1/3) + 15*c*V**(2/3) + 6*b*V); 
        B2P =(V**(8/3)*(9*d*(14*e + 5*c*V**(2/3) + 8*b*V) +  2*V**(1/3)*(126*b*e*V**(1/3) + 5*c*(28*e + b*V))))/(2*(14*e + 9*d*V**(1/3) + 5*c*V**(2/3) + 2*b*V)**3);
        res1=[V, P, B, BP, B2P];

#%%%%
    if icase == 3:
        V =np.sqrt(-((4*c**3 - 9*b*c*d + np.sqrt((c**2 - 3*b*d)*(4*c**2 - 3*b*d)**2))/b**3));

    if icase == 4:
        fun=lambda x:((8*e)/(3*x**(11/3)) + (2*d)/x**3 + (4*c)/(3*x**(7/3)) + (2*b)/(3*x**(5/3)))*chunit;
        V=fsolve(fun,vzero);
		
    if icase ==3 or icase == 4:
        P=(8*e)/(3*V**(11/3)) + (2*d)/V**3 + (4*c)/(3*V**(7/3)) + (2*b)/(3*V**(5/3)); 
        B =(2*(44*e + 27*d*V**(2/3) + 14*c*V**(4/3) + 5*b*V**2))/(9*V**(11/3));
        BP=(484*e + 243*d*V**(2/3) + 98*c*V**(4/3) + 25*b*V**2)/(132*e + 81*d*V**(2/3) + 42*c*V**(4/3) + 15*b*V**2);
        B2P =(4*V**(13/3)*(27*d*(22*e + 7*c*V**(4/3) + 10*b*V**2) + V**(2/3)*(990*b*e*V**(2/3) +  7*c*(176*e + 5*b*V**2))))/(44*e + 27*d*V**(2/3) + 14*c*V**(4/3) + 5*b*V**2)**3;
        res1=[V, P, B, BP, B2P];
		
    return res1

## ==========================================
def pveosfit(volume, fzero, pressure, vv, isave, ifigure, kkfig):#print
    numbereos = 4  
    ieos=[1, 2, 3, 4]; 
    chunit=160.2189;
    vzero=np.mean(volume); 
    res=[];  resdiff=[]; resxx=[]; respp=[];
#%
    for L in ieos :
#%%% 
        if L==1:   #%%%   mBM4    
            qq=volume;
            bb=pressure;
            AA=[qq**(-4/3)*(1/3), qq**(-5/3)*(2/3), qq**(-2)];  
            AA=np.array(AA)
            AA=AA.T
            xx1=np.linalg.pinv(AA)
            xx=xx1.dot(bb);   #%(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
            b=xx[0];  c=xx[1];  d=xx[2]; e=0.0;  xx=[b,c,d,e];

            pp= (4*e)/(3*vv**(7/3)) + d/vv**2 + (2*c)/(3*vv**(5/3)) + b/(3*vv**(4/3));  
            pp0= (4*e)/(3*volume**(7/3)) + d/volume**2 + (2*c)/(3*volume**(5/3)) + b/(3*volume**(4/3)); 
            if ifigure >1:
                plt.plot(vv,pp, volume, pressure, 'o')
                plt.title('P-V FITTED curve, mBM4, No. 1, GPa')
            diffp=pp0-pressure;
            prop = pv2prop(xx, vzero, L); #% [V0, P, B, BP, B2P];
            vzero=prop[0];
            newxx   =pvbuildeos(prop, L); 
            xxnp=np.array(xx)
            resxx1=np.insert(xxnp,0,L)
            resxx2=np.insert(newxx,0,L)
            resxx_2   =np.vstack((resxx1,resxx2))
            #print('resxx=',resxx)
            #print('resxx_2=',resxx_2)
            resxx=resxx_2
            nnn=len(bb);
            qwe=diffp**2; 
            asd=math.sqrt(sum(qwe/nnn)); 
            respre=np.insert(prop,0,L);
            res_2=np.hstack((respre,asd))
            res=res_2			
            resee=res; 
            respp=pp;
            resdiff=diffp 
 
#%%% 
        if L == 2:  #%% mBM5       
            qq=volume;
            bb=pressure;
            AA=[qq**(-4/3)*(1/3), qq**(-5/3)*(2/3), qq**(-2), qq**(-7/3)*(4/3)];  
            AA=np.array(AA)
            AA=AA.T
            xx1=np.linalg.pinv(AA)
            xx=xx1.dot(bb);   #%(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
            b=xx[0];  c=xx[1];  d=xx[2]; e=xx[3];  xx=[b,c,d,e];
 
            pp=(4*e)/(3*vv**(7/3)) + d/vv**2 + (2*c)/(3*vv**(5/3)) + b/(3*vv**(4/3));  
            pp0= (4*e)/(3*volume**(7/3)) + d/volume**2 + (2*c)/(3*volume**(5/3)) + b/(3*volume**(4/3)); 
            if ifigure >1:
                plt.plot(vv,pp, volume, pressure, 'o')
                plt.title('P-V FITTED curve, mBM5, No. 2, GPa')
            diffp=pp0-pressure;
            prop =pv2prop(xx, vzero, L); #% [V0, P, B, BP, B2P];
            newxx   =pvbuildeos(prop, L);
            xxnp=np.array(xx)
            resxx1=np.insert(xxnp,0,L)
            resxx2=np.insert(newxx,0,L)
            resxx_2   =np.vstack((resxx1,resxx2))
            resxx=np.hstack((resxx,resxx_2))
            nnn=len(bb);
            qwe=diffp**2; 
            asd=math.sqrt(sum(qwe/nnn)); 
            respre=np.insert(prop,0,L);
            res_2=np.hstack((respre,asd))
            res=np.vstack((res,res_2))
            resee=np.vstack((resee,res)); 
            respp=np.vstack((respp,pp));
            resdiff=np.vstack((resdiff,diffp)) 
#%%%%%% 
        if L == 3:  #%% BM4     
            qq=volume;
            bb=pressure;
            AA=[qq**(-5/3)*(2/3), qq**(-7/3)*(4/3), qq**(-3)*2]; 
            AA=np.array(AA)
            AA=AA.T
            xx1=np.linalg.pinv(AA)
            xx=xx1.dot(bb);   #%(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
            b=xx[0];  c=xx[1];  d=xx[2]; e=0.0;  xx=[b ,c, d, e];
            x=vv;
            pp=(8*e)/(3*x**(11/3))+(2*d)/x**3+(4*c)/(3*x**(7/3))+(2*b)/(3*x**(5/3));
            x=volume;
            pp0=(8*e)/(3*x**(11/3))+(2*d)/x**3+(4*c)/(3*x**(7/3))+(2*b)/(3*x**(5/3));
            if ifigure >1:
                plt.plot(vv,pp, volume, pressure, 'o')
                plt.title('P-V FITTED curve, BM4, No. 3, GPa')
            diffp=pp0-pressure;
            prop= pv2prop(xx, vzero, L); #% [V0, P, B, BP, B2P]; 
            newxx   =pvbuildeos(prop, L);
        
            xxnp=np.array(xx)
            resxx1=np.insert(xxnp,0,L)
            resxx2=np.insert(newxx,0,L)
            resxx_2   =np.vstack((resxx1,resxx2))
            resxx=np.hstack((resxx,resxx_2))
            nnn=len(bb);
            qwe=diffp**2; 
            asd=math.sqrt(sum(qwe/nnn)); 
            respre=np.insert(prop,0,L);
            res_2=np.hstack((respre,asd))
            res=np.vstack((res,res_2))
            resee=np.vstack((resee,res)); 
            respp=np.vstack((respp,pp));
            resdiff=np.vstack((resdiff,diffp)) 

#%%% 
        if L == 4:#%% BM5     
            qq=volume;
            bb=pressure;
            AA=[qq**(-5/3)*(2/3), qq**(-7/3)*(4/3), qq**(-3)*2, qq**(-11/3)*(8/3)]; 
            AA=np.array(AA)
            AA=AA.T
            xx1=np.linalg.pinv(AA)
            xx=xx1.dot(bb);   #%(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
            b=xx[0];  c=xx[1];  d=xx[2]; e=xx[3];  xx=[b, c, d, e];
 
            x=vv;
            pp=(8*e)/(3*x**(11/3))+(2*d)/x**3+(4*c)/(3*x**(7/3))+(2*b)/(3*x**(5/3));
            x=volume;
            pp0=(8*e)/(3*x**(11/3))+(2*d)/x**3+(4*c)/(3*x**(7/3))+(2*b)/(3*x**(5/3));
            if ifigure >1:
                plt.plot(vv,pp, volume, pressure, 'o')
                plt.title('P-V curve, BM5, No. 4, GPa')
            diffp=pp0-pressure;
            prop= pv2prop(xx, vzero, L); #% [V0, P, B, BP, B2P];
  
            newxx   =pvbuildeos(prop, L);
            xxnp=np.array(xx)
            resxx1=np.insert(xxnp,0,L)
            resxx2=np.insert(newxx,0,L)
            resxx_2   =np.vstack((resxx1,resxx2))
            resxx=np.hstack((resxx,resxx_2))
            nnn=len(bb);
            qwe=diffp**2; 
            asd=math.sqrt(sum(qwe/nnn)); 
            respre=np.insert(prop,0,L);
            res_2=np.hstack((respre,asd))
            res=np.vstack((res,res_2))
            resee=np.vstack((resee,res)); 
            respp=np.vstack((respp,pp));
            resdiff=np.vstack((resdiff,diffp)) 

#%%%%%% 
    if numbereos==4:
        pp4=np.vstack((respp[0,...], respp[2,...])); 
        pp4=pp4.T
        pp5=np.vstack((respp[1,...], respp[3,...])); 
        pp5=pp5.T
        if kkfig > 0:
            figpv4 = plt.figure('fig-pv4')
            plt.plot(vv, pp4, vv, pp5, '--', volume, pressure,'o')
            plt.title('P-V Fitted of 4 curves')
            plt.savefig('fig2_pv_fitted.png') 

    if numbereos==2:
        pp4=respp[0,...]; 
        pp4=pp4.T
        pp5=respp[1,...]; 
        pp5=pp5.T
        if kkfig > 0:
            figpv2 = plt.figure('fig-pv2')
            plt.plot(vv, pp4, vv, pp5, '--', volume, pressure,'o')
            plt.title('P-V Fitted of 2 curves'), 

    resdiff=resdiff.T
    max_diff=[]
    n=0
    max_diff1= np.fabs(resdiff);
    max_diff2=max_diff1.argmax(axis=0)
    for i in max_diff2:
        max_diff=np.append(max_diff,max_diff1[i,n])
        n=n+1
    
    av_diff =np.mean(max_diff1,axis=0);
    dpp_av_max=np.vstack((av_diff, max_diff))
    #print('dpp_av_max=',dpp_av_max)

    pvfit_res=res; #%[res(:,1:2), res(:,4:end)]
    np.savetxt("pvfit_res.txt",pvfit_res,fmt='%.4f')
#%ff=['outpv_eosres'];   eval(['save ' ff ' pvfit_res -ascii']); 

    if isave > 0:  
        resvp =np.hstack((vv.T, respp.T));         
        np.savetxt("out_fit_VP.txt",resvp,fmt='%.4f')
    #print('res=',res)
    return res

#### end of functions ###
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

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
    plt.savefig('fig1_strain_volume')

#%%%%%%%%%%%%%%%%%
chunit=160.2189;
vzero=np.mean(volume); 
res=[];  
resdiff=[]; 
resee=[]; 
resxx=[]; 
resdiffpp=[]; 
respp=[];

##print(np.shape(volume))
for L in ieos:
    if L==1 :#  %%% mBM4 
        bb=fzero;
        AA=np.vstack((np.ones(np.shape(volume)), volume**(-1/3), volume**(-2/3), volume**(-1)));  #%(nx4)  
        AA=AA.T
        xx1=np.linalg.pinv(AA);
        xx=xx1.dot(bb);   #%(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
        a=xx[0];  b=xx[1];  c=xx[2];  d=xx[3];  e=0.0;  
        xx=[a, b, c, d, e];

        pp=(4*e)/(3*vv**(7/3)) + d/vv**2 + (2*c)/(3*vv**(5/3)) + b/(3*vv**(4/3)); 
        pp=pp*chunit; 
        pp0= (4*e)/(3*volume**(7/3)) + d/volume**2 + (2*c)/(3*volume**(5/3)) + b/(3*volume**(4/3)); 
        pp0=pp0*chunit; 
        if ifigure >1:
            plt.plot(vv,pp, volume, pressure, 'o')
        diffp=pp0-pressure;
        resdiffpp=diffp;
        ee = a + b*(vv)**(-1/3) + c*(vv)**(-2/3) + d*(vv)**(-1) + e*(vv)**(-4/3);             
		
        if ifigure > 1:
            plt.plot(vv,ee, volume, bb,'o');
        curve= a + b*(volume)**(-1/3) + c*(volume)**(-2/3) + d*(volume)**(-1) + e*(volume)**(-4/3);
        diff = curve-bb;
        prop = eosparameter45(xx, vzero, L); #% [V0, E0, P, B, BP, B2P];
                
        newxx = build_eos(prop,L)
        xxnp  = np.array(xx)
        resxx1=np.insert(xxnp,0,L)
        resxx2=np.insert(newxx,0,L)
        resxx =np.vstack((resxx1,resxx2))
        nnn=len(bb);
        qwe=(diff/bb)**2; 
        asd=math.sqrt(sum(qwe/nnn)); 
        respre=np.insert(prop,0,L);
        res=np.hstack((respre,asd))
        resee =ee; 
        respp =pp;
        resdiff = diff; 
        xini = [prop[0], prop[1], prop[3]/chunit, prop[4]]; #% used for ieos= 7 and 8
		
    if L == 2: ## mBM5 
        bb=fzero;
        AA=np.vstack((np.ones(np.shape(volume)), volume**(-1/3), volume**(-2/3), volume**(-1),volume**(-4/3)))
        AA=AA.T
        xx1=np.linalg.pinv(AA);
        xx=xx1.dot(bb);     #(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
        a=xx[0];  b=xx[1];  c=xx[2];  d=xx[3]; e=xx[4];  
        xx=[a, b, c, d, e];
 
        pp=(4*e)/(3*vv**(7/3)) + d/vv**2 + (2*c)/(3*vv**(5/3)) + b/(3*vv**(4/3));  pp=pp*chunit; 
        pp0= (4*e)/(3*volume**(7/3)) + d/volume**2 + (2*c)/(3*volume**(5/3)) + b/(3*volume**(4/3)); 
        pp0=pp0*chunit; 
        if  ifigure >1:
            fig2 = plt.figure('fig2')              
            plt.plot(vv,pp, volume, pressure, 'o')
            plt.title('P-V curve, mBM5, No. 2, GPa')                 
        diffp=pp0-pressure;
        resdiffpp=np.vstack((resdiffpp, diffp));
        ee = a + b*(vv)**(-1/3) + c*(vv)**(-2/3) + d*(vv)**(-1) + e*(vv)**(-4/3); 
        if ifigure > 1:
            fig3 = plt.figure('fig3')            
            plt.plot(vv,ee, volume, bb,'o')
            plt.title('E-V curve, mBM5, No. 2');
        curve= a + b*(volume)**(-1/3) + c*(volume)**(-2/3) + d*(volume)**(-1) + e*(volume)**(-4/3);
        diff = curve-bb;
        prop =eosparameter45(xx, vzero, L); # [V0, E0, P, B, BP, B2P];

        newxx = build_eos(prop,L)
        xxnp = np.array(xx)
        resxx1 = np.insert(xxnp,0,L)
        resxx2 = np.insert(newxx,0,L)
        resxx_2= np.vstack((resxx1,resxx2))
        resxx= np.hstack((resxx,resxx_2))
        nnn=len(bb);
        qwe=(diff/bb)**2; 
        asd=math.sqrt(sum(qwe/nnn)); 
        respre=np.insert(prop,0,L);
        res_2=np.hstack((respre,asd))
        res=np.vstack((res,res_2))
        resee=np.vstack((resee,ee)); 
        respp=np.vstack((respp,pp));
        resdiff=np.vstack((resdiff,diff)) 
### 
    if L == 3: ## BM4 
        bb=fzero;
        AA=np.vstack((np.ones(np.shape(volume)), volume**(-2/3), volume**(-4/3), volume**(-2)))
        AA=AA.T
        xx1=np.linalg.pinv(AA);
        xx=xx1.dot(bb);     #(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
        a=xx[0];  b=xx[1];  c=xx[2];  d=xx[3]; e=0.0;  
        xx=[a, b, c, d, e];
 
        x=vv;
        pp=((8*e)/(3*x**(11/3))+(2*d)/x**3+(4*c)/(3*x**(7/3))+(2*b)/(3*x**(5/3)))*chunit;
        x=volume;
        pp0=((8*e)/(3*x**(11/3))+(2*d)/x**3+(4*c)/(3*x**(7/3))+(2*b)/(3*x**(5/3)))*chunit;
        if ifigure >1:
            fig4 = plt.figure('fig4')            
            plt.plot(vv,pp, volume, pressure, 'o')
            plt.title('P-V curve, BM4, No. 3, GPa'),
        diffp=pp0-pressure;
        resdiffpp=np.vstack((resdiffpp, diffp));
        ee = a + b*(vv)**(-2/3) + c*(vv)**(-4/3) + d*(vv)**(-2) + e*(vv)**(-8/3);
		
        if ifigure > 1:
            fig5 = plt.figure('fig5')            
            plt.plot(vv,ee, volume, bb,'o')
            plt.title('E-V curve, BM4, No. 3')
        curve= a + b*(volume)**(-2/3) + c*(volume)**(-4/3) + d*(volume)**(-2) + e*(volume)**(-8/3);
        diff = curve-bb;
        prop = eosparameter45(xx, vzero, L); # [V0, E0, P, B, BP, B2P]; 
        bm4ev_bulk = np.array2string(prop[3], precision=1) 
        bm4ev_bp   = np.array2string(prop[4], precision=2)
        #print('BM4 B Bp = ', bm4ev_bulk, bm4ev_bp, type(bm4ev_bulk)) 
        newxx =build_eos(prop,L)
        xxnp =np.array(xx)
        resxx1=np.insert(xxnp,0,L)
        resxx2=np.insert(newxx,0,L)
        resxx_2 =np.vstack((resxx1,resxx2))
        resxx=np.hstack((resxx,resxx_2))
        nnn=len(bb);
        qwe=(diff/bb)**2; 
        asd=math.sqrt(sum(qwe/nnn)); 
        respre=np.insert(prop,0,L);
        res_2=np.hstack((respre,asd))
        res=np.vstack((res,res_2))
        resee=np.vstack((resee,ee)); 
        respp=np.vstack((respp,pp));
        resdiff=np.vstack((resdiff,diff)) 

####### 
    if L == 4: ## BM5    
        bb=fzero;
        AA=np.vstack((np.ones(np.shape(volume)), volume**(-2/3), volume**(-4/3), volume**(-2),volume**(-8/3))) 
        AA=AA.T
        xx1=np.linalg.pinv(AA);
        xx=xx1.dot(bb);     #(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
        a=xx[0];  b=xx[1];  c=xx[2];  d=xx[3]; e=xx[4];  
        xx=[a ,b, c, d, e];
 
        x=vv;
        pp=((8*e)/(3*x**(11/3))+(2*d)/x**3+(4*c)/(3*x**(7/3))+(2*b)/(3*x**(5/3)))*chunit;
        x=volume;
        pp0=((8*e)/(3*x**(11/3))+(2*d)/x**3+(4*c)/(3*x**(7/3))+(2*b)/(3*x**(5/3)))*chunit;
        if ifigure >1:
            fig6 = plt.figure('fig6')            
            plt.plot(vv,pp, volume, pressure, 'o')
            plt.title('P-V curve, BM5, No. 4, GPa'),

        diffp=pp0-pressure;
        resdiffpp=np.vstack((resdiffpp, diffp));
        ee = a + b*(vv)**(-2/3) + c*(vv)**(-4/3) + d*(vv)**(-2) + e*(vv)**(-8/3); 
		
        if ifigure > 1:
            fig7 = plt.figure('fig7')            
            plt.plot(vv,ee, volume, bb,'o')
            plt.title('E-V curve, BM5, No. 4'),;
        curve= a + b*(volume)**(-2/3) + c*(volume)**(-4/3) + d*(volume)**(-2) + e*(volume)**(-8/3);
        diff = curve-bb;
        prop=eosparameter45(xx, vzero, L); # [V0, E0, P, B, BP, B2P];

        newxx = build_eos(prop,L)
        xxnp  = np.array(xx)
        resxx1 = np.insert(xxnp,0,L)
        resxx2 = np.insert(newxx,0,L)
        resxx_2= np.vstack((resxx1,resxx2))
        resxx=np.hstack((resxx,resxx_2))
        nnn=len(bb);
        qwe=(diff/bb)**2; 
        asd=math.sqrt(sum(qwe/nnn)); 
        respre=np.insert(prop,0,L);
        res_2=np.hstack((respre,asd))
        res=np.vstack((res,res_2))
        resee=np.vstack((resee,ee)); 
        respp=np.vstack((respp,pp));
        resdiff=np.vstack((resdiff,diff)) 

####### 
    if L == 5:  ## LOG4
        bb=fzero;
        AA=np.vstack((np.ones(np.shape(volume)), np.log(volume), np.log(volume)**2, np.log(volume)**3)) 
        AA=AA.T
        xx1=np.linalg.pinv(AA);
        xx=xx1.dot(bb);     #(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
        a=xx[0];  b=xx[1];  c=xx[2];  d=xx[3]; e=0.0;
        xx=[a,b ,c, d, e,];
 
        x=vv;
        pp=(-((b + 2*c*np.log(x) + 3*d*np.log(x)**2 + 4*e*np.log(x)**3)/x))*chunit;
        x=volume;
        pp0=(-((b + 2*c*np.log(x) + 3*d*np.log(x)**2 + 4*e*np.log(x)**3)/x))*chunit;
        if ifigure >1:
            fig8 = plt.figure('fig8')            
            plt.plot(vv,pp, volume, pressure, 'o')
            plt.title('P-V curve, math.LOG4, No. 5, GPa'),

        diffp=pp0-pressure;
        resdiffpp=np.vstack((resdiffpp, diffp));
        ee = a + b*np.log(vv) + c*np.log(vv)**2 + d*np.log(vv)**3 + e*np.log(vv)**4;
		
        if ifigure > 1:
            fig9 = plt.figure('fig9')            
            plt.plot(vv,ee, volume, bb,'o')
            plt.title('E-V curve, math.LOG4, No. 5')
        curve= a + b*np.log(volume) + c*np.log(volume)**2 + d*np.log(volume)**3 + e*np.log(volume)**4;
        diff = curve-bb;
        prop=eosparameter45(xx, vzero, L); # [V0, E0, P, B, BP, B2P]; 
        
        newxx =build_eos(prop,L)
        xxnp=np.array(xx)
        resxx1=np.insert(xxnp,0,L)
        resxx2=np.insert(newxx,0,L)
        resxx_2   =np.vstack((resxx1,resxx2))
        resxx=np.hstack((resxx,resxx_2))
        nnn=len(bb);
        qwe=(diff/bb)**2; 
        asd=math.sqrt(sum(qwe/nnn)); 
        respre=np.insert(prop,0,L);
        res_2=np.hstack((respre,asd))
        res=np.vstack((res,res_2))
        resee=np.vstack((resee,ee)); 
        respp=np.vstack((respp,pp));
        resdiff=np.vstack((resdiff,diff)) 

###### 
    if L == 6:  ## LOG5
        bb=fzero;
        AA=np.vstack((np.ones(np.shape(volume)), np.log(volume), np.log(volume)**2, np.log(volume)**3,np.log(volume)**4))   
        AA=AA.T
        xx1=np.linalg.pinv(AA);
        xx=xx1.dot(bb);     #(4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
        a=xx[0];  b=xx[1];  c=xx[2];  d=xx[3]; e=xx[4];  
        xx=[a ,b, c, d, e];
 
        x=vv;
        pp=(-((b + 2*c*np.log(x) + 3*d*np.log(x)**2 + 4*e*np.log(x)**3)/x))*chunit;
        x=volume;
        pp0=(-((b + 2*c*np.log(x) + 3*d*np.log(x)**2 + 4*e*np.log(x)**3)/x))*chunit;
        if ifigure >1:
            fig10 = plt.figure('fig10')            
            plt.plot(vv,pp, volume, pressure, 'o')
            plt.title('P-V curve, np.LOG5, No. 6, GPa')

        diffp=pp0-pressure;
        resdiffpp=np.vstack((resdiffpp, diffp));
        ee = a + b*np.log(vv) + c*np.log(vv)**2 + d*np.log(vv)**3 + e*np.log(vv)**4; 
		
        if ifigure > 1:
            fig11 = plt.figure('fig11')
            plt.plot(vv,ee, volume, bb,'o')
            plt.title('E-V curve, np.LOG5, No. 6')            
        curve= a + b*np.log(volume) + c*np.log(volume)**2 + d*np.log(volume)**3 + e*np.log(volume)**4;
        diff = curve-bb;
        prop=eosparameter45(xx, vzero, L); # [V0, E0, P, B, BP, B2P]; 

        newxx   =build_eos(prop,L)
        xxnp=np.array(xx)
        resxx1=np.insert(xxnp,0,L)
        resxx2=np.insert(newxx,0,L)
        resxx_2   =np.vstack((resxx1,resxx2))
        resxx=np.hstack((resxx,resxx_2))
        nnn=len(bb);
        qwe=(diff/bb)**2; 
        asd=math.sqrt(sum(qwe/nnn)); 
        respre=np.insert(prop,0,L);
        res_2=np.hstack((respre,asd))
        res=np.vstack((res,res_2))
        resee=np.vstack((resee,ee)); 
        respp=np.vstack((respp,pp));
        resdiff=np.vstack((resdiff,diff)) 

    if L == 7:
        xdata=volume;
        ydata=fzero;
        Data=np.vstack((xdata, ydata));
        Data=Data.T
   
           #% V-1      E-2       B-3           bp-4
   #%xini = [prop(1), prop(2), prop(3)/chunit, prop(4)]; % input as ieos==0
        [xout,resnorm] = leastsq(murnaghan,xini,Data);
        V=xout[0]; E0=xout[1]; B=xout[2]; bp=xout[3]; 
 
        x=vv;
        pp=chunit*(B*(-1 + (V/x)**bp))/bp;
        x=volume;
        pp0=chunit*(B*(-1 + (V/x)**bp))/bp;
        if ifigure >1:
            fig12 = plt.figure('fig12')            
            plt.plot(vv,pp, volume, pressure, 'o')
            plt.title('P-V curve, Mur, No. 7, GPa')

        diffp=pp0-pressure;
        resdiffpp=np.vstack((resdiffpp, diffp));
        ee = E0-(B*V)/(-1+bp) + (B*(1+(V/vv)**bp/(-1+bp))*vv)/bp; 
    	#ee=  E0-(B*V)/(-1+bp) + (B*(1+(V./vv).^bp./(-1+bp)).*vv)./bp;

        if ifigure > 1:
            fig13 = plt.figure('fig13')            
            plt.plot(vv,ee, volume, bb,'o')
            plt.title('E-V curve, Mur, No. 7')

        curve= (E0-(B*V)/(-1+bp) + (B*(1+(V/volume)**bp/(-1+bp))*volume)/bp);
        diff = curve-bb;
        prop=[V, E0, 0, B*chunit, bp, 0]; 

        nnn=len(bb);
        qwe=(diff/bb)**2; 
        asd=math.sqrt(sum(qwe/nnn)); 
        respre=np.insert(prop,0,L);
        res_2=np.hstack((respre,asd))
        res=np.vstack((res,res_2))
        resee=np.vstack((resee,ee)); 
        respp=np.vstack((respp,pp));
        resdiff=np.vstack((resdiff,diff)) 

    if L == 8:  #%#% Vinet
        xdata=volume;
        ydata=fzero;
        Data=np.vstack((xdata, ydata));
        Data=Data.T
        
        [xout,resnorm] = leastsq(vineteos,xini,Data);
        V=xout[0]; E0=xout[1]; B=xout[2]; bp=xout[3]; 
    
        x=vv;
        pp=chunit*(-3*B*(-1 + (x/V)**(1/3)))/(np.exp((3*(-1 + bp)*(-1 + (x/V)**(1/3)))/2)*(x/V)**(2/3));
        x=volume;
        pp0=chunit*(-3*B*(-1 + (x/V)**(1/3)))/(np.exp((3*(-1 + bp)*(-1 + (x/V)**(1/3)))/2)*(x/V)**(2/3));
        if ifigure >1:
            plt.plot(vv,pp, volume, pressure, 'o')
            plt.title('P-V curve, Vinet, No. 8, GPa')
            fig14 = plt.figure('fig14')
        diffp=pp0-pressure;
        resdiffpp=np.vstack((resdiffpp, diffp));
        ee=E0 + (4*B*V)/(-1 + bp)**2 - (4*B*V*(1 + (3*(-1 + bp)*(-1 + (vv/V)**(1/3)))/2))/((-1 + bp)**2*np.exp((3*(-1 + bp)*(-1 + (vv/V)**(1/3)))/2));
		
        if ifigure > 1:
            fig15 = plt.figure('fig15')            
            plt.plot(vv,ee, volume, bb,'o')
            plt.title('E-V curve, Vinet, No. 8')

        curve=E0 + (4*B*V)/(-1 + bp)**2 - (4*B*V*(1 + (3*(-1 + bp)*(-1 + (volume/V)**(1/3)))/2))/((-1 + bp)**2*np.exp((3*(-1 + bp)*(-1 + (volume/V)**(1/3)))/2));
        diff = curve-bb;
        b2p= (19 - 18*bp - 9*bp**2)/(36*B);
        prop=[V, E0, 0, B*chunit, bp, b2p/chunit]; 

        nnn=len(bb);
        qwe=(diff/bb)**2; 
        asd=math.sqrt(sum(qwe/nnn)); 
        respre=np.insert(prop,0,L);
        res_2=np.hstack((respre,asd))
        res=np.vstack((res,res_2))
        resee=np.vstack((resee,ee)); 
        respp=np.vstack((respp,pp));
        resdiff=np.vstack((resdiff,diff)) 

#%#%#%#%#%#% 
    if L == 9:  #%#% Morse
        xdata=volume;
        ydata=fzero;
        Data=np.vstack((xdata, ydata));
        Data=Data.T
        [xout,resnorm] = leastsq(morseeos,xini,Data);
        V=xout[0]; E0=xout[1]; B=xout[2]; bp=xout[3]; 
   
        a= E0 + (9*B*V)/(2*(-1 + bp)**2);
        b= (-9*B*np.exp(-1 + bp)*V)/(-1 + bp)**2;
        c= (9*B*np.exp(-2 + 2*bp)*V)/(2*(-1 + bp)**2);
        d= (1 - bp)/V**(1/3);
   
        x=vv;
        pp=-chunit*(d*np.exp(d*x**(1/3))*(b + 2*c*np.exp(d*x**(1/3))))/(3*x**(2/3)); 
        x=volume;
        pp0=-chunit*(d*np.exp(d*x**(1/3))*(b + 2*c*np.exp(d*x**(1/3))))/(3*x**(2/3)); 
        
        if ifigure >1:
            fig16 = plt.figure('fig16')            
            plt.plot(vv,pp, volume, pressure, 'o')
            plt.title('P-V curve, Morse, No. 9, GPa')
        diffp=pp0-pressure;
        resdiffpp=np.vstack((resdiffpp, diffp));   
        ee=a + b*np.exp(d*vv**(1/3)) + c*np.exp(2*d*vv**(1/3));
		
        if ifigure > 1:
            fig17 = plt.figure('fig17')             
            plt.plot(vv,ee, volume, bb,'o')
            plt.title('E-V curve, Vinet, No. 8')

        x=volume;
        curve=a + b*np.exp(d*x**(1/3)) + c*np.exp(2*d*x**(1/3)); 
        diff = curve-bb;
        b2p= (5 - 5*bp - 2*bp**2)/(9*B);
        prop=[V, E0, 0, B*chunit, bp, b2p/chunit]; 

        nnn=len(bb);
        qwe=(diff/bb)**2; 
        asd=math.sqrt(sum(qwe/nnn)); 
        respre=np.insert(prop,0,L);
        res_2=np.hstack((respre,asd))
        res=np.vstack((res,res_2))
        resee=np.vstack((resee,ee)); 
        respp=np.vstack((respp,pp));
        resdiff=np.vstack((resdiff,diff))     
        respp=np.array(respp)        

if numbereos==6:
    pp4=np.hstack((respp[0,:], respp[2,:], respp[4,:])); 
    pp5=np.hstack((respp[1,:], respp[3,:], respp[5,:])); 
    plt.plot(vv, pp4, vv, pp5, '--', volume, pressure,'o')
    plt.title('P-V curves'), 
    plt.legend('1-mBM4', '3-BM4', '5-LOG4', '2-mBM5','4-BM5','6-LOG5', '10-Calc');
    fig18 = plt.figure('fig18') 

    ee4=np.vstack((resee[0,:], resee[2,:], resee[4,:])); 
    ee5=np.vstack((resee[1,:], resee[3,:], resee[5,:])); 
    plt.plot(vv, ee4, vv, ee5, '--', volume, bb,'o')
    plt.title('E-V curves')
    plt.legend('1-mBM4', '3-BM4', '5-LOG4', '2-mBM5','4-BM5','6-LOG5', '10-Calc');
    fig19 = plt.figure('fig19') 

if numbereos==9:
    a=respp[1,...]
    b=respp[3,...]
    pp51=np.vstack((a,b))
    pp5=np.vstack((pp51,respp[5,:]));
    pp5=pp5.T
    pp4=respp
    pp4=np.delete(pp4,[1,3,5],0)
    pp4=pp4.T
    if kkfig > 0:
        fig20pv = plt.figure('fig20-pv') 
        plt.plot(vv, pp4, vv, pp5, '--', volume, pressure,'o')
        plt.title('P-V curves from E-V'), 
        plt.legend(['1:mBM4', '3:BM4', '5-LOG4', '7-Mur', '8-Vinet', '9-Morse', '2-mBM5','4-BM5','6-LOG5', '10-Calc']);
        plt.savefig('fig20_pv_from_ev.png')
    
    ee5=np.vstack((resee[1,:], resee[3,:], resee[5,:])); 
    ee5=ee5.T
    ee4=resee
    ee4=np.delete(ee4,[1,3,5],0)
    ee4=ee4.T
    #print('ee4',ee4)
    #print('resee',resee[:,3:10])
    if kkfig > 0:   
        fig21ev = plt.figure('fig21-ev')     
        plt.plot(vv, ee4, vv, ee5, '--', volume, bb,'o')            
        plt.title('9 E-V eos curves: BM4 B Bp = ' + bm4ev_bulk + ', ' + bm4ev_bp) 
        plt.legend(['1-mBM4', '3-BM4', '5-LOG4', '7-Mur', '8-Vinet', '9-Morse', '2-mBM5','4-BM5','6-LOG5', '10-Calc']);  
        plt.savefig('fig21_ev_fitted.png')  

resdiffpp=resdiffpp.T
resdiff=resdiff.T
max_diff=[]
n=0
max_diff1= np.fabs(resdiffpp);
max_diff2=max_diff1.argmax(axis=0)
for i in max_diff2:
    max_diff=np.append(max_diff,max_diff1[i,n])
    n=n+1
    
av_diff =np.mean(max_diff1,axis=0);
dpp_av_max=np.vstack((av_diff, max_diff))
#print('dpp_av_max=',dpp_av_max)

max_diff=[]
n=0
max_diff1= np.fabs(resdiff);
max_diff2=max_diff1.argmax(axis=0)
for i in max_diff2:
    max_diff=np.append(max_diff,max_diff1[i,n])
    n=n+1

av_diff =np.mean(max_diff1,axis=0);
dene_av_max=np.vstack((av_diff,max_diff))
qwe=res[...,-1]*10**4; 
res[...,-1]=qwe
fitted_res=np.delete(res,[3],1);

np.savetxt("fitted_res.txt",fitted_res,fmt='%.4f')
diff_av_max=np.vstack((res[:,0], dene_av_max, dpp_av_max)); 
diff_av_max=diff_av_max.T
np.savetxt("diff_av_max.txt",diff_av_max,fmt='%.4f')
if isave > 0: 
    resve =np.hstack((vv.T, resee.T));
    resvp =np.hstack((vv.T, respp.T)); 
    np.savetxt("out_fit_VE.txt",resve,fmt='%.4f')
    np.savetxt("out_fit_VP.txt",resvp,fmt='%.4f')

#%  N-list  V0  E0   B  BP  B2P  av_diff_e  max_diff_e  av_diff_p   max_diff_p   
#%    1     2   3    4  5   6        2        3          4          5                     

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ipress > 0:
    pvfit_res=pveosfit(volume, fzero, pressure, vv, isave, ifigure, kkfig);
    compare_res=np.vstack((fitted_res[0,...],pvfit_res[0,...],fitted_res[1,...],pvfit_res[1,...],fitted_res[2,...],pvfit_res[2,...],fitted_res[3,...], pvfit_res[3,...])) 
    final_res=np.vstack((fitted_res,pvfit_res))	
if ipress < 0:
    final_res=fitted_res

np.savetxt("out_eosres.txt",final_res,fmt='%.6f')
if kkshow > 0: plt.show()

if os.path.isfile('magtot.0'): 
    with open('magtot.0') as f: content0=f.read().splitlines()
    if content0[0] != '':
        mm=np.loadtxt('magtot.0')
        fig22mm = plt.figure('fig22-mag')
        plt.plot(volume, mm[datarange], '-o')
        plt.title ('magnetic moment')
        if kkfig  > 0: plt.savefig('fig22_v_mag')
        if kkshow > 0: plt.show()
