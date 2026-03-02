#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:13:55 2024

@author: shin
"""

import numpy as np
from scipy import interpolate, linalg, optimize ,integrate
from optparse import OptionParser
from collections import OrderedDict
import pickle
import time
import sys
import pandas as pd
from astropy.coordinates import angular_separation as ang 
from pathlib import Path
import matplotlib.pyplot as plt
import astropy.units as u
usage = 'usage: %prog [options]'
parser = OptionParser(usage)
parser.add_option( "-d", "--details", action="store", type="int", default=2, dest="DET", help="1: Do pheno Q fit with JLA only. 2: Fit for a non scale dependent dipolar modulation in Q. 3: Fit for a top hat scale dependent dipolar modulation in Q. 4. Fit for an exponentially falling scale dependent dipolar modulation in Q. 5. Fit for a linearly falling scale dependent dipolar modulation in Q. ")
parser.add_option( "-r", "--reversebias", action = "store_true", default=True, dest="REVB", help = "Reverse the bias corrections")
parser.add_option( "-s", "--scan", action = "store_true", default=False, dest="SCAN", help = "Whether to do a scan")
parser.add_option( "--method", action="store", type="int", default=7, dest="MET", help="1:Nelder-Mead 2: SLSQP 3: Powell. ")
parser.add_option( "--dipole", action = "store_true", default=False, dest="DIP", help = "Need to estimate dipole too?")
parser.add_option( "--dipoledir", action = "store", type="int", default=1, dest="DIPDIR", help = "DIPOLE DIRECTION FIX TO?")
parser.add_option( "-e","--evaluate", action = "store",type='int', dest="EVAL", default=3, help = "What evaluation needs to be done, 1: LCDM , 2: z Taylor , 3: z Dipole Taylor , 4: H0 dipole, 5: H0,Q0 dipole ,6: Quadrupolar ?")

parser.add_option( "--zind", action = "store", type='int',default=9, dest="ZINDEX", help = "ZINDEX?")
parser.add_option("--age_bias", action="store_true", default=False, dest="AGEBIAS", help="Apply progenitor age bias correction")
parser.add_option("--age_bias_statistics",action='store',type = "string", default="median", dest="AGEBAST", help="Use mean or median progenitor age bias correction")

parser.add_option( "--arg1", action = "store", type="float", default=0, dest="ARG1", help = "ARGUMENT1 for QM?")
parser.add_option( "--arg2", action = "store", type="float", default=0, dest="ARG2", help = "ARGUMENT2 for QD? ")
parser.add_option("--zlim", action = "store",type="float", default=0.00937, dest="ZLIM")
parser.add_option("--cluster", action="store", type="str", dest="CLUSTER", help="Cluster ID for parallel processing")
parser.add_option("--process", action="store", type="str", dest="PROCESS", help="Process ID within cluster")
parser.add_option("--multiprocess", action="store_true", default=False, dest="MULTIPROC", help="Enable multiprocessing")
parser.add_option( "--debug", action = "store_true", default=False, dest="DEBUG", help = "Want detailed outputs?")
(options, args) = parser.parse_args()

dctdet={1:"Non scalar",2:"Exponential"}

if options.DET==1:
    STYPE='NoScDep'
elif options.DET==2:
    STYPE='Exp'

else:
    STYPE='None'
    
if options.MET==1:
    met='Nelder-Mead'
elif options.MET==2:
    met='SLSQP'
elif options.MET==3:
    met='Powell'
elif options.MET==4:
    met="trust-constr"
elif options.MET==5:
    met="TNC"
elif options.MET ==6:
    met="COBYLA"
elif options.MET ==7:
    met="L-BFGS-B"
elif options.MET==8:
    met= "Newton-CG"
elif options.MET==9:
    met='BFGS'
elif options.MET==10:
    met='CG'
elif options.MET==11:
    met='trust-exact'

c = 299792.458 # km/s
H0 = 70 #(km/s) / Mpc

CMBdipdec = -7
CMBdipra = 168
two_mppra=164.6877348*u.deg 
two_mppdec=-17.1115932*u.deg
raLG_SUN= 333.53277784*u.deg
decLG_SUN = 49.33111122*u.deg

vLG_SUN= 299 #km/s


path='/Storage/animesh/DATA_PPLUS'
if options.AGEBIAS:
    print("Applying progenitor age bias correction using ",options.AGEBAST," statistics")
    with open(f'{path}/cs_'+options.AGEBAST+'.pkl', 'rb') as f:
        cs_bias = pickle.load(f)    
def Minimizer(zlim1=0,qd=None ,qm=None):
    radip=None
    decdip=None
    name=str(met)
    index=np.load('/Storage/animesh/Analysis_C2/index_sorted_lane.npy')
    Z = pd.read_csv(path+'/Zpan.csv')

    df=pd.read_csv( path+'/Pantheon+SH0ES.dat',delimiter=' ')
    if options.REVB:
        print ('reversing bias')
        Z['mB'] = Z['mB'] + df['biasCor_m_b']
    if options.AGEBIAS:
        print("Applying progenitor age bias correction")
        Z['mB'] = Z['mB'] - 0.03*cs_bias(df['zHEL'].to_numpy())
    Z=Z.loc[index]
    Z=Z.reset_index(drop=True)
    Z=Z[Z['zHEL']<0.8]
    if zlim1!=0:
        print("Using the zlim provided",zlim1)
        Z=Z[Z['zHEL']>zlim1]
        

    ra=np.array(Z['RA'])*u.deg
    dec=np.array(Z['DEC'])*u.deg
    zDF=np.sqrt((1-vLG_SUN*np.cos(ang(raLG_SUN,decLG_SUN,ra,dec)).value/c)/(1+vLG_SUN*np.cos(ang(raLG_SUN,decLG_SUN,ra,dec)).value/c))-1
    zLG=((1+Z['zHEL'])/(1+zDF))-1
    l=Z.index.values.tolist() 
    l=np.array(l)
    tempind=np.array([[3*i,3*i+1,3*i+2] for i in l])
    tempind=tempind.flatten(order='c')
    N= len(Z)
    Z['zLG']=zLG
    Z=Z.to_numpy()
    INDEX_DCT={9:"zHEL",0:"zHD",10:"zCMB",11:"ZLG"}
    ZINDEX=options.ZINDEX
    print('Using Redshift:',INDEX_DCT[ZINDEX])  
    if not options.DIP:
        name=name+"_FIX_DIP_TO_"
        if options.DIPDIR==1:
            radip=CMBdipra
            decdip=CMBdipdec
            name=name+"CMB_"
    else:
        name=name+"_FLOAT_DIP"
    def cdAngle(ra1, dec1, ra2, dec2):
        return np.cos(np.deg2rad(dec1))*np.cos(np.deg2rad(dec2))*np.cos(np.deg2rad(ra1) - np.deg2rad(ra2))+np.sin(np.deg2rad(dec1))*np.sin(np.deg2rad(dec2))
 
   
        
    def MUZ(Zc, Q0, J0,S0=None,L0=None,OK=None):
        k = 5.*np.log10( c/H0 * dLPhenoF3(Zc, Q0, J0)) + 25.   
        if np.any(np.isnan(k)):
            k[np.isnan(k)] = 63.15861331456834
        return k
    
    
    def dLPhenoF3(z, q0, j0):
        return z*(1.+0.5*(1.-q0)*z -1./6.*(1. - q0 - 3.*q0**2. + j0)*z**2.) *(1+Z[:,9])/(1+z)
    
    if ZINDEX==11:
        radip= 162.95389715
        decdip=-25.96734154
    name+=str(INDEX_DCT[ZINDEX])+'_'
    print("Using Redshift:",INDEX_DCT[ZINDEX])
    zlim=Z[:,ZINDEX][0]
    print(tempind)
    COVd = np.load('/Storage/animesh/Analysis_C2/cov_final.npy')
    COVd=COVd[:,tempind][tempind]

    def COV( sM,A ,sX, B, sC , RV=0): # Total covariance matrix
        if options.DEBUG:
            print(f'sM:{sM}, A:{A}, sX:{sX}, B:{B}, sC:{sC}')
        block3 = np.array( [[sM**2 + (sX**2)*A**2 + (sC**2)*B**2,    -(sX**2)*A, (sC**2)*B],
                                                    [-(sX**2)*A , (sX**2), 0],
                                                    [ (sC**2)*B ,  0, (sC**2)]] )
        ATCOVlA = linalg.block_diag( *[ block3 for i in range(N) ] ) ;
        
        if RV==0:
            return np.array( COVd + ATCOVlA );
        elif RV==1:
            return np.array( COVd );
        elif RV==2:
            return np.array( ATCOVlA );
    
    

    
    
    def RESVF3Dip(M0,  A ,X0, B , C0,Q0,J0, QD, DS=np.inf,ra=radip,dec=decdip, stype = STYPE): #Total residual, \hat Z - Y_0*A
        if options.DEBUG:
            print(f'm0:{M0}, A:{A}, X0:{X0}, B:{B}, C0:{C0}, Q0:{Q0}, J0:{J0}, QD:{QD}, DS:{DS}')
        Y0A = np.array([ M0-A*X0+B*C0, X0, C0 ])
        cosangle = cdAngle(ra, dec, Z[:,6], Z[:,7])
        Zc = Z[:,ZINDEX]
        Q0=qm
        QD=qd
        Qdip = QD*cosangle*np.exp(-1.*Zc/DS)
        Q = Q0 + Qdip
        mu = MUZ(Zc, Q, J0);
        return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - Y0A ) for i in range(N) ] )  
     
 
    if options.EVAL==3:
        print("Evaluating dipole")
        function=RESVF3Dip
    else:
        raise Exception("Invalid EVAL value")
        

    #print('function value',function(M0=-19.3,  A=0.123 ,X0=0.03, B=2.23 , C0=0.04,OM=0.3,OL=0.7))

    def m2loglike(pars , RV = 0):
        
        if RV != 0 and RV != 1 and RV != 2:
            raise ValueError('Inappropriate RV value')
            
        else:  

            cov = COV( *[ pars[i] for i in [1,2,4,5,7] ] )
            try:
                chol_fac = linalg.cho_factor(cov, overwrite_a = True, lower = True ) 
            except np.linalg.linalg.LinAlgError: # If not positive definite
                print("LINALGERRROR error")
                return +13993*10.**20 
            except ValueError: # If contains infinity
                print("Value error")
                return 13995*10.**20

            
            res = RESVF3Dip(*[pars[i] for i,val in enumerate(pars) if i!=1 and i!=4 and i!=7])

            
            part_log = 3*N*np.log(2*np.pi) + np.sum( np.log( np.diag( chol_fac[0] ) ) ) * 2
            try:
                part_exp = np.dot( res, linalg.cho_solve( chol_fac, res) )
            except:
                print("Cholesky solve error")
                print(f'AT location {[pars[i] for i in range(len(pars)) if i != 1 and i != 4 and i != 7]}')
                import sys
                sys.exit(1)
            if pars[-1]<zlim:
                    part_exp += 100* np.sum(np.array([ _**2 for _ in pars ]))   
            if RV==0:
                m2loglike = part_log + part_exp
                if options.DEBUG:
                    print('LogLikelihood:',m2loglike)
                return m2loglike 
            elif RV==1: 
                return part_exp 
            elif RV==2:
                return part_log 
    
    pre_found_best=np.array([-1.915e+01,  1.567e-01  ,1.583e-01, -5.836e-02 , 9.630e-01,
             3.175e+00, -3.845e-02,  5.522e-02 , 9.477e-03, -6.462e-01,
            -3.176e+01 , 9.380e-03])
            
    bounds=[(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(-3,1.5),(-10,10),(None,None),(zlim,None)]
            
    #print(len(pre_found_best),len(bounds))
    #MLE = optimize.minimize(m2loglike, pre_found_best , method = met, tol=10**-14, options={'maxiter':19205000},bounds=bounds) #
    if options.MET ==7:
        options_dict={'maxiter':19205000,'maxfun':123040}
    else:
        options_dict={'maxiter':19205000}
    MLE = optimize.minimize(m2loglike, pre_found_best , method = met, tol=10**-12, options=options_dict,bounds= bounds) #



    return MLE.fun ,name, MLE.success ,MLE.x
                
 

   

INDEX_DCT={9:"HEL",0:"HD",10:"CMB",11:"LG"}
ix=options.ZINDEX  
i=options.ZLIM
if options.MULTIPROC:
    from multiprocessing import Pool, cpu_count
    num_cpus = cpu_count()

    print(f'Number of CPUs available: {num_cpus}')
    qm_values = np.linspace(-0.2,0.4,40)
    qd_values = np.linspace(-30,1,40)
    args_list = [(i, qm, qd) for qm, qd in zip(qm_values, qd_values)]
    
    filename = f'Aniso_qm_qd_{ix}_agebias.txt'
    output_file = Path(filename)
    
    with Pool(processes=num_cpus) as pool:
        results = pool.starmap(Minimizer, args_list)
    
    with open(output_file, 'w') as f:
        for res, qm, qd in zip(results, qm_values, qd_values):
            m=res
            print(f'{qm},{qd},{m[0]},{m[-2]},{m[-1][-1]}')
            output_data = f"{ix},{qm},{qd},{m[0]},{m[-2]},{m[-1][-1]}\n"
            f.write(output_data)
    
    print(f'All results saved to {filename}')
    sys.exit()
else:
    qm=options.ARG1
    qd=options.ARG2
    filename=f'{qm}_{qd}_results.txt'
    print('Runiing the minimizer for QM:',qm,' QD:',qd)
    m=Minimizer(zlim1=i,qm=qm,qd=qd)
    print(f'{options.ARG1},{options.ARG2},{m[0]},{m[-2]},{m[-1][-1]}')
    print('Saving results to ',filename)
    output_file = Path(filename)
    output_data = f"{ix},{options.ARG1},{options.ARG2},{m[0]},{m[-2]},{m[-1][-1]}\n"
    output_file.write_text(output_file.read_text() + output_data if output_file.exists() else output_data)

