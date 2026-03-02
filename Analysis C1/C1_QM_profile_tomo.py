
import time
import numpy as np
from scipy import linalg ,optimize  ,integrate
from optparse import OptionParser
import pandas as pd
from astropy.coordinates import angular_separation as ang 
import astropy.units as u
import sys
import pickle
usage = 'usage: %prog [options]'
parser = OptionParser(usage)
#import ray
import multiprocessing
###note before using the code-

#use of option parser
# -e or --evaluate options sets what function inimization is to be done for exampe a lambda cdm or kinematic taylor expansion.
#-d or --details options sets the functional form of dipole
# -m or --method is the method used for minimization
# -r or --reversebias reverses bias corrections on the magnitude
# --fixastro fixes the astrophysical parameters for when shell analysis is to be done 
# -t or --taylor remove supernovae above 0.8
# --dipoledir fixes dipole direction preferrred direction (right now it ony has one option i.e CMB dipole direction)
# --dipole can be used when you need to find dipole direction as well with the minimzed parameters
# --scand is used for estimating confidence intervals on paramters only in shell analysis 


parser.add_option( "-e","--evaluate", action = "store", dest="EVAL", type='int',default=3, help = "What evaluation needs to be done, 1: LCDM , 2: Kinematic Taylor, 3:  Dipole in deceleration parameter Kinematic  , 4:  Dipole in Hubble parameter Kinematic, 5: H0 as well as Q0 dipole ?, 6: Qduadropolar Hubble analysis?")
parser.add_option("-d", "--details", action="store", type="int", default=4,dest="DET", help="1: Do pheno Q fit with JLA only. 2: Fit for a non scale dependent dipolar modulation in Q. 3: Fit for a top hat scale dependent dipolar modulation in Q. 4. Fit for an exponentially falling scale dependent dipolar modulation in Q. 5. Fit for a linearly falling scale dependent dipolar modulation in Q. ")
parser.add_option("-m", "--method", action="store", type="int", default=7, dest="MET", help="1:Nelder-Mead 2: SLSQP 3: Powell.4: Trust-Constraint 5: TNC 6: Cobyla 7: L-BFGS-B 8: Newton-CG 9: BFGS 10: CG 11:trust-exact")
parser.add_option( "-z","--redshift", action = "store",type="int", default=7, dest="ZINDEX", help = "Which Redshift to use  7: zHEL 8: zCMB 0: zHD 9: zLG? ")
parser.add_option("-r", "--reversebias", action = "store_true", default=True, dest="REVB", help = "Reverse the bias corrections")

parser.add_option( "--fixastro", action = "store", dest="FIXA", default=0, help = " 0: Doesnt fix astrophysical paramete,( Use only for shell analysis ) fix astrophysical paramters for 1: deceleration paramter 2: hubble parameter?")
parser.add_option("-t", "--taylor", action = "store_true", default=True, dest="TAY", help = "Remove high z redhsit for taylor analysis")
parser.add_option( "--dipoledir", action = "store", type="int", default=1, dest="DIPDIR", help = "DIPOLE DIRECTION FIX TO? 1:CMB directon ")
parser.add_option( "--dipole", action = "store_true", default=False, dest="DIP", help = "Need to estimate dipole direction too?")
parser.add_option("--schand", action = "store_true", default=False, dest="SCANSHELLDIPOLE")

parser.add_option("--fixpars", action = "store_true", default=False, dest="FIXPARS",help='Fixing parmater values for hubble parameter estimation? ')
# if fixpars is TRUE it fixes parameter values to following 
parser.add_option("--aa", action = "store",type="float", default=0.15, dest="A0")
parser.add_option("--bb", action = "store",type="float", default=3, dest="B0")
parser.add_option("--xx", action = "store",type="float", default=-0.075, dest="X0")
parser.add_option("--sM", action = "store",type="float", default=0.2, dest="sM0")
parser.add_option("--sC", action = "store",type="float", default=0.055, dest="sC0")
parser.add_option("--sX", action = "store",type="float", default=0.965, dest="sX0")
parser.add_option("--cc", action = "store",type="float", default=-0.042, dest="C0")

parser.add_option("--age_bias", action="store_true", default=False, dest="AGEBIAS", help="Apply progenitor age bias correction")
parser.add_option("--age_bias_statistics",action='store',type = "string", default="median", dest="AGEBAST", help="Use mean or median progenitor age bias correction")

(options, args) = parser.parse_args()
   

dctdet={2:"Non scalar",3:"Flat",4:"Exponential",5:"Linear",6:'Power',8:'mix'}

if options.DET==2:
    STYPE='NoScDep'
elif options.DET==3:
    STYPE='Flat'
elif options.DET==4:
    STYPE='Exp'
elif options.DET==5:
    STYPE='Lin'
elif options.DET==6:
    STYPE='Power'
elif options.DET==8:
    STYPE='mix'
else:
    STYPE='None'
    
if not options.DIP:
    if options.DIPDIR==1:
        print("Dipole Direction is CMB")
    elif options.DIPDIR==2:
        print("Dipole Direction is LG-SUN")

else:
    print("Floating Dipole")

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

print("Using Method: ",met,"\nUsing profile: ",dctdet[options.DET] )



shell_width=100


if options.EVAL==1:
    evalu='lkly'
elif options.EVAL==2:
    evalu='RESVF3'
elif options.EVAL==3:
    evalu='RESVF3Dip'
elif options.EVAL==4:
    evalu="RESVF3_H0Dip"
elif options.EVAL==5:
    evalu="RESVF3Q0_H0dip"
elif options.EVAL==6:
    evalu="RESVF3Quad"
c = 299792.458 # km/s
H0 = 70 #(km/s) / Mpc

CMBdipdec = -7
CMBdipra = 169

LGra= 164.6877348

LGdec= -17.1115932
LGvel=620 #km/s
vcmb=369
racmb=168
deccmb=-7

raSUN_LG= 333.53277784*u.deg
decSUN_LG = 49.33111122*u.deg
vLG_SUN= 299 #km/s


raLG_SUN= 153.53277784*u.deg
decLG_SUN = -49.33111122*u.deg

H0=70
path='/Storage/animesh/DATA_PPLUS/'

if options.FIXA>0 and options.DET!=2:
    print('Error: need to give correction functional form for shell analysis')
    sys.exit(1)

if options.AGEBIAS:
    print("Applying progenitor age bias correction using ",options.AGEBAST," statistics")
    with open(f'{path}cs_'+options.AGEBAST+'.pkl', 'rb') as f:
        cs_bias = pickle.load(f)    

# ray.init()
# @ray.remote
def Minimizer(arg1,zlim=0):
    radip=0
    decdip=0
    Z = pd.read_csv( f'{path}/Z_mbcorr.csv' )
    df=pd.read_csv(f'{path}/Pantheon+SH0ES.dat',delimiter=' ')
    name=str(met)
    if options.REVB: ##reversing bias corrrections
        print ('reversing bias')
        Z['m_b_corr'] = Z['m_b_corr'] + df['biasCor_m_b']
    if options.AGEBIAS:
        print ('applying progenitor age bias correction')
        Z['m_b_corr'] = Z['m_b_corr'] - 0.03*cs_bias(Z['zHEL'])
    #Z=Z.sort_values('zHEL')
    
    #Z=Z[Z['zHEL']<0.8]
    Z=Z[Z['zHEL']<0.8]
    Z=Z.sort_values('zHEL')

    ra=np.array(Z['RA'])*u.deg
    dec=np.array(Z['DEC'])*u.deg
    
    
    zDF1=np.sqrt((1-vLG_SUN*np.cos(ang(raSUN_LG,decSUN_LG,ra,dec)).value/c)/(1+vLG_SUN*np.cos(ang(raSUN_LG,decSUN_LG,ra,dec)).value/c))-1
    zLG=((1+Z['zHEL'])/(1+zDF1))-1           ##using redshift addition
    Z['zLG']=zLG


    if options.TAY:
        print("Removing higher redshifts")
        Z=Z[Z['zHEL']<0.8]
        name+='TAYLOR_EXP_'
    
        
    
    status=" "

    zlim1 = zlim
    status="Using zlim_"+str(zlim1)
    print("Using the zlim provided",zlim1)
    Z=Z[Z['zHEL']>zlim1]



    l=Z.index.values.tolist()  
    l=np.array(l)
    med= np.median(Z['zHEL'])
    Z=Z.to_numpy()
    ZINDEX=options.ZINDEX
    
    
    zlim=Z[0,ZINDEX]
    dct={0:'zHD',7:'zHEL',8:'zCMB',9:'zLG'}
    
    print("USING redshift: ",dct[ZINDEX] )
    N=Z.shape[0]
    name=name+"Z="+str(dct[ZINDEX])

    if not options.DIP:
        name=name+"_FIX_DIP_TO_"
        if options.DIPDIR==1:
            
            radip=CMBdipra
            decdip=CMBdipdec
            
            name=name+"CMB_"

    else:
        name=name+"_FLOAT_DIP"
    

    if ZINDEX==9:   #cmb dipole direction in LG frame
        radip= 162.95389715
        decdip=-25.96734154   # values taken from Planck 2018 paper
    
    print(len(Z))
    def func(Zc,OM,OL,zh=None,zp=None):
            OK= 1.-OM-OL
            def I (z):
                return 1./np.sqrt(OM*(1+z)**3+OL+OK*(1+z)**2)
            if OK==0:
                integ=integrate.quad(I,0,Zc)[0]
            elif OK>0:
                integ= (1./OK)**0.5 *np.sinh(integrate.quad(I,0,Zc)[0]*OK**(0.5))
            elif OK<0:
                integ= (-1./OK)**0.5 *np.sin(integrate.quad(I,0,Zc)[0]*(-OK)**(0.5))
            if zp is not None:
                return (1.+zp)*(1+zh)*integ
            elif zh is not None:
                return (1.+zh)*integ
                            
            return (1.+Zc)*integ
    def dL_lcdm(Zc, OM, OL, Zh=None, Zp=None):
        if Zp is not None:
            return np.hstack([func(zc, OM, OL, zh, zp) for zc, zh, zp in zip(Zc, Zh, Zp)])
        elif Zh is not None:
            return np.hstack([func(zc, OM, OL, zh) for zc, zh in zip(Zc, Zh)])
        return np.hstack([func(z, OM, OL) for z in Zc])




    def MUZ(Zc, Q0, J0,S0=None,OK=None,L0=None):

        k = 5.*np.log10( c/H0 * dLPhenoF3(Zc, Q0, J0)) + 25.   
        if np.any(np.isnan(k)):
            k[np.isnan(k)] = 63.15861331456834
        return k
    def MUZ_H0Dip(Zc, Q0, J0,H0):
        k = 5.*np.log10( c/H0 * dLPhenoF3(Zc, Q0, J0)) + 25.  
        if np.any(np.isnan(k)):
            k[np.isnan(k)] = 63.15861331456834
        return k
    def dLPhenoF3(z, q0, j0):
        return z*(1.+0.5*(1.-q0)*z -1./6.*(1. - q0 - 3.*q0**2. + j0)*z**2.)*(1+Z[:,7])/(1+z)
    

    
    COVd = np.load( f'{path}statsys_mbcorr.npy' ) [:,l][l]# Constructing data covariance matrix w/ sys.
    
    
    #for n1 in np.random.randint(0,1683,10):

     #   print('Z is: ',Z[:,7][n1],'\n COV elements are: \n', COVd[:,n1][n1])
    def COV(sM=0.045,RV=0):
        #sM=0.045
        #sM=9.495e-02
        #sM=8.531e-02
        #sM=1e-8
        #sM=8.531e-02
        ## for shell analysis
        
        if options.FIXA>0:
            if ZINDEX==7:
                #pass
                sM=8.531e-02
            elif ZINDEX==8:
                sM=5.776e-02
            elif ZINDEX==9:
                sM=9.495e-02
        
        COVl=np.diag((sM**2)*np.ones(N))
     
        if RV==0: 
            return np.array(COVl+COVd)
        elif RV==1:
            return np.array( COVd )
        elif RV==2:        
            return np.array(COVl)
        
    def cdAngle(ra1, dec1, ra2, dec2):
        return np.cos(np.deg2rad(dec1))*np.cos(np.deg2rad(dec2))*np.cos(np.deg2rad(ra1) - np.deg2rad(ra2))+np.sin(np.deg2rad(dec1))*np.sin(np.deg2rad(dec2))
        

    


                
    def RESVF3Dip( M0,Q0, J0 , QD, DS=np.inf,ra=radip,dec=decdip, stype = STYPE): #Total residual, \hat Z - Y_0*A
        #QD=0
        Q0 = arg1
        
        
        Y0A = np.array([ M0])
        cosangle = cdAngle(ra,dec, Z[:,5], Z[:,6])
        Zc = Z[:,ZINDEX]
        newdf=pd.DataFrame(np.array([Z[:,5],Z[:,6],Zc]).T,columns=['RA' , 'DEC','zHEL'])
        newdf.to_csv('altered_df.csv')
        if stype=='NoScDep':
            Q = Q0 + QD*cosangle
        elif stype=='Flat':
            Qdip = QD*cosangle
            
            Qdip[Zc>(DS+0.1)] = 0
            Qdip[Zc>DS] = Qdip[Zc>DS]*np.exp(-1.*(Zc[Zc>DS]-DS)/0.03) #minimizer steps are too small to probe an actual top hat
            Q = Q0 + Qdip
        elif stype=='Exp':

            Qdip = QD*cosangle*np.exp(-1.*Zc/DS)
            Q = Q0 + Qdip

        elif stype=='Lin':
            Qd = QD - Zc*DS
            Qd[Qd<0] = 0
            Q = Q0 + Qd*cosangle
        elif stype=='Power':
            Qd = QD*cosangle/(1+Zc)
            Q = Q0 + Qd*cosangle
            
        mu = MUZ(Zc, Q, J0) ;
           
        return np.hstack( [ (Z[i,1] -np.array([mu[i]]) - Y0A ) for i in range(N) ] )
    

            

    function=RESVF3Dip


    #name+='_'+str(function)  
    def m2loglike(pars , RV = 0):
        if RV != 0 and RV != 1 and RV != 2:
            raise ValueError('Inappropriate RV value')
        else:
            if options.FIXA==0:
                 cov = COV( *[ pars[i] for i in [1] ] )
                
            else:
                 cov = COV()

                 
            try:
                chol_fac = linalg.cho_factor(cov, overwrite_a = True, lower = True )
            except np.linalg.linalg.LinAlgError: # If not positive definite
                return +13993*10.**20 
            except ValueError: # If contains infinity
                return 13995*10.**20
            

            if  options.FIXA==0:
                res=function(*[pars[i] for i,val in enumerate(pars) if i!=1])
            else:
                res = function(*[pars[i] for i,val in enumerate(pars) if i!=6 ])

            part_log = N*np.log(2*np.pi) + np.sum( np.log( np.diag( chol_fac[0] ) ) ) * 2
            
            part_exp = np.dot( res, linalg.cho_solve( chol_fac, res) )
            if  options.EVAL>0:
                if options.DET >2  and options.FIXA==0:
                    if pars[-1]<0.01:
                        #pass
                        part_exp += 100* np.sum(np.array([ _**2 for _ in pars ]))
            else:
                if pars[0]<0 or pars[1]<0 or pars[1]>1 or pars[0]>1:
                    pass
                    #part_exp += 100* np.sum(np.array([ _**2 for _ in pars ]))
            if RV==0:
                m2loglike = part_log + part_exp
                #print(pars,m2loglike)
                return m2loglike 
            elif RV==1: 
                return part_exp 
            elif RV==2:
                return part_log  

   


    #pre_found_best=[-1.93145269e+01 ,0.03  ]
    pre_found_best=[-1.9145269e+01 ,0.03  ]

    bounds=((-21,-16),(None,None))
    if options.EVAL==1:
        name=name+" LCDM"
        ar=[0.3,0.7]
        new=((None,None),(None,None))
        bounds+=((None,None),(None,None))
        pre_found_best=np.hstack([pre_found_best,ar])
    else: 
        ar=[ -0.35,0.12]
        bounds+=((None,None),(None,None))
        name=name+" TAY"
        pre_found_best=np.hstack([pre_found_best,ar])
        
        if options.EVAL==3:
            ar=[0]
            if options.ZINDEX==8 or options.ZINDEX==0:
           
                bounds+=((-0.1,20),)
            else: 
                bounds+=((-20,0.1),)
            pre_found_best=np.hstack([pre_found_best,ar])
            if  options.DET >1:
                name=name+" F=f(z,s)"
                ar=[0.09]
                bounds+=((zlim,0.4),)
                pre_found_best=np.hstack([pre_found_best,ar])
        elif options.EVAL==4:
            pre_found_best[0]=70
            bounds=list(bounds)
            bounds[0]=(65,75)
            bounds=tuple(bounds)
            ar=[-2]
            name=name+" H0dip"
            bounds+=((None,None),)
            pre_found_best=np.hstack([pre_found_best,ar])
            if options.DET >2:
                name=name+" F=f(z,S)"
                ar=[0.02]
                bounds+=((zlim,1),)
                pre_found_best=np.hstack([pre_found_best,ar])
        
        elif options.EVAL==5:
            ar=[-6,-6]
            bounds+=((None,None),(None,None),)
            pre_found_best=np.hstack([pre_found_best,ar])
            if  options.DET >2:
                name=name+" F=f(z,s)"
                ar=[0.02]
                bounds+=((zlim,None),)
                pre_found_best=np.hstack([pre_found_best,ar])
                if options.DET==4:
                    ar=[0.02]
                    bounds+=((zlim,None),)
                    pre_found_best=np.hstack([pre_found_best,ar])
            if options.DIP:
                ar=[132,35]
                #ar=[20,42]
                name=name+" Estimated Dipole"
                bounds+=((0,360),(-90,90))
                pre_found_best=np.hstack([pre_found_best,ar])
            
                
        elif options.EVAL==6:
            pre_found_best[0]=71.02
            bounds=list(bounds)
            bounds[0]=(65,75)
            bounds=tuple(bounds)
            ar=[-1.671]
            bounds+=((None,None),)
            pre_found_best=np.hstack([pre_found_best,ar])
            
            name=name+" F=f(z,S)"
            ar=[0.07]
            bounds+=((zlim,1),)
            pre_found_best=np.hstack([pre_found_best,ar])
            ar=[0.02]
            bounds+=((zlim,20),)
            pre_found_best=np.hstack([pre_found_best,ar])
            ar=[ 0.0253 , -0.0078]
            bounds+=((None,None),(None,None))
            pre_found_best=np.hstack([pre_found_best,ar])
        
        if options.DIP:
            ar=[133,33]
     
            name=name+" Estimated Dipole"
            bounds+=((0,360),(-90,90))
            pre_found_best=np.hstack([pre_found_best,ar])
        
        if options.FIXA>0:
            temp=options.FIXA
            
            pre_found_best=[-1,-19.3]
            bounds=((None,None),(None,None))
            if options.FIXA==2:
                pre_found_best=[0,70]
                bounds=((None,None),(None,None))
                
            if options.DIP:
                pre_found_best=[-1,70,220,-54,-0.55,1,0.01]
                bounds=((None,None),(None,None),(0,360),(-90,90),(None,None),(None,None),(None,None))
        
    
    def No_dip(pars):
        return pars[0]- dip
    constraints=({'type':'eq','fun':No_dip})
    print(bounds)
    print(pre_found_best)
    

        
    MLE = optimize.minimize(m2loglike, pre_found_best , method = met, tol=10**-12 , options={'maxiter':150000},bounds = bounds)



     
    return MLE.fun #,MLE.x

Z = pd.read_csv( f'{path}/Z_mbcorr.csv' )
Z=Z.sort_values('zHEL')

if options.TAY:
    Z=Z[Z['zHEL']<0.8]
N=len(Z)

ZINDEX_MAPPING={0:'HD',7:'HEL',8:'CMB',9:'LG'}

true_vals = pd.read_csv(f'{ZINDEX_MAPPING[options.ZINDEX]}_C1.csv')





a=time.time()


# We need to remove ray dependency and redefine Minimizer as a regular function
# Since Minimizer is defined with @ray.remote, we need to call its underlying logic
# We'll use a wrapper that calls the original function's logic

def minimizer_wrapper(args):
    qv, zlim = args
    result = Minimizer.remote(qv, zlim=zlim)
    return ray.get(result)

# Actually, let's avoid ray entirely and define a plain function wrapper
# We need to unwrap the ray.remote decorator. Since Minimizer was defined with @ray.remote,
# we can access the original function via Minimizer._function if needed,
# but it's simpler to just call it as a regular function using _function attribute.
def mp_minimizer(args):
    qv, zlim = args
    return Minimizer(qv, zlim=zlim)

profile=[]
for zlim in true_vals['zlim']:
    if zlim in [0.10762,0.06976,0.05057,0.03486]:
    #if zlim in [0.10762]:
        print("Working on zlim: ",zlim)
        qm_best= true_vals['qm'][true_vals['zlim']==zlim].values[0]
        qm_vary = np.linspace(qm_best - 0.25, qm_best + 0.25, 30)
        task_args = [(qv, zlim) for qv in qm_vary]
        with multiprocessing.Pool() as pool:
            res = pool.map(mp_minimizer, task_args)
        profile.append([qm_vary,res])

b  = time.time()
print(profile)

# Create a dataframe with qm_vary, res and zlim columns
data_rows = []
ix= 0
for i, zlim in enumerate(true_vals['zlim']):
    if zlim in [0.10762,0.06976,0.05057,0.03486]:

        qm_vary = profile[ix][0]
        res = profile[ix][1]
        for qv, r in zip(qm_vary, res):
            data_rows.append({'zlim': zlim, 'qm_vary': qv, 'res': r})
        ix+=1

result_df = pd.DataFrame(data_rows)
filename = f"QM_profile_tomo_{options.ZINDEX}_results_v2.txt"
result_df.to_csv(filename, sep=' ', index=False)
print("Time taken: ",b-a)
print('Results saved to ',filename)

    




