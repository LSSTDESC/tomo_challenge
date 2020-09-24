#!/usr/local/bin/python

""" bpz_tools.py: Contains useful functions for I/O and math.
    TO DO:
      Include higher order interpolations
"""

#from Numeric import *
from numpy import *
from MLab_coe import *
from useful import *
from string import *
import os,sys

clight_AHz=2.99792458e18
Vega='Vega_reference'

#Smallest number accepted by python
eps=1e-300
eeps=log(eps)

from MLab_coe import log10 as log10a
def log10(x):
    return log10a(x+eps)
    #return log10clip(x, -33)

#This quantities are used by the AB files
zmax_ab=12.
#zmax_ab=4.
dz_ab=0.01
ab_clip=1e-6


#Initialize path info
bpz_dir=os.getenv('BPZPATH')
fil_dir=bpz_dir+'/FILTER/'
sed_dir=bpz_dir+'/SED/'
ab_dir=bpz_dir+'/AB/'

#Auxiliary synthetic photometry functions

def flux(xsr,ys,yr,ccd='yes',units='nu'):
    """ Flux of spectrum ys observed through response yr,
        both defined on xsr 
        Both f_nu and f_lambda have to be defined over lambda
        If units=nu, it gives f_nu as the output
    """
    if ccd=='yes': yr=yr*xsr
    norm=trapz(yr,xsr)
    f_l=trapz(ys*yr,xsr)/norm
    if units=='nu':
        lp=sqrt(norm/trapz(yr/xsr/xsr,xsr))      #Pivotal wavelenght	
        return f_l*lp**2/clight_AHz
    else: return f_l

def ABtofl(ABmag,filter,ccd='yes'):
    """Converts AB magnitudes to flux in ergs s^-1 cm^-2 AA^-1"""
    lp=pivotal_wl(filter,ccd)
    f=AB2Jy(ABmag)
    return f/lp**2*clight_AHz*1e23

def pivotal_wl(filter,ccd='yes'):
    xr,yr=get_filter(filter)
    if ccd=='yes': yr=yr*xr
    norm=trapz(yr,xr)
    return sqrt(norm/trapz(yr/xr/xr,xr))  

def filter_center(filter,ccd='yes'):
    """Estimates the central wavelenght of the filter"""
    if type(filter)==type(""):
        xr,yr=get_filter(filter)
    else:
        xr=filter[0]
        yr=filter[1]
    if ccd=='yes': yr=yr*xr
    return trapz(yr*xr,xr)/trapz(yr,xr)

def filter_fwhm(filter,ccd='yes'):
    xr,yr=get_filter(filter)
    if ccd=='yes': yr=yr*xr/mean(xr)
    imax=argmax(yr)
    ymax=yr[imax]
    xmax=xr[imax]
    ih_1=argmin(abs(yr[:imax]-ymax/2.))
    ih_2=argmin(abs(yr[imax:]-ymax/2.))+imax
    return xr[ih_2]-xr[ih_1]

def AB(flux):
    """AB magnitude from f_nu"""
    return -2.5*log10(flux)-48.60

def flux2mag(flux):
    """Convert arbitrary flux to magnitude"""
    return -2.5*log10(flux) 

def Jy2AB(flux):
    """Convert flux in Jy to AB magnitudes"""
    return -2.5*log10(flux*1e-23)-48.60

def AB2Jy(ABmag):
    """Convert AB magnitudes to Jansky"""
    return 10.**(-0.4*(ABmag+48.60))/1e-23


def mag2flux(mag):
    """Convert flux to arbitrary flux units"""
    return 10.**(-.4*mag)

def e_frac2mag(fracerr):
    """Convert fractionary flux error to mag error"""
    return 2.5*log10(1.+fracerr)

def e_mag2frac(errmag):
    """Convert mag error to fractionary flux error"""
    return 10.**(.4*errmag)-1.

def flux_det(aperture,pixelnoise,s2noise=1):
    """Given an aperture, the noise per pixel and the 
       signal to noise, it estimates the detection flux limit"""
    npixels=pi*(aperture/2.)**2
    totalnoise=sqrt(npixels)*pixelnoise
    return s2noise*totalnoise

def get_limitingmagnitude(m,dm,sigma=1.,dm_int=0.2):
    """Given a list of magnitudes and magnitude errors,
    calculate by extrapolation the 1-sigma error limit"""
    g=less(m,99.)*greater(m,-99.)
    x,y=autobin_stats(compress(g,m),compress(g,dm),n_points=11,stat="median")
    return match_resol(y,x,dm_int)-flux2mag(1./sigma/e_mag2frac(dm_int))


#Synthetic photometry functions

def etau_madau(wl,z):
    """
    Madau 1995 extinction for a galaxy spectrum at redshift z 
    defined on a wavelenght grid wl
    """
    n=len(wl)
    l=array([1216.,1026.,973.,950.])
    xe=1.+z

    #If all the spectrum is redder than (1+z)*wl_lyman_alfa 
    if wl[0]> l[0]*xe: return zeros(n)+1.

    #Madau coefficients
    c=array([3.6e-3,1.7e-3,1.2e-3,9.3e-4])
    ll=912.
    tau=wl*0.
    i1=searchsorted(wl,ll)
    i2=n-1
    #Lyman series absorption
    for i in range(len(l)):
        i2=searchsorted(wl[i1:i2],l[i]*xe)
        tau[i1:i2]=tau[i1:i2]+c[i]*(wl[i1:i2]/l[i])**3.46

    if ll*xe < wl[0]:
        return exp(-tau)

    #Photoelectric absorption
    xe=1.+z
    i2=searchsorted(wl,ll*xe)
    xc=wl[i1:i2]/ll
    xc3=xc**3
    tau[i1:i2]=tau[i1:i2]+\
                (0.25*xc3*(xe**.46-xc**0.46)\
                 +9.4*xc**1.5*(xe**0.18-xc**0.18)\
                 -0.7*xc3*(xc**(-1.32)-xe**(-1.32))\
                 -0.023*(xe**1.68-xc**1.68))

    tau = clip(tau, 0, 700)
    return exp(-tau)
    # if tau>700. : return 0.
    # else: return exp(-tau)


def etau(wl,z):
    """
    Madau 1995 and Scott 2000 extinction for a galaxy spectrum
    at redshift z observed on a wavelenght grid wl
    """

    n=len(wl)
    l=array([1216.,1026.,973.,950.])
    xe=1.+z

    #If all the spectrum is redder than (1+z)*wl_lyman_alfa 
    if wl[0]> l[0]*xe: return zeros(n)+1.

    #Extinction coefficients

    c=array([1.,0.47,0.33,0.26])
    if z>4.:
        #Numbers from Madau paper
        coeff=0.0036
        gamma=2.46
    elif z<3:
        #Numbers from Scott et al. 2000 paper
        coeff=0.00759
        gamma=1.35
    else:
        #Interpolate between two numbers
        coeff=.00759+(0.0036-0.00759)*(z-3.)
        gamma=1.35+(2.46-1.35)*(z-3.)
    c=coeff*c

    ll=912.
    tau=wl*0.
    i1=searchsorted(wl,ll)
    i2=n-1
    #Lyman series absorption
    for i in range(len(l)):
        i2=searchsorted(wl[i1:i2],l[i]*xe)
        tau[i1:i2]=tau[i1:i2]+c[i]*(wl[i1:i2]/l[i])**(1.+gamma)

    if ll*xe < wl[0]: return exp(-tau)

    #Photoelectric absorption
    xe=1.+z
    i2=searchsorted(wl,ll*xe)
    xc=wl[i1:i2]/ll
    xc3=xc**3
    tau[i1:i2]=tau[i1:i2]+\
                (0.25*xc3*(xe**.46-xc**0.46)\
                +9.4*xc**1.5*(xe**0.18-xc**0.18)\
                -0.7*xc3*(xc**(-1.32)-xe**(-1.32))\
                -0.023*(xe**1.68-xc**1.68))
    return exp(-tau)

def get_sednfilter(sed,filter):
    #Gets a pair of SED and filter from the database
    #And matches the filter resolution to that of the spectrum
    #where they overlap
    """Usage:
    xs,ys,yr=get_sednfilter(sed,filter)
    """
    #Figure out the correct names
    if filter[-4:]!='.res':filter=filter+'.res'
    if sed[-4:]!='.sed':sed=sed+'.sed'
    sed=sed_dir+sed
    filter=fil_dir+filter
    #Get the data
    x_sed,y_sed=get_data(sed,list(range(2)))
    nsed=len(x_sed)
    x_res,y_res=get_data(filter,list(range(2)))
    nres=len(x_res)
    if not ascend(x_sed):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % sed)
        print('They should start with the shortest lambda and end with the longest')        
    if not ascend(x_res):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % filter)
        print('They should start with the shortest lambda and end with the longest')

    #Define the limits of interest in wavelenght
    i1=searchsorted(x_sed,x_res[0])-1
    i1=maximum(i1,0)
    i2=searchsorted(x_sed,x_res[nres-1])+1
    i2=minimum(i2,nsed-1)
    r=match_resol(x_res,y_res,x_sed[i1:i2])
    r=where(less(r,0.),0.,r) #Transmission must be >=0
    return x_sed[i1:i2],y_sed[i1:i2],r

def get_sed(sed):
    #Get x_sed,y_sed from a database spectrum
    """Usage:
    xs,ys=get_sed(sed)
    """
    #Figure out the correct names
    if sed[-4:]!='.sed':sed=sed+'.sed'
    sed=sed_dir+sed
    #Get the data
    x,y=get_data(sed,list(range(2)))
    if not ascend(x):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % sed)
        print('They should start with the shortest lambda and end with the longest')
    return x,y


def get_filter(filter):
    #Get x_res,y_res from a database spectrum
    """Usage:
    xres,yres=get_filter(filter)
    """
    #Figure out the correct names
    if filter[-4:]!='.res':filter=filter+'.res'
    filter=fil_dir+filter
    #Get the data
    x,y= get_data(filter,list(range(2)))
    if not ascend(x):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % filter)
        print('They should start with the shortest lambda and end with the longest')
    return x,y

def redshift(wl,flux,z):
    """ Redshift spectrum y defined on axis x 
      to redshift z
      Usage:
         y_z=redshift(wl,flux,z) 
    """
    if z==0.: return flux
    else: 
        f=match_resol(wl,flux,wl/(1.+z))
        return where(less(f,0.),0.,f)

def normalize(x_sed,y_sed,m,filter='F814W_WFPC2',units='nu'):
    """Normalizes a spectrum (defined on lambda) to 
    a broad band (AB) magnitude and transforms the 
    spectrum to nu units""
    Usage:
    normflux=normalize(wl,spectrum,m,filter='F814W_WFPC2')
    """
    if filter[-4:]!='.res':filter=filter+'.res'
    filter=fil_dir+filter
    x_res,y_res=get_data(filter,list(range(2)))
    nres=len(x_res)
    nsed=len(x_sed)
    i1=searchsorted(x_sed,x_res[0])-1
    i1=maximum(i1,0)
    i2=searchsorted(x_sed,x_res[nres-1])+1
    i2=minimum(i2,nsed-1)
    r=match_resol(x_res,y_res,x_sed[i1:i2])
    r=where(less(r,0.),0.,r) #Transmission must be >=0
    flujo=flux(x_sed[i1:i2],y_sed[i1:i2],r,ccd='yes',units='nu')
    norm=flujo/mag2flux(m)
    if units=='nu':
        return y_sed*x_sed*x_sed/clight_AHz/norm
    else:
        return y_sed/norm

class Normalize:
    def __init__(self,x_sed,y_sed,m,filter='F814W_WFPC2',units='nu'):
        """Normalizes a spectrum (defined on lambda) to 
        a broad band (AB) magnitude and transforms the 
        spectrum to nu units""
        Usage:
        normflux=normalize(wl,spectrum,m,filter='F814W_WFPC2')
        """
        if filter[-4:]!='.res':filter=filter+'.res'
        filter=fil_dir+filter
        x_res,y_res=get_data(filter,list(range(2)))
        nres=len(x_res)
        nsed=len(x_sed)
        i1=searchsorted(x_sed,x_res[0])-1
        i1=maximum(i1,0)
        i2=searchsorted(x_sed,x_res[nres-1])+1
        i2=minimum(i2,nsed-1)
        r=match_resol(x_res,y_res,x_sed[i1:i2])
        r=where(less(r,0.),0.,r) #Transmission must be >=0
        flujo=flux(x_sed[i1:i2],y_sed[i1:i2],r,ccd='yes',units='nu')
        self.norm=flujo/mag2flux(m)
        if units=='nu': self.flux_norm = y_sed*x_sed*x_sed/clight_AHz/self.norm
        else:           self.flux_norm = y_sed/self.norm


def obs_spectrum(sed,z,madau=1):
    """Generate a redshifted and madau extincted spectrum"""
    #Figure out the correct names
    if sed[-4:]!='.sed':sed=sed+'.sed'
    sed=sed_dir+sed
    #Get the data
    x_sed,y_sed=get_data(sed,list(range(2)))
    #ys_z will be the redshifted and corrected spectrum    
    ys_z=match_resol(x_sed,y_sed,x_sed/(1.+z))
    if madau: ys_z=etau_madau(x_sed,z)*ys_z
    return x_sed,ys_z


def nf_z_sed(sed,filter,z=array([0.]),ccd='yes',units='lambda',madau='yes'):
    """Returns array f with f_lambda(z) or f_nu(z) through a given filter 
       Takes into account intergalactic extinction. 
       Flux normalization at each redshift is arbitrary 
    """
    if type(z)==type(0.): z=array([z])

    #Figure out the correct names
    if sed[-4:]!='.sed':sed=sed+'.sed'
    sed=sed_dir+sed
    if filter[-4:]!='.res':filter=filter+'.res'
    filter=fil_dir+filter

    #Get the data
    x_sed,y_sed=get_data(sed,list(range(2)))
    nsed=len(x_sed)
    x_res,y_res=get_data(filter,list(range(2)))
    nres=len(x_res)

    #Wavelenght range of interest as a function of z
    wl_1=x_res[0]/(1.+z)
    wl_2=x_res[-1]/(1.+z)
    n1=clip(searchsorted(x_sed,wl_1)-1,0,1000000)
    n2=clip(searchsorted(x_sed,wl_2)+1,0,nsed-1)

    #Change resolution of filter
    x_r=x_sed[n1[0]:n2[0]]
    r=match_resol(x_res,y_res,x_r)
    r=where(less(r,0.),0.,r) #Transmission must be >=0

    #Operations necessary for normalization and ccd effects
    if ccd=='yes': r=r*x_r
    norm_r=trapz(r,x_r)
    if units=='nu': const=norm_r/trapz(r/x_r/x_r,x_r)/clight_AHz
    else: const=1.
    const=const/norm_r

    nz=len(z)
    f=zeros(nz)*1.
    for i in range(nz):
        i1,i2=n1[i],n2[i]
        ys_z=match_resol(x_sed[i1:i2],y_sed[i1:i2],x_r/(1.+z[i]))
        if madau!='no': ys_z=etau_madau(x_r,z[i])*ys_z
        f[i]=trapz(ys_z*r,x_r)*const        
    if nz==1: return f[0]
    else: return f

def lf_z_sed(sed,filter,z=array([0.]),ccd='yes',units='lambda',madau='yes'):
    """
    Returns array f with f_lambda(z) or f_nu(z) through a given filter 
    Takes into account intergalactic extinction. 
    Flux normalization at each redshift is arbitrary 
    """

    if type(z)==type(0.): z=array([z])

    #Figure out the correct names
    if sed[-4:]!='.sed':sed=sed+'.sed'
    sed=sed_dir+sed
    if filter[-4:]!='.res':filter=filter+'.res'
    filter=fil_dir+filter

    #Get the data
    x_sed,y_sed=get_data(sed,list(range(2)))
    nsed=len(x_sed)
    x_res,y_res=get_data(filter,list(range(2)))
    nres=len(x_res)

    if not ascend(x_sed):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % sed)
        print('They should start with the shortest lambda and end with the longest')        
        print('This will probably crash the program')

    if not ascend(x_res):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % filter)
        print('They should start with the shortest lambda and end with the longest')
        print('This will probably crash the program')

    if x_sed[-1]<x_res[-1]: #The SED does not cover the whole filter interval
        print('Extrapolating the spectrum')
        #Linear extrapolation of the flux using the last 4 points
        #slope=mean(y_sed[-4:]/x_sed[-4:])
        d_extrap=(x_sed[-1]-x_sed[0])/len(x_sed)
        x_extrap=arange(x_sed[-1]+d_extrap,x_res[-1]+d_extrap,d_extrap)
        extrap=lsq(x_sed[-5:],y_sed[-5:])
        y_extrap=extrap.fit(x_extrap)
        y_extrap=clip(y_extrap,0.,max(y_sed[-5:]))
        x_sed=concatenate((x_sed,x_extrap))
        y_sed=concatenate((y_sed,y_extrap))
        #connect(x_sed,y_sed)
        #connect(x_res,y_res)

    #Wavelenght range of interest as a function of z
    wl_1=x_res[0]/(1.+z)
    wl_2=x_res[-1]/(1.+z)
    n1=clip(searchsorted(x_sed,wl_1)-1,0,100000)
    n2=clip(searchsorted(x_sed,wl_2)+1,0,nsed-1)

    #Typical delta lambda
    delta_sed=(x_sed[-1]-x_sed[0])/len(x_sed)
    delta_res=(x_res[-1]-x_res[0])/len(x_res)

    #Change resolution of filter
    if delta_res>delta_sed:
        x_r=arange(x_res[0],x_res[-1]+delta_sed,delta_sed)
        #print 'Changing filter resolution from %.2f AA to %.2f AA' % (delta_res,delta_sed)
        r=match_resol(x_res,y_res,x_r)
        r=where(less(r,0.),0.,r) #Transmission must be >=0
    else:
        x_r,r=x_res,y_res

    #Operations necessary for normalization and ccd effects
    if ccd=='yes': r=r*x_r
    norm_r=trapz(r,x_r)
    if units=='nu': const=norm_r/trapz(r/x_r/x_r,x_r)/clight_AHz
    else: const=1.

    const=const/norm_r

    nz=len(z)
    f=zeros(nz)*1.
    for i in range(nz):
        i1,i2=n1[i],n2[i]
        ys_z=match_resol(x_sed[i1:i2],y_sed[i1:i2],x_r/(1.+z[i]))
        #p=FramedPlot();p.add(Curve(x_r,ys_z));p.show()
        if madau!='no': ys_z=etau_madau(x_r,z[i])*ys_z
        #pp=FramedPlot();pp.add(Curve(x_r,ys_z*etau_madau(x_r,z[i])));pp.show()
        #ask('More?')
        f[i]=trapz(ys_z*r,x_r)*const        
    if nz==1: return f[0]
    else: return f


def of_z_sed(sed,filter,z=array([0.]),ccd='yes',units='lambda',madau='yes'):
    """Returns array f with f_lambda(z) or f_nu(z) through a given filter 
       Takes into account intergalactic extinction. 
       Flux normalization at each redshift is arbitrary 
    """
    if type(z)==type(0.): z=array([z])

    #Figure out the correct names
    if sed[-4:]!='.sed':sed=sed+'.sed'
    sed=sed_dir+sed
    if filter[-4:]!='.res':filter=filter+'.res'
    filter=fil_dir+filter

    #Get the data
    x_sed,y_sed=get_data(sed,list(range(2)))
    nsed=len(x_sed)
    x_res,y_res=get_data(filter,list(range(2)))
    nres=len(x_res)

    #Define the limits of interest in wl
    i1=searchsorted(x_sed,x_res[0])-1
    i1=maximum(i1,0)
    i2=searchsorted(x_sed,x_res[-1])+1
    i2=minimum(i2,nsed-1)
    if x_sed[-1]<x_res[-1]: #The SED does not cover the whole filter interval
        #Linear extrapolation of the flux using the last 4 points
        #slope=mean(y_sed[-4:]/x_sed[-4:])
        d_extrap=(x_sed[-1]-x_sed[0])/len(x_sed)
        x_extrap=arange(x_sed[-1]+d_extrap,x_res[-1]+d_extrap,d_extrap)
        extrap=lsq(x_sed[-5:],y_sed[-5:])
        y_extrap=extrap.fit(x_extrap)
        y_extrap=clip(y_extrap,0.,max(y_sed[-5:]))
        x_sed=concatenate((x_sed,x_extrap))
        y_sed=concatenate((y_sed,y_extrap))
        i2=len(y_sed)-1
    r=match_resol(x_res,y_res,x_sed[i1:i2])
    r=where(less(r,0.),0.,r) #Transmission must be >=0
    nz=len(z)
    f=zeros(nz)*1.
    for i in range(nz):
        ys_z=match_resol(x_sed,y_sed,x_sed/(1.+z[i]))
        if madau!='no': ys_z[i1:i2]=etau_madau(x_sed[i1:i2],z[i])*ys_z[i1:i2]
        f[i]=flux(x_sed[i1:i2],ys_z[i1:i2],r,ccd,units)
    if nz==1: return f[0]
    else: return f


f_z_sed=lf_z_sed
#f_z_sed=nf_z_sed
#f_z_sed=of_z_sed


def f_z_sed_AB(sed,filter,z=array([0.]),units='lambda'):
    #It assumes ccd=yes,madau=yes by default
    z_ab=arange(0.,zmax_ab,dz_ab) #zmax_ab and dz_ab are def. in bpz_tools
    lp=pivotal_wl(filter)

    #AB filter

    #Figure out the correct names
    if sed[-4:]!='.sed':sed=sed+'.sed'
    ab_file=ab_dir+sed[:-4]+'.'
    if filter[-4:]!='.res':filter=filter+'.res'
    ab_file+=filter[:-4]+'.AB'
    #print 'AB file',ab_file
    if not os.path.exists(ab_file):
        ABflux(sed,filter)
    z_ab,f_ab=get_data(ab_file,list(range(2)))
    fnu=match_resol(z_ab,f_ab,z)
    if units=='nu':      return fnu
    elif units=='lambda': return fnu/lp**2*clight_AHz
    else:
        print('Units not valid')


def ABflux(sed,filter,madau='yes'):
    """
    Calculates a AB file like the ones used by bpz
    It will set to zero all fluxes
    which are ab_clip times smaller than the maximum flux.
    This eliminates residual flux which gives absurd
    colors at very high-z
    """

    print(sed, filter)
    ccd='yes'
    units='nu'
    madau=madau
    z_ab=arange(0.,zmax_ab,dz_ab) #zmax_ab and dz_ab are def. in bpz_tools

    #Figure out the correct names
    if sed[-4:]!='.sed':sed=sed+'.sed'
    sed=sed_dir+sed
    if filter[-4:]!='.res':filter=filter+'.res'
    filter=fil_dir+filter

    #Get the data
    x_sed,y_sed=get_data(sed,list(range(2)))
    nsed=len(x_sed)
    x_res,y_res=get_data(filter,list(range(2)))
    nres=len(x_res)

    if not ascend(x_sed):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % sed)
        print('They should start with the shortest lambda and end with the longest')        
        print('This will probably crash the program')

    if not ascend(x_res):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % filter)
        print('They should start with the shortest lambda and end with the longest')
        print('This will probably crash the program')

    if x_sed[-1]<x_res[-1]: #The SED does not cover the whole filter interval
        print('Extrapolating the spectrum')
        #Linear extrapolation of the flux using the last 4 points
        #slope=mean(y_sed[-4:]/x_sed[-4:])
        d_extrap=(x_sed[-1]-x_sed[0])/len(x_sed)
        x_extrap=arange(x_sed[-1]+d_extrap,x_res[-1]+d_extrap,d_extrap)
        extrap=lsq(x_sed[-5:],y_sed[-5:])
        y_extrap=extrap.fit(x_extrap)
        y_extrap=clip(y_extrap,0.,max(y_sed[-5:]))
        x_sed=concatenate((x_sed,x_extrap))
        y_sed=concatenate((y_sed,y_extrap))
        #connect(x_sed,y_sed)
        #connect(x_res,y_res)

    #Wavelenght range of interest as a function of z_ab
    wl_1=x_res[0]/(1.+z_ab)
    wl_2=x_res[-1]/(1.+z_ab)
    #print 'wl', wl_1, wl_2
    #print 'x_res', x_res
    print('x_res[0]', x_res[0])
    print('x_res[-1]', x_res[-1])
    n1=clip(searchsorted(x_sed,wl_1)-1,0,100000)
    n2=clip(searchsorted(x_sed,wl_2)+1,0,nsed-1)

    #Typical delta lambda
    delta_sed=(x_sed[-1]-x_sed[0])/len(x_sed)
    delta_res=(x_res[-1]-x_res[0])/len(x_res)


    #Change resolution of filter
    if delta_res>delta_sed:
        x_r=arange(x_res[0],x_res[-1]+delta_sed,delta_sed)
        print('Changing filter resolution from %.2f AA to %.2f' % (delta_res,delta_sed))
        r=match_resol(x_res,y_res,x_r)
        r=where(less(r,0.),0.,r) #Transmission must be >=0
    else:
        x_r,r=x_res,y_res

    #Operations necessary for normalization and ccd effects
    if ccd=='yes': r=r*x_r
    norm_r=trapz(r,x_r)
    if units=='nu': const=norm_r/trapz(r/x_r/x_r,x_r)/clight_AHz
    else: const=1.

    const=const/norm_r

    nz_ab=len(z_ab)
    f=zeros(nz_ab)*1.
    for i in range(nz_ab):
        i1,i2=n1[i],n2[i]
        #if (x_sed[i1] > max(x_r/(1.+z_ab[i]))) or (x_sed[i2] < min(x_r/(1.+z_ab[i]))):
        if (x_sed[i1] > x_r[-1]/(1.+z_ab[i])) or (x_sed[i2-1] < x_r[0]/(1.+z_ab[i])) or (i2-i1<2):
            print('bpz_tools.ABflux:')
            print("YOUR FILTER RANGE DOESN'T OVERLAP AT ALL WITH THE REDSHIFTED TEMPLATE")
            print("THIS REDSHIFT IS OFF LIMITS TO YOU:")
            print('z = ', z_ab[i])
            print(i1, i2)
            print(x_sed[i1], x_sed[i2])
            print(y_sed[i1], y_sed[i2])
            print(min(x_r/(1.+z_ab[i])), max(x_r/(1.+z_ab[i])))
            # NOTE: x_sed[i1:i2] NEEDS TO COVER x_r(1.+z_ab[i])
            # IF THEY DON'T OVERLAP AT ALL, THE PROGRAM WILL CRASH
            #sys.exit(1)
        else:
            try:
                ys_z=match_resol(x_sed[i1:i2],y_sed[i1:i2],x_r/(1.+z_ab[i]))
            except:
                print(i1, i2)
                print(x_sed[i1], x_sed[i2-1])
                print(y_sed[i1], y_sed[i2-1])
                print(min(x_r/(1.+z_ab[i])), max(x_r/(1.+z_ab[i])))
                print(x_r[1]/(1.+z_ab[i]), x_r[-2]/(1.+z_ab[i]))
                print(x_sed[i1:i2])
                print(x_r/(1.+z_ab[i]))
                pause()
            if madau!='no': ys_z=etau_madau(x_r,z_ab[i])*ys_z
            f[i]=trapz(ys_z*r,x_r)*const        

    ABoutput=ab_dir+split(sed,'/')[-1][:-4]+'.'+split(filter,'/')[-1][:-4]+'.AB'

    #print "Clipping the AB file"
    #fmax=max(f)
    #f=where(less(f,fmax*ab_clip),0.,f)

    print('Writing AB file ',ABoutput)
    put_data(ABoutput,(z_ab,f))


def VegatoAB(m_vega,filter,Vega=Vega):
    cons=AB(f_z_sed(Vega,filter,z=0.,units='nu',ccd='yes'))
    return m_vega+cons

def ABtoVega(m_ab,filter,Vega=Vega):
    cons=AB(f_z_sed(Vega,filter,z=0.,units='nu',ccd='yes'))
    return m_ab-cons

#Photometric redshift functions

def likelihood(f,ef,ft_z):
    """ 
    Usage: ps[:nz,:nt]=likelihood(f[:nf],ef[:nf],ft_z[:nz,:nt,:nf])
    """
    global minchi2
    axis=ft_z.shape
    nz=axis[0]
    nt=axis[1]

    chi2=zeros((nz,nt),float)
    ftt=zeros((nz,nt),float)
    fgt=zeros((nz,nt),float)

    ief2=1./(ef*ef)
    fgg=add.reduce(f*f*ief2)
    factor=ft_z[:nz,:nt,:]*ief2

    ftt[:nz,:nt]=add.reduce(ft_z[:nz,:nt,:]*factor,-1)
    fgt[:nz,:nt]=add.reduce(f[:]*factor,-1)
    chi2[:nz,:nt]=fgg-power(fgt[:nz,:nt],2)/ftt[:nz,:nt]

    min_chi2=min(chi2)
    minchi2=min(min_chi2)
#   chi2=chi2-minchi2
    chi2=clip(chi2,0.,-2.*eeps)

    p=where(greater_equal(chi2,-2.*eeps),0.,exp(-chi2/2.))

    norm=add.reduce(add.reduce(p))
    return p/norm

def new_likelihood(f,ef,ft_z):
    """ 
    Usage: ps[:nz,:nt]=likelihood(f[:nf],ef[:nf],ft_z[:nz,:nt,:nf])
    """
    global minchi2
    rolex=reloj()
    rolex.set()
    nz,nt,nf=ft_z.shape

    foo=add.reduce((f/ef)**2)

    fgt=add.reduce(
        #f[NewAxis,NewAxis,:nf]*ft_z[:nz,:nt,:nf]/ef[NewAxis,NewAxis,:nf]**2
        reshape(f, (1, 1, nf)) * ft_z / reshape(ef, (1, 1, nf))**2
        ,-1)

    ftt=add.reduce(
        #ft_z[:nz,:nt,:nf]*ft_z[:nz,:nt,:nf]/ef[NewAxis,NewAxis,:nf]**2
        ft_z**2 / reshape(ef, (1, 1, nf))**2
        ,-1)

    ao=fgt/ftt
#    print mean(ao),std(ao)

    chi2=foo-fgt**2/ftt+(1.-ao)**2*ftt

    minchi2=min(min(chi2))
    chi2=chi2-minchi2
    chi2=clip(chi2,0.,-2.*eeps)
    p=exp(-chi2/2.)
    norm=add.reduce(add.reduce(p))
    return p/norm

#class p_c_z_t:
#    def __init__(self,f,ef,ft_z):
#	self.nz,self.nt,self.nf=ft_z.shape
#	self.foo=add.reduce((f/ef)**2)
#	self.fgt=add.reduce(
#	    f[NewAxis,NewAxis,:]*ft_z[:,:,:]/ef[NewAxis,NewAxis,:]**2
#	    ,-1)
#	self.ftt=add.reduce(
#	    ft_z[:,:,:]*ft_z[:,:,:]/ef[NewAxis,NewAxis,:]**2
#	    ,-1)
#        #When all the model fluxes are equal to zero
#        self.chi2=self.foo-(self.fgt**2+1e-100)/(self.ftt+1e-100)
#	self.chi2_minima=loc2d(self.chi2[:self.nz,:self.nt],'min')
#	self.i_z_ml=self.chi2_minima[0]
#	self.i_t_ml=self.chi2_minima[1]
#	self.min_chi2=self.chi2[self.i_z_ml,self.i_t_ml]
#	self.likelihood=exp(-0.5*clip((self.chi2-self.min_chi2),0.,1400.))
#        self.likelihood=where(equal(self.chi2,1400.),0.,self.likelihood)
#        #Add the f_tt^-1/2 multiplicative factor in the exponential
#        self.chi2+=-0.5*log(self.ftt+1e-100)
#        min_chi2=min(min(self.chi2))
#	self.Bayes_likelihood=exp(-0.5*clip((self.chi2-min_chi2),0.,1400.))
#        self.Bayes_likelihood=where(equal(self.chi2,1400.),0.,self.Bayes_likelihood)
#        
#        #plo=FramedPlot()
#        #for i in range(self.ftt.shape[1]):
#        #    norm=sqrt(max(self.ftt[:,i]))
#        #    # plo.add(Curve(arange(self.ftt.shape[0]),self.ftt[:,i]**(0.5)))
#        #    plo.add(Curve(arange(self.ftt.shape[0]),self.likelihood[:,i],color='red'))
#        #    plo.add(Curve(arange(self.ftt.shape[0]),self.likelihood[:,i]*sqrt(self.ftt[:,i])/norm))
#        #plo.show()#
#
#    def bayes_likelihood(self):
#        return self.Bayes_likelihood


class p_c_z_t:
    def __init__(self,f,ef,ft_z):
        self.nz,self.nt,self.nf=ft_z.shape
        #Get true minimum of the input data (excluding zero values)
        #maximo=max(f)
        #minimo=min(where(equal(f,0.),maximo,f))
        #print 'minimo=',minimo        
        #maximo=max(ft_z)
        #minimo=min(min(min(where(equal(ft_z,0.),maximo,ft_z))))
        #print 'minimo=',minimo
        #minerror=min(f)
        #maxerror=max(ef)
        #maxerror=max(where(equal(ef,maxerror),minerror,ef))
        #print 'maxerror',maxerror

        #Define likelihood quantities taking into account non-observed objects
        self.foo=add.reduce(where(less(f/ef,1e-4),0.,(f/ef)**2))
        #nonobs=less(f[NewAxis,NewAxis,:]/ef[NewAxis,NewAxis,:]+ft_z[:,:,:]*0.,1e-4)
        #nonobs=less(reshape(f, (1, 1, self.nf)) / reshape(ef, (1, 1, self.nf)) + ft_z*0., 1e-4)
        # Above was wrong: non-detections were ignored as non-observed --DC
        nonobs=greater(reshape(ef, (1, 1, self.nf)) + ft_z*0., 1.0)
        self.fot=add.reduce(
            #where(nonobs,0.,f[NewAxis,NewAxis,:]*ft_z[:,:,:]/ef[NewAxis,NewAxis,:]**2)
            where(nonobs,0.,reshape(f, (1, 1, self.nf)) * ft_z / reshape(ef, (1, 1, self.nf))**2)
            ,-1)
        self.ftt=add.reduce(
            #where(nonobs,0.,ft_z[:,:,:]*ft_z[:,:,:]/ef[NewAxis,NewAxis,:]**2)
            where(nonobs,0.,ft_z**2 / reshape(ef, (1, 1, self.nf))**2)
            ,-1)

        #############################################
        #Old definitions    
        #############################################
        #self.foo=add.reduce((f/ef)**2)
        #self.fot=add.reduce(
        #    f[NewAxis,NewAxis,:]*ft_z[:,:,:]/ef[NewAxis,NewAxis,:]**2
        #    ,-1)
        #self.ftt=add.reduce(
        #    ft_z[:,:,:]*ft_z[:,:,:]/ef[NewAxis,NewAxis,:]**2
        #    ,-1)
        ################################################

        #Define chi2 adding eps to the ftt denominator to avoid overflows
        self.chi2=where(equal(self.ftt,0.),
                        self.foo,
                        self.foo-(self.fot**2)/(self.ftt+eps))
        self.chi2_minima=loc2d(self.chi2[:self.nz,:self.nt],'min')
        self.i_z_ml=int(self.chi2_minima[0])
        self.i_t_ml=int(self.chi2_minima[1])
        self.min_chi2=self.chi2[self.i_z_ml,self.i_t_ml]
        self.likelihood=exp(-0.5*clip((self.chi2-self.min_chi2),0.,-2*eeps))
        #self.likelihood=where(equal(self.chi2,1400.),0.,self.likelihood)

        #Now we add the Bayesian f_tt^-1/2 multiplicative factor to the exponential
        #(we don't multiply it by 0.5 since it is done below together with the chi^2
        #To deal with zero values of ftt we again add an epsilon value.
        self.expo=where(
            equal(self.ftt,0.),
            self.chi2,
            self.chi2+log(self.ftt+eps)
            )
        #Renormalize the exponent to preserve dynamical range
        self.expo_minima=array(loc2d(self.expo,'min'), dtype=int)
        self.min_expo=self.expo[self.expo_minima[0],self.expo_minima[1]]
        self.expo-=self.min_expo
        self.expo=clip(self.expo,0.,-2.*eeps)
        #Clip very low values of the probability
        self.Bayes_likelihood=where(
            equal(self.expo,-2.*eeps),
            0.,
            exp(-0.5*self.expo))

    def bayes_likelihood(self):
        return self.Bayes_likelihood

    def various_plots(self):        
        #Normalize and collapse likelihoods (without prior)
        norm=add.reduce(add.reduce(self.Bayes_likelihood))
        bl=add.reduce(self.Bayes_likelihood/norm,-1)
        norm=add.reduce(add.reduce(self.likelihood))
        l=add.reduce(self.likelihood/norm,-1)
        plo=FramedPlot()
        plo.add(Curve(arange(self.nz),bl,color='blue'))
        plo.add(Curve(arange(self.nz),l,color='red'))
        plo.show()


        #plo2=FramedPlot()
        #for i in range(self.ftt.shape[1]):
        #for i in range(2):
        #    #plo2.add(Curve(arange(self.nz),log(self.fot[:,i]*self.fot[:,i])))
        #    plo2.add(Curve(arange(self.nz),log(self.ftt[:,i])))
        #plo2.show()


        #for i in range(self.ftt.shape[1]):
        #for i in range(2):
        #    plo2.add(Curve(arange(self.nz),-0.5*(self.fot[:,i]*self.fot[:,i]/self.ftt[:,i]+log(self.ftt[:,i]))))
        #    plo2.add(Curve(arange(self.nz),-0.5*(self.fot[:,i]*self.fot[:,i]/self.ftt[:,i]),color='red'))
        #plo2.show()

        #plo3=FramedPlot()
        #for i in range(self.ftt.shape[1]):
        #    norm=sqrt(max(self.ftt[:,i]))
        #    plo3.add(Curve(arange(self.nz),self.fot[:,i]*self.fot[:,i]/self.ftt[:,i]))
        #plo3.show()


        ask('More?')


class p_c_z_t_color:
    def __init__(self,f,ef,ft_z):
        self.nz,self.nt,self.nf=ft_z.shape
        self.chi2=add.reduce(
        #((f[NewAxis,NewAxis,:]-ft_z[:,:,:])/ef[NewAxis,NewAxis,:])**2
        ((reshape(f, (1, 1, self.nf)) - ft_z) / reshape(ef, (1, 1, self.nf)))**2
        ,-1)
        self.chi2_minima=loc2d(self.chi2[:self.nz,:self.nt],'min')
        self.i_z_ml=self.chi2_minima[0]
        self.i_t_ml=self.chi2_minima[1]
        self.min_chi2=self.chi2[self.i_z_ml,self.i_t_ml]
        self.likelihood=exp(-0.5*clip((self.chi2-self.min_chi2),0.,1400.))
    def bayes_likelihood(self):
        return self.likelihood

#def gr_likelihood(f,ef,ft_z):
#    #Color-redshift Likelihood a la Rychards et al. (SDSS QSOs)
#    global minchi2
#    nf=f.shape[0]
#    nz=ft_z.shape[0]
#    nt=ft_z.shape[1]
#    print f,ef,ft_z[:10,0,:]
#    chi2=add.reduce(
#	((f[NewAxis,NewAxis,:nf]-ft_z[:nz,:nt,:nf])/ef[NewAxis,NewAxis,:nf])**2
#	,-1)
#    minchi2=min(min(chi2))
#    chi2=chi2-minchi2
#    chi2=clip(chi2,0.,1400.)
#    p=exp(-chi2/2.)
#    norm=add.reduce(add.reduce(p))
#    return p/norm

#def p_and_minchi2(f,ef,ft_z):
#    p=gr_likelihood(f,ef,ft_z)
#    return p,minchi2

#def new_p_and_minchi2(f,ef,ct):
#    p=color_likelihood(f,ef,ct)
#    return p,minchi2

def prior(z,m,info='hdfn',nt=6,ninterp=0,x=None,y=None):
    """Given the magnitude m, produces the prior  p(z|T,m)
    Usage: pi[:nz,:nt]=prior(z[:nz],m,info=('hdfn',nt))
    """    
    if info=='none' or info=='flat': return
    #We estimate the priors at m_step intervals
    #and keep them in a dictionary, and then
    #interpolate them for other values
    m_step=0.1
    accuracy=str(len(str(int(1./m_step)))-1)#number of decimals kept

    exec("from prior_{} import *".format(info), globals())
    global prior_dict
    try:
        len(prior_dict)
    except NameError:
        prior_dict={}

    #The dictionary keys are values of the 
    #magnitud quantized to mstep mags
    #The values of the dictionary are the corresponding
    #prior probabilities.They are only calculated once 
    #and kept in the dictionary for future
    #use if needed. 
    forma='%.'+accuracy+'f'
    m_dict=forma %m    
    if m_dict not in prior_dict or info=='lensing': #if lensing, the magnitude alone is not enough
        if info!='lensing':
            prior_dict[m_dict]=function(z,float(m_dict),nt)
        else:
            prior_dict[m_dict]=function(z,float(m_dict),nt,x,y)                    
        if ninterp:
            pp_i=prior_dict[m_dict]
            nz=pp_i.shape[0]
            nt=pp_i.shape[1]
            nti=nt+(nt-1)*int(ninterp)
            tipos=arange(nt)*1.
            itipos=arange(nti)*1./(1.+float(ninterp))
            buffer=zeros((nz,nti))*1.
            for iz in range(nz):
                buffer[iz,:]=match_resol(tipos,pp_i[iz,:],itipos)
            prior_dict[m_dict]=buffer
    return prior_dict[m_dict]

def interval(p,x,ci=.99):
    """Gives the limits of the confidence interval
       enclosing ci of the total probability
       i1,i2=limits(p,0.99)
    """
    q1=(1.-ci)/2.
    q2=1.-q1
    cp=add.accumulate(p)
    if cp[-1]!=1.: cp=cp/cp[-1]
    i1=searchsorted(cp,q1)-1
    i2=searchsorted(cp,q2)
    i2=minimum(i2,len(p)-1)
    i1=maximum(i1,0)
    return x[i1],x[i2] 

def odds(p,x,x1,x2):
    """Estimate the fraction of the total probability p(x)
    enclosed by the interval x1,x2"""
    cp=add.accumulate(p)
    i1=searchsorted(x,x1)-1
    i2=searchsorted(x,x2)
    if i1<0:
        return cp[i2]/cp[-1]
    if i2>len(x)-1:
        return 1.-cp[i1]/cp[-1]
    return (cp[i2]-cp[i1])/cp[-1]


class p_bayes:
    #This class reads the information contained in the files produced by BPZ
    #when the option -PROBS_LITE is on
    def __init__(self,file):
        self.file=file
        dummy=get_2Darray(file)
        self.id_list=list(map(int,list(dummy[:,0])))
        self.p=dummy[:,1:]
        del(dummy)
        header=get_header(file)
        header=split(header,'(')[2]
        header=split(header,')')[0]
        zmin,zmax,dz=list(map(float,tuple(split(header,','))))
        self.z=arange(zmin,zmax,dz)

    def plot_p(self,id,limits=None):
        if type(id)!=type((1,)):
            try:
                j=self.id_list.index(int(id))
                p_j=self.p[j,:]
                if limits==None:
                    connect(self.z,p_j)
                else:   
                    connect(self.z,p_j,limits)
            except:
                print('Object %i not in the file %s' % (id,self.file))
            self.prob=p_j/max(p_j)
        else:
            p=FramedPlot()
            p.frame1.draw_grid=1
            pall=self.p[0,:]*0.+1.
            pmax=0.
            if limits!=None:
                p.xrange=limits[0],limits[1]
                p.yrange=limits[2],limits[3]
            for i in id:
                try:
                    j=self.id_list.index(int(i))
                    p_j=self.p[j,:]
                    if max(p_j)>pmax: pmax=max(p_j)
                    pall*=p_j
                    p.add(Curve(self.z,p_j))
                except:
                    print('Object %i not in the file %s' % (id,self.file))
            p.add(Curve(self.z,pall/max(pall)*pmax,color='red'))
            p.show()
            self.prob=pall/max(pall)

    def maxima(self,limits=(0.,6.5)):
        g=greater_equal(self.z,limits[0])*less_equal(self.z,limits[1])
        z,p=multicompress(g,(self.z,self.prob))
        imax=argmax(p)
        xp=add.accumulate(p)
        xp/=xp[-1]
        self.q66=match_resol(xp,z,array([0.17,0.83]))
        self.q90=match_resol(xp,z,array([0.05,0.95]))
        #print self.q66
        #print self.q90
        return z[imax]




    #def hist_p(self,dz=0.25):
    #    self.pt=sum(self.p)
    #    self.xz=arange(self.z[0],self.z[-1]+dz,dz)
    #    self.hb=bin_stats(self.z,self.pt,self.xz,'sum')

#Misc stuff

def get_datasex(file,cols,purge=1,mag=(2,99.),emag=(4,.44),flag=(24,4),detcal='none'):
    """
      Usage:
      x,y,mag,emag=get_datasex('file.cat',(0,1,24,12))
      If purge=1, the function returns the corresponding columns 
      of a SExtractor output file, excluding those objects with 
      magnitude <mag[1], magnitude error <=emag[1] and flag <=flag[1]
      mag[0],emag[0] and flag[0] indicate the columns listing 
      these quantities in the file
      detcal: a detection image was used to create the catalog
      It will be used now to determine which objects are good.
    """
    if type(cols)==type(0): nvar=1
    else:nvar=len(cols)

    if purge:
        if nvar>1:datos=get_2Darray(file,cols)
        else: datos=get_data(file,cols)
        if detcal=='none': detcal=file
        m,em,f=get_data(detcal,(mag[0],emag[0],flag[0]))
        good=less_equal(f,flag[1])*less_equal(em,emag[1])*less(m,mag[1])
        datos=compress(good,datos,0)
        lista=[]
        if nvar>1:
            for i in range(datos.shape[1]):lista.append(datos[:,i])
            return tuple(lista)
        else: return datos
    else:
        return get_data(file,cols)

def sex2bpzmags(f,ef,zp=0.,sn_min=1.,m_lim=None):
    """
    This function converts a pair of flux, error flux measurements from SExtractor
    into a pair of magnitude, magnitude error which conform to BPZ input standards:
    - Nondetections are characterized as mag=99, errormag=m_1sigma
    - Objects with absurd flux/flux error combinations or very large errors are
      characterized as mag=-99 errormag=0.
    """

    nondetected=less_equal(f,0.)*greater(ef,0) #Flux <=0, meaningful phot. error
    nonobserved=less_equal(ef,0.) #Negative errors
    #Clip the flux values to avoid overflows
    f=clip(f,1e-100,1e10)
    ef=clip(ef,1e-100,1e10)
    nonobserved+=equal(ef,1e10)
    nondetected+=less_equal(f/ef,sn_min) #Less than sn_min sigma detections: consider non-detections

    detected=logical_not(nondetected+nonobserved)

    m=zeros(len(f))*1.
    em=zeros(len(ef))*1.

    m = where(detected,-2.5*log10(f)+zp,m)
    m = where(nondetected,99.,m)
    m = where(nonobserved,-99.,m)

    em = where(detected,2.5*log10(1.+ef/f),em)
    if not m_lim:
        em = where(nondetected,-2.5*log10(ef)+zp,em)
    else:
        em = where(nondetected,m_lim,em)        
    em = where(nonobserved,0.,em)
    return m,em


class bpz_diagnosis:
    def __init__(self,bpz_file='/home/txitxo/bpz/TEST/hdfn.bpz',
                 columns=(1,4,5,6,9,10)):
        # columns correspond to the positions of the variables z_b,odds,z_ml, z_s and m_0
        # in the bpz file
        self.zb,self.tb,self.odds,self.zm,self.zs,self.mo=get_data(bpz_file,columns)
    def stats(self,type='rms',
              odds_min=0.99,
              mo_min=0.,mo_max=99.,
              zs_min=0.,zs_max=6.5,
              t_min=0,t_max=100,
              plots='yes',
              thr=.2):
        good=greater_equal(self.mo,mo_min)
        good*=less_equal(self.mo,mo_max)
        good*=greater_equal(self.zs,zs_min)
        good*=less_equal(self.zs,zs_max)
        good*=greater_equal(self.tb,t_min)
        good*=less_equal(self.tb,t_max)
        self.n_total=len(good)
        self.good=good*greater_equal(self.odds,odds_min)
        self.n_selected=sum(self.good)
        self.d=compress(self.good,(self.zb-self.zs)/(1.+self.zs))
        #try: self.std_thr=std_thr(self.d,thr)
        #except: self.std_thr=1e10
        #try: self.med_thr=med_thr(self.d,thr)
        #except: self.med_thr=1
        #try: self.n_thr=self.n_selected-out_thr(self.d,thr)
        #except: self.n_thr=0
        b=stat_robust(self.d,3,5)
        b.run()
        self.n_remaining=b.n_remaining
        self.n_outliers=b.n_outliers
        self.rms=b.rms
        self.med=b.median
        self.std_log=std_log(self.d)
        if plots=='yes':
            #points(compress(self.good,self.zs),compress(self.good,self.zb),(0.,zs_max,0.,zs_max))
            p=FramedPlot()
            xmin=min(compress(self.good,self.zs))
            xmax=max(compress(self.good,self.zs))
            print(xmin,xmax)
            x=arange(xmin,xmax,.01)
            p.add(Curve(x,x,width=3))
            p.add(Curve(x,x+3.*self.rms*(1.+x),width=1))
            p.add(Curve(x,x-3.*self.rms*(1.+x),width=1))
            p.add(Points(compress(self.good,self.zs),compress(self.good,self.zb)))
            p.xlabel=r"$z_{spec}$"
            p.ylabel=r"$z_b$"
            p.show()
            if ask("save plot?"):
                name=input("name?")
                p.write_eps(name)


class bpz_diagnosis_old:
    #This class characterized the quality of a bpz run by comparing
    #the output with the input spectroscopic redshifts

    def __init__(self,bpz_file='/home/txitxo/bpz/TEST/hdfn.bpz',
                 columns=(1,5,6,9,10)):
        #columns correspond to the positions of the variables z_b,odds,z_ml, z_s and m_0
        #in the bpz file
        self.zb,self.odds,self.zm,self.zs,self.mo=get_data(bpz_file,columns)
#        print self.zb[:10],self.odds[:10],self.zm[:10],self.zs[:10],self.mo[:10]

    def stats(self,odds_thr=(0.95,0.99),z_lim=(0.,10.)):
        #z_lim selects in the *estimated* quantities
        #parameters controlling the 'purging' of outliers
        d_thr=3. #Should remove only 1% of all points
        n=5
        #Produce stats characterizing the quality of the results
        #rms and fraction of outliers for zm: rms_zm, fo_zm
        if z_lim[0]!=0. or z_lim[1]!=8.:
            good_b=greater_equal(self.zb,z_lim[0])*less_equal(self.zb,z_lim[1])
            good_m=greater_equal(self.zm,z_lim[0])*less_equal(self.zm,z_lim[1])
            good_s=greater_equal(self.zs,z_lim[0])*less_equal(self.zs,z_lim[1])
            zb,zsb,oddsb=multicompress(good_b,(self.zb,self.zs,self.odds))
            zm,zsm=multicompress(good_m,(self.zm,self.zs))
            nzs=sum(good_s)
        else:
            zb=self.zb
            zm=self.zm
            zsb=self.zs
            oddsb=self.odds
            zsm=zsb
            nzs=len(self.zs)
        dzb=(zb-zsb)/(1.+zsb)
        dzm=(zm-zsm)/(1.+zsm)
        nzb=len(dzb)
        nzm=len(dzm)

        greater_equal(self.zm,z_lim[0])*less_equal(self.zm,z_lim[1])

        print("Number of galaxies selected using zs=",nzs)
        print("Number of galaxies selected using zb=",nzb)
        print("Number of galaxies selected using zm=",nzm)

        #ZB
#        zb_stat=stat_robust(dzb,d_thr,n)
#        zb_stat.run()
#        mean_zb,rms_zb,n_out_zb,frac_zb=\
#          zb_stat.mean,zb_stat.rms,zb_stat.n_outliers,zb_stat.fraction
#        print "Z_B vs Z_S"
#        print "<z_b-z_s>/(1+zs)=%.4f, rms=%.4f, n_outliers=%i, fraction outliers=%.2f" %\
#              (mean_zb,rms_zb,n_out_zb,frac_zb)

        #ZM
#        zm_stat=stat_robust(dzm,d_thr,n)
#        zm_stat.run()
#        mean_zm,rms_zm,n_out_zm,frac_zm=\
#          zm_stat.mean,zm_stat.rms,zm_stat.n_outliers,zm_stat.fraction
#        print "Z_M vs Z_S"
#        print "<z_m-z_s>/(1+zs)=%.4f, rms=%.4f, n_outliers=%i, fraction outliers=%.2f" %\
#              (mean_zm,rms_zm,n_out_zm,frac_zm)

        #Total Fraction of zm with dz larger than rms_zb
#        f_zm_rms_zb=sum(greater(abs(self.dzm),3*rms_zb))/ float(self.nz)
#        print "Fraction of zm with |zm-zs|/(1+zs) > 3*rms_dzb= %.2f" % f_zm_rms_zb

        #Total Fraction of zb with dz larger than rms_zm
#        f_zb_rms_zm=sum(greater(abs(self.dzb),3.*rms_zm))/ float(self.nz)
#        print "Fraction of zb with |zb-zs|/(1+zs) > 3*rms_dzm= %.2f" % f_zb_rms_zm

        #Total Fraction of zb with dz larger than 0.06(1+zs)
        f_zb_0p06=sum(greater(abs(dzb),3*0.06))/ float(nzb)
        print("Fraction of zb with <zb-zs> > 3*0.06(1+z)= %.2f" % f_zb_0p06)

        #Total Fraction of zm with dz larger than 0.06(1+zs)
        f_zm_0p06=sum(greater(abs(dzm),3*0.06))/ float(nzm)
        print("Fraction of zm with <zm-zs> > 3*0.06(1+z)= %.2f" % f_zm_0p06)

        print("\nSelect objects using odds thresholds\n")
        for i in range(len(odds_thr)):            
            goodo=greater_equal(oddsb,odds_thr[i])
            print("# of objects with odds > %.2f = %.2f " % (odds_thr[i],sum(goodo)))
            zbo,zso=multicompress(goodo,(zb,zsb))
            dzbo=(zbo-zso)/(1.+zso)
            zbo_stat=stat_robust(dzbo,d_thr,n)
            zbo_stat.run()
            mean_zbo,rms_zbo,n_out_zbo,frac_zbo=\
               zbo_stat.mean,zbo_stat.rms,zbo_stat.n_outliers,zbo_stat.fraction
            print("     Z_BO vs Z_S")
            print("     <z_bo-z_s>=%.4f, rms=%.4f, n_outliers=%i, fraction outliers=%.2f" %\
                  (mean_zbo,rms_zbo,n_out_zbo,frac_zbo))

            #Total Fraction of zb with dz larger than 0.06(1+zs)
            f_zbo_0p06=sum(greater(abs(dzbo),3*0.06))/ float(len(dzbo))
            print("Fraction of zbo with <zbo-zso> > 3*0.06(1+z)= %.2f" % f_zbo_0p06)

            #Plot
            p=FramedPlot()
            p.add(Points(zbo,zso,type='circle'))
            p.xlabel=r"$z_s$"
            p.ylabel=r"$z_b$"
            p.add(Slope(1.,type='dotted'))
            p.show()
            p.write_eps('plot_'+str(odds_thr[i])+'.eps')

            #Otroplot
            p=FramedPlot()
            xz=arange(-1.,1.,0.05)
            hd=hist(dzbo,xz)
            p.add(Histogram(hd,xz[0],0.05))
            p.show()

            #Completeness fractions as a function of redshift
            #Odds fractions as a function of magnitude

    def plots(self):
        pass
        #Produce main plots

    def webpage(self):
        pass
        #Produce a webpage with a summary of the numeric estimates and plots

def test():
    """ Tests some functions defined in this module"""

    test='flux'
    Testing(test)

    x=arange(912.,10001.,.1)
    r=exp(-(x-3500.)**2/2./200.**2)
    f=1.+sin(x/100.)

    e_ccd=add.reduce(f*r*x)/add.reduce(r*x)
    e_noccd=add.reduce(f*r)/add.reduce(r)

    r_ccd=flux(x,f,r,ccd='yes',units='lambda')
    r_noccd=flux(x,f,r,ccd='no',units='lambda')

    if abs(1.-e_ccd/r_ccd)>1e-6 or abs(1.-e_noccd/r_noccd)>1e-6: raise ValueError(test)

    #print '        f_lambda          '
    #print 'Results                  Expected'
    #print 'CCD ',r_ccd,e_ccd
    #print 'No CCD ',r_noccd,e_noccd

    nu=arange(1./x[-1],1./x[0],1./x[0]/1e2)*clight_AHz
    fn=(1.+sin(clight_AHz/100./nu))*clight_AHz/nu/nu
    xn=clight_AHz/nu
    rn=match_resol(x,r,xn)
    e_ccd=add.reduce(fn*rn/nu)/add.reduce(rn/nu)
    e_noccd=add.reduce(fn*rn)/add.reduce(rn)
    r_ccd=flux(x,f,r,ccd='yes',units='nu')
    r_noccd=flux(x,f,r,ccd='no',units='nu')

    #print '           f_nu           '
    #print 'Results                  Expected'
    #print 'CCD',r_ccd,e_ccd
    #print 'no CCD',r_noccd,e_noccd

    if abs(1.-e_ccd/r_ccd)>1e-6 or abs(1.-e_noccd/r_noccd)>1e-6: raise ValueError(test)

    test='AB'
    Testing(test)
    if AB(10.**(-.4*48.60))!=0.: raise ValueError(test)

    test='flux2mag and mag2flux'
    Testing(test)
    m,f=20.,1e-8
    if mag2flux(m)!=f: raise ValueError(test)
    if flux2mag(f)!=m: raise ValueError(test)

    test='e_frac2mag and e_mag2frac'
    Testing(test)
    f=1e8
    df=1e7/f
    m=flux2mag(f)
    dm=m-flux2mag(f*(1.+df))
    if abs(e_frac2mag(df)-dm)>1e-12: 
        print(abs(e_frac2mag(df)-dm))
        raise ValueError(test)
    if abs(e_mag2frac(dm)-df)>1e-12: 
        print(e_mag2frac(dm),df)
        raise ValueError(test)

    test='etau_madau'
    #Un posible test es generar un plot de la absorpcion a distintos redshifts
    #igual que el que viene en el paper de Madau.

    test='f_z_sed'
    Testing(test)
    #Estimate fluxes at different redshift for a galaxy with a f_nu\propto \nu spectrum
    # (No K correction) and check that their colors are constant
    x=arange(1.,10001.,10.)
    f=1./x
    put_data(sed_dir+'test.sed',(x,f))
    z=arange(0.,10.,.25)
    b=f_z_sed('test','B_Johnson.res',z,ccd='no',units='nu',madau='no')
    v=f_z_sed('test','V_Johnson.res',z,ccd='no',units='nu',madau='no')
    c=array(list(map(flux2mag,b/v)))
    if(sometrue(greater(abs(c-c[0]),1e-4))): 
        print(c-c[0])
        raise ValueError(test)

    test='VegatoAB' # To be done
    test='ABtoVega'
    test='likelihood'

    #Test: generar un catalogo de galaxias con colores, e intentar recuperar 
    #sus redshifts de nuevo utilizando solo la likelihood

    test='p_and_minchi2' # To be done
    test='prior'

    test='interval'
    test='odds'

    test=' the accuracy of our Johnson-Cousins-Landolt Vega-based zero-points'
    Testing(test)
    #filters=['U_Johnson.res','B_Johnson.res','V_Johnson.res','R_Cousins.res',
    #'I_Cousins.res']

    filters=[
        'HST_ACS_WFC_F435W',
        'HST_ACS_WFC_F475W',
        'HST_ACS_WFC_F555W',
        'HST_ACS_WFC_F606W',
        'HST_ACS_WFC_F625W',
        'HST_ACS_WFC_F775W',
        'HST_ACS_WFC_F814W',
        'HST_ACS_WFC_F850LP'
        ]


    ab_synphot=array([
        -0.10719,
        -0.10038,
        8.743e-4,
        0.095004,
        0.174949,
        0.40119,
        0.44478,
        0.568605
        ])


    f_l_vega=array([
        6.462e-9,
        5.297e-9,
        3.780e-9,
        2.850e-9,
        2.330e-9,
        1.270e-9,
        1.111e-9,
        7.78e-10])


    print('     f_l for Vega')
    sufix='cgs A^-1'
    print('                               f_lambda(Vega)     synphot(IRAF)   difference %')
    for i in range(len(filters)):
        f_vega=f_z_sed(Vega,filters[i],ccd='yes')
        tupla=(ljust(filters[i],16),f_vega,f_l_vega[i],f_vega/f_l_vega[i]*100.-100.)
        print('     %s         %.6e       %.6e      %.4f'%tupla +"%")


    print('    ')
    print('    AB zeropoints for Vega ')
    sufix='cgs Hz'
    tipo='nu'
    print("                                AB zero point     synphot(IRAF)   difference")
    for i in range(len(filters)):
        f_vega=f_z_sed(Vega,filters[i],units=tipo,ccd='yes')
        tupla=(ljust(filters[i],16),AB(f_vega),ab_synphot[i],AB(f_vega)-ab_synphot[i])
        print('     %s         %.6f       %.6f      %.6f' % tupla)


    print('    ')
    print('    AB zeropoints for a c/lambda^2 spectrum (flat in nu)')
    sufix='cgs Hz'
    tipo='nu'
    print("                                 Result             Expected  ")
    for i in range(len(filters)):
        f_flat=f_z_sed('flat',filters[i],units=tipo,ccd='yes')
        tupla=(ljust(filters[i],16),AB(f_flat),0.)
        print('     %s         %.6e       %.6f' % tupla)


    print('')
    print('         Everything OK    in   bpz_tools ')
    print('')


if __name__ == '__main__':
    test()
else:
    pass
#    print 'bpz_tools loaded as module'
