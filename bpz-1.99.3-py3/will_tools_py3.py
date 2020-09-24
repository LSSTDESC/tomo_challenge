import numpy as np
import astropy.io.fits as pyfits
from spline import *
from types import *
import os,sys
from numpy import *
from MLab_coe import *
from scipy.interpolate import interp1d
from string import *  # AFTER numpy, WHICH HAS ITS OWN split, (and more?)

# imports are  mess and should be tidied up!!!



#Files containing strings

def get_str(file,cols=0,nrows='all'):
    """ 
        Reads strings from a file
        Usage: 
             x,y,z=get_str('myfile.cat',(0,1,2))
        x,y,z are returned as string lists
    """
    if type(cols)==type(0):
        cols=(cols,)
        nvar=1
    else: nvar=len(cols)
    lista=[]
    for i in range(nvar): lista.append([])
    buffer=open(file).readlines() 
    if nrows=='all': nrows=len(buffer)
    counter=0
    for lines in buffer:
        if counter>=nrows : break
        if lines[0]=='#': continue
        pieces=lines.split()
        if len(pieces)==0: continue
        for j in range(nvar):lista[j].append(pieces[cols[j]])
        counter=counter+1
    if nvar==1: return lista[0]
    else: return tuple(lista) 



# read 2D array using fits (topcat style, 1st extension)

def get_2Darray_fromfits(filename,cols='all',nrows='all',verbose='no'):
    data = pyfits.open(filename)[1].data
    if cols=='all':
        # just convert data to a 2Darray
        array_out = np.array(data)
    else:
        nc=len(cols)
        array_out = None
        for col in cols:
            array_in = data[col]
            array_out = np.hstack((array_out,array_in))
        array_out = array_out[1:].reshape((nc,len(array_in))).T
    return np.array(array_out,dtype=float)

def get_long_fromfits(filename,col='COADD_OBJECTS_ID',nrows='all',verbose='no'):
    data = pyfits.open(filename)[1].data
    try:
        array_in = data[col]
    except:
        print(('Column not found:'+col))
    return np.array(array_in, dtype=int)

def get_AB_data(file,cols=0,nrows='all'):
    """ Returns data in the columns defined by the tuple
    (or single integer) cols as a tuple of float arrays 
    (or a single float array)"""
    if type(cols)==type(0):
        cols=(cols,)
        nvar=1
    else: nvar=len(cols)
    data=get_str(file,cols,nrows)
    if nvar==1: return array(list(map(float,data)))
    else:
        data=list(data)
        for j in range(nvar): data[j]=array(list(map(float,data[j])))
        return data[0], tuple(data[1:]) 


def match_resol(xg,yg,xf,method="linear"):
    """ 
    Interpolates and/or extrapolate yg, defined on xg, onto the xf coordinate set.
    Options are 'linear' or 'spline' (uses spline.py from Johan Hibscham)
    Usage:
    ygn=match_resol(xg,yg,xf,'spline')
    """
    if method!="spline":
        if type(xf)==type(1.): xf=array([xf])
        ng=len(xg)
        # print argmin(xg[1:]-xg[0:-1]),min(xg[1:]-xg[0:-1]),xg[argmin(xg[1:]-xg[0:-1])]
        d=(yg[1:]-yg[0:-1])/(xg[1:]-xg[0:-1])
        #Get positions of the new x coordinates
        ind=clip(searchsorted(xg,xf)-1,0,ng-2)
        ygn=take(yg,ind)+take(d,ind)*(xf-take(xg,ind))
        if len(ygn)==1: ygn=ygn[0]
        return ygn
    else:
        low_slope=(yg[1]-yg[0])/(xg[1]-xg[0])
        high_slope=(yg[-1]-yg[-2])/(xg[-1]-xg[-2])
        sp=Spline(xg,yg,low_slope,high_slope)
        return sp(xf)	


def make_AB_dict(f_mod, spine_mags, targ_mags_def, method="linear"):
    """
    Build dictionary of model fluxes, where key is the apparent mag.
    First, build a big grid of interpolated values.
    Second, build the dict. from slices of the array.
    Interp types are any allowed in interp1d
    """
    targ_mags = np.around(np.linspace(targ_mags_def[0], targ_mags_def[1], ((targ_mags_def[1]-targ_mags_def[0])/targ_mags_def[2])+1), decimals=1)
    f_tmp = np.zeros((f_mod.shape[0],len(targ_mags),f_mod.shape[2],f_mod.shape[3]))

    # interp mags
    for it in range(f_mod.shape[2]): # template
        for jf in range(f_mod.shape[3]): # filter
            for kz in range(f_mod.shape[0]): # redshift
                f_tmp[kz,:,it,jf] = interp1d(spine_mags, f_mod[kz,:,it,jf], fill_value="extrapolate", kind=method)(targ_mags)

    # build dict
    AB_dict = {}
    for i, mag in enumerate(targ_mags):
        AB_dict[mag] = f_tmp[:,i,:,:]

    del f_tmp

    return AB_dict




def interp_AB(f_mod, m_0, m_arr, method="linear"):
    """ returns the correct model fluxes, given the object magnitude.
    Interp types are any allowed in interp1d (or could use linear from numpy).
    Need to change this so that we set-up a look up table earlier, and pick closest - like in Ben's code.
    Or maybe just prep. the interp functions, and re-use - would need LOTS though."""
    # set-up output array
    f_out = np.zeros((f_mod.shape[0],f_mod.shape[2],f_mod.shape[3]))

    # for each combination of filter and template (i.e. .AB file),
    # we interpolate the fluxes at each z.
    for it in range(f_mod.shape[2]): # template
        for jf in range(f_mod.shape[3]): # filter
            for kz in range(f_mod.shape[0]): # redshift
                f_out[kz,it,jf] = interp1d(m_arr, f_mod[kz,:,it,jf], fill_value="extrapolate")(m_0)
                #f_out[kz,it,jf] = np.interp(m_0, m_arr, f_mod[kz,:,it,jf])
    return f_out


"""
n_mag
f_mod[nz, nmag, ntempl, nfilt]
m_0
"""
