# HISTORY COMMENTS (A. Raichoor)
# ====================
#
# * code structure from prior_hdfn.py
# * prior values similar as LePhare new prior nzpriorNGVS.f
#    - Iab>20: same as nzprior2.f
#    - 17.<Iab<20: I "extrapolate" by hand the two ends
#             (for (zot,kt) and (ft,ktf), I impose continuity to zmax and rapp respectively)
#             (for alpt0, I take the mean values of the SDSS and nzprior2 samples)
#    - 13.<Iab<17: SDSS
#    - Iab<13.: square prior with p(z)=1 if z<0.1, p(z)=0 else
#

from bpz_tools import *

def function(z,m,nt):
    """NGVS prior for the main six types of Benitez 2000
    Returns an array pi[z[:],:6]
    The input magnitude is F814W AB
    """	   
    mmax=28.

    if nt<>6:
	print "Wrong number of template spectra!"
	sys.exit()

    global zt_at_a
    nz=len(z)
    
    if m>32.: m=32.
    if m<13:
    	p_i=zeros((nz,6),float)
    	idx=where(z<=0.1)
    	for i in (0,max(idx)):
    		p_i[i,:6]=1.
        norm=add.reduce(p_i[:nz,:6],0)
        p_i[:nz,:6]=p_i[:nz,:6]/norm[:6]*ones(6,float)/6.
    else:
        if 13<=m<17:
            momin=12.5
            a=array((2.69,2.19,2.19,1.99,1.99,1.99))
            zo=array((0.005,0.004,0.004,0.003,0.003,0.003))
            km=array((0.0256,0.0200,0.0200,0.018,0.018,0.018))
            fo_t=array((0.52,0.135,0.135))
            k_t=array((0.030,-0.048,-0.048))
        elif 17<=m<20:
            momin=17.
            a=array((2.58,2.00,2.00,2.00,2.00,2.00))
            zo=array((0.122,0.094,0.094,0.084,0.084,0.084))
            km=array((0.103,0.099,0.099,0.072,0.072,0.072))
            fo_t=array((0.45,0.17,0.17))
            k_t=array((0.138,-0.011,-0.011))
        else:
            momin=20.
            a=array((2.46,1.81,1.81,2.00,2.00,2.00))
            zo=array((0.431,0.39,0.39,0.3,0.3,0.3))
            km=array((0.091,0.10,0.10,0.15,0.15,0.15))
            fo_t=array((0.30,0.175,0.175))
            k_t=array((0.40,0.3,0.3))
        dm=m-momin
        zmt=zo+km*dm # zmt=clip(zo+km*dm,0.01,15.)
        zmt_at_a=zmt**(a)
        #We define z**a as global to keep it 
        #between function calls. That way it is 
        # estimated only once
        try:
            zt_at_a.shape
        except NameError:
            zt_at_a=power.outer(z,a)
            
        #Morphological fractions
        f_t = zeros((len(a),), 'f')
        f_t[:3]=fo_t*exp(-k_t*dm)
        f_t[3:]=(1.-add.reduce(f_t[:3]))/3.
        #Formula:
       	#zm=zo+km*(m_m_min)
       	#p(z|T,m)=(z**a)*exp(-(z/zm)**a)
        p_i=zt_at_a[:nz,:6]*exp(-clip(zt_at_a[:nz,:6]/zmt_at_a[:6],0.,700.))
        #This eliminates the very low level tails of the priors
        #norm=add.reduce(p_i[:nz,:6],0)
        #p_i[:nz,:6]=where(less(p_i[:nz,:6]/norm[:6],1e-2/float(nz)),
        #0.,p_i[:nz,:6]/norm[:6])
        norm=add.reduce(p_i[:nz,:6],0)
        p_i[:nz,:6]=p_i[:nz,:6]/norm[:6]*f_t[:6]
		
    return p_i
