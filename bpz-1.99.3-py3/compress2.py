from numpy import *
#from numpy import compress as compress1b

# THE numpy1.0b compress SUCKS!
# I REIMPORT IT AS compress1b AND I USE IT HERE
def compress2(c, m):
    #print 'compress2: NEW VERSION OF compress BECAUSE THE Numpy 1.0b VERSION IS SHWAG'
    m = array(m)
    if len(m.shape) == 1:
        mc = compress(c, m)
    elif len(m.shape) == 2:
        nm, nc = m.shape
        if type(c) != list:
            c = c.tolist()
        c = c * nm
        m = ravel(m)  # REQUIRED ON SOME MACHINES
        mc = compress(c, m)
        mc = reshape(mc, (nm, len(mc)/nm))
    else:
        print('MORE THAN 2 AXES NOT SUPPORTED BY compress2')
    return mc
