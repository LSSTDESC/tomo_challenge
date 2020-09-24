## Automatically adapted for numpy Jun 08, 2006 by convertcode.py
## ALSO CHECKED MANUALLY: from numpy import *
## CHANGED MANUALLY: inf -> Inf; nan -> NaN

"""Matlab(tm) compatibility functions.

This will hopefully become a complete set of the basic functions available in
matlab.  The syntax is kept as close to the matlab syntax as possible.  One 
fundamental change is that the first index in matlab varies the fastest (as in 
FORTRAN).  That means that it will usually perform reductions over columns, 
whereas with this object the most natural reductions are over rows.  It's perfectly
possible to make this work the way it does in matlab if that's desired.
"""
# I CHANGED median -- DC
# I ADDED thetastd -- DC
# I ADDED histogram -- DC
# avgstd2, std2, sum, total, size, divisible, ndec, interp, bilin
# HAD TO REMOVE RandomArray BECAUSE OF AN ERROR: 
#   ImportError: ld.so.1: python: fatal: /home/coe/python/ranlib.so: wrong ELF data format: ELFDATA2LS

#from Numeric import *
from numpy import *
from compress2 import compress2 as compress
from bisect import bisect
from scipy.integrate import quad
from scipy.special import erf
from numpy.random import *  # random
#from biggles import *
import string

try:
    from roman import roman  # Roman numerals
except:
    pass

def argmin2d(a):
    i = argmin(a.flat)
    ny, nx = a.shape
    iy = i / nx
    ix = i % nx
    return iy, ix

def argmax2d(a):
    i = argmax(a.flat)
    ny, nx = a.shape
    iy = i / nx
    ix = i % nx
    return iy, ix

def matrix_multiply(MM):
    """Multiplies a list of matrices: M[0] * M[1] * M[2]..."""
    P = MM[0]
    for M in MM[1:]:
        P = dot(P, M)
    return P

def sinn(x):
    """
    x < 0: sin
    x > 0: sinh
    """
    if x < 0:
        return sin(x)
    else:
        return sinh(x)

def multiples(lo, hi, x=1, eps=1e-7):
    """Returns an array of the multiples of x between [lo,hi] inclusive"""
    l = ceil((lo-eps)/x)*x
    return arange(l, hi+eps, x)

def multiples2(lohi, x=1, eps=1e-7):
    """Returns an array of the multiples of x between [lo,hi] inclusive"""
    lo, hi = lohi
    return multiples(lo, hi, x, eps)

def multipleslog(lo, hi):
    """Returns an array of the log multiples between [lo,hi] inclusive.
    That didn't make sense, but what I'm trying to say is:
    multipleslog(2, 30) = 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30"""
    loglo = log10(lo)
    loghi = log10(hi)
    ll = multiples(loglo, loghi)
    ll = concatenate([[loglo], ll, [loghi]])
    mm = []
    for i in range(len(ll)-1):
        lo = 10 ** ll[i]
        hi = 10 ** ll[i+1]
        ex = 10 ** floor(ll[i])
        m1 = multiples(lo, hi, ex)
        if len(mm):
            if close(m1[0], mm[-1]):
                m1 = m1[1:]
        mm = concatenate([mm, m1])
    return mm

def multiples2log(lohi):
    """Returns an array of the log multiples between [lo,hi] inclusive.
    That didn't make sense, but what I'm trying to say is:
    multipleslog(2, 30) = 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30"""
    lo, hi = lohi
    return multipleslog(lo, hi)


def onlyids(data, ids):
    """ALTERS ARRAY data TO INCLUDE ONLY NUMBERS IN ids
    ALL OTHER VALUES SET TO zero"""
    keys = arange(data.size)

    keysc = compress(data.flat, keys)
    valsc = compress(data.flat, data.flat)

    mask = zeros(data.size)
    for id in ids:
        ee = equal(valsc, id)
        mask = logical_or(mask, ee)

    keyscm = compress(mask, keysc)
    valscm = compress(mask, valsc)

    datanew = zeros(data.shape)
    datanew.put(keyscm, valscm)

    return datanew

def cliplohi(xlo, xhi, xmin, xmax):
    return max([xlo, xmin]), min([xhi, xmax])

def base(b, nums):
    """base(10, [1, 2, 3]) RETURNS 123"""
    if not isinstance(nums, list):
        nums = nums.tolist()
    nums.reverse()
    x = 0
    for i, num in enumerate(nums):
        x += array(num) * b**i
    return x

def strbegin(str, phr):  # coetools.py
    return str[:len(phr)] == phr

def minsec(x, format=(), precision=None):
    """
    CONVERTS decimal degrees/hours to degrees/hours : minutes : seconds
    minsec(13.52340987)
    minsec(13.52340987, ':')
    minsec(13.52340987, 'hms')
    minsec(13.52340987, 'dms')
    minsec(13.52340987, 'dms', 1)
    """
    f, i = math.modf(x)
    i = int(i)
    m = 60 * f
    s, m = math.modf(m)
    m = int(m)
    s = 60 * s
    if type(format) == str:
        if precision == None:
            s = '%f' % s
        else:
            fmt = '%%.%df' % precision
            s = fmt % s
        if strbegin(s, '60'):  # rounded up
            s = '0'
            m = m + 1
        m = '%d' % m
        if m == '60':  # rounded up
            m = '0'
            i += 1
        i = '%d' % i
        ims = (i,m,s)
        if len(format) == 1:
            out = string.join(ims, format)
        elif len(format) == 3:
            out = i+format[0] + m+format[1] + s+format[2]
    else:
        out = (i, m, s)
    return out

def sec2hms(x, precision=0, mpersist=True):
    """
    CONVERTS decimal seconds to hours : minutes : seconds
    """
    out = ''
    if x > 60:
        if x > 3600:
            h = int(x / 3600)
            out = '%d:' % h
            x = x - 3600 * h
        m = int(x / 60)
        out += '%d:' % m
        x = x - 60 * m
    elif mpersist:
        out = '0:'
    if precision == None:
        fmt = '%g'
    elif precision == 0:
        fmt = '%d'
    else:
        fmt = '%%.%df' % precision
    s = fmt % x
    if (x < 10) and mpersist:
        s = '0' + s
    out += s
    return out

def sec2yr(x, precision=0, mpersist=True):
    """
    CONVERTS decimal seconds to years, months, days, hours : minutes : seconds
    """
    out = ''
    minsec = 60  # minute
    if x > minsec:  # minutes
        hoursec = minsec * 60  # hour
        if x > hoursec:  # hours
            daysec = 24 * hoursec  # day
            if x > daysec:  # days
                yearsec = 365.25 * daysec
                monthsec = yearsec / 12.
                if x > monthsec:  # months
                    if x > yearsec:  # years
                        y = int(x / yearsec)
                        out = '%d years, ' % y
                        x = x - y * yearsec
                    months = int(x / monthsec)
                    out += '%d months, ' % months
                    x = x - months * monthsec
                d = int(x / daysec)
                out += '%d days, ' % d
                x = x - d * daysec
            h = int(x / 3600)
            out += '%d hours, ' % h
            x = x - 3600 * h
        m = int(x / 60)
        out += '%d minutes, ' % m
        x = x - 60 * m
    elif mpersist:
        out = '0 minutes, '
    if precision == None:
        fmt = '%g'
    elif precision == 0:
        fmt = '%d'
    else:
        fmt = '%%.%df' % precision
    s = fmt % x
    if (x < 10) and mpersist:
        s = '0' + s
    out += s
    out += ' seconds'
    return out

sec2yr(33333333333)

minsec = 60  # minute
hoursec = minsec * 60  # hour
daysec = 24 * hoursec  # day
yearsec = 365.25 * daysec
monthsec = yearsec / 12.

def prange(x, xinclude=None, margin=0.05):
    """RETURNS GOOD RANGE FOR DATA x TO BE PLOTTED IN.
    xinclude = VALUE YOU WANT TO BE INCLUDED IN RANGE.
    margin = FRACTIONAL MARGIN ON EITHER SIDE OF DATA."""
    xmin = min(x)
    xmax = max(x)
    if xinclude != None:
        xmin = min([xmin, xinclude])
        xmax = max([xmax, xinclude])

    dx = xmax - xmin
    if dx:
        xmin = xmin - dx * margin
        xmax = xmax + dx * margin
    else:
        xmin = xmin - margin
        xmax = xmax + margin
    return [xmin, xmax]

def minmax(x, range=None):
    if range:
        lo, hi = range
        good = between(lo, x, hi)
        x = compress(good, x)
    return min(x), max(x)

def rescale(x, lohi):
    lo, hi = lohi
    xlo, xhi = minmax(x)
    dx = xhi - xlo
    dy = hi - lo
    y = x / dx * dy + lo
    return y

def inrange(x, r):
    lo, hi = minmax(r)
    return between(lo, x, hi)

def pairs(x):
    p = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            p.append((x[i], x[j]))
    return p

def Psig(P, nsigma=1):
    """(ir, il) bound central nsigma of P
    -- edges contain equal amounts of P"""
    Pn = P / total(P)
    g = gausst(nsigma)
    Pl = cumsum(Pn)
    Pr = cumsum(Pn[::-1])
    n = len(P)
    i = arange(n)
    il = interp(g, Pl, i)
    ir = interp(g, Pr, i)
    ir = n - ir
    return il, ir

def xsig(x, P, nsigma=1):
    print('xsigmom MUCH MORE ACCURATE THAN xsig IN MLab_coe')
    return p2p(take(x, Psig(P, nsigma))) / 2.

def gaussin(nsigma=1):
    """FRACTION WITHIN nsigma"""
    return erf(nsigma / sqrt(2))

def gaussp(nsigma=1):
    """FRACTION INCLUDED UP TO nsigma"""
    return 0.5 + gaussin(nsigma) / 2.

def gaussbtw(nsig1, nsig2):
    """FRACTION BETWEEN nsig1, nsig2"""
    return abs(gaussp(nsig2) - gaussp(nsig1))

#gaussbtw(0, 3)

sigma = gaussin

def gausst(nsigma=1):
    """FRACTION IN TAIL TO ONE SIDE OF nsigma"""
    return 1 - gaussp(nsigma)

###
# ~/glens/h0limits/gravlens/mock/1/ptdrawbox24dsepL0.py

from scipy.optimize import golden

def mom2(x, y):
    return sqrt(total(x**2 * y) / total(y))

def mom2dx(dx, x, y):
    return mom2(x+dx, y)

def xsigmom(x, y):
    """1-sigma of y(x) calculated using moments"""
    dx = golden(mom2dx, (x, y))
    return mom2(x+dx, y)

def testxsigmom():
    x = mgrid[-5:5:100001j]
    g = gauss1(abs(x-0.98765), 0.123456789)
    print(xsig(x, g))
    print(xsigmom(x, g))

    x = mgrid[-5:5:101j]
    g = gauss1(abs(x-0.98765), 0.123456789)
    print(xsig(x, g))
    print(xsigmom(x, g))

###



def pick(x):
    n = len(x)
    i = random_integers(n)
    return x[i-1]

def randrange(N=1):
    return (2 * random(N) - 1)

def randrange2(lo, hi, N=1):
    return ((hi - lo) * random(N) + lo)

class PDraw:
    def __init__(self, x, P):
        self.x = x
        self.P = P
        self.Pcum = cumsum(P)
        self.N = self.Pcum[-1]
    def draw(self, n=1):
        r = self.N * random(n)
        i = searchsorted(self.Pcum, r)
        return take(self.x, i)

def hypotsq(dx, dy):
    return dx**2 + dy**2

def hypotn(x):
    return sqrt(sum(x**2))

def hypotnn(*x):
    return hypotn(array(x))

#hypotnn(3, 4, 5)

def hypotxy(x1, y1, x2, y2):
    return hypot(x1-x2, y1-y2)

def hypotinvn(x):
    return 1 / sqrt(sum(1./x**2))

def hypotinvnn(*x):
    return hypotinvn(array(x))

def hypotinv(x, y):
    return hypotinvnn(x, y)

def subtend(x1, y1, x2, y2):
    """ANGLE SUBTENDED BY TWO VECTORS (wrt THE ORIGIN)"""
    # v1 (dot) v2 = |v1| |v2| cos(theta)
    # d = r1 r2 cos(theta)
    d = dot([x1, y1], [x2, y2])
    r1 = hypot(x1, y1)
    r2 = hypot(x2, y2)
    costheta = d / (r1 * r2)
    theta = arccos(costheta)
    return theta

def subtends(x, y):
    n = len(x)
    dd = []
    for i in range(n-1):
        for j in range(i+1,n):
            dd.append(subtend(x[i], y[i], x[j], y[j]))
    return array(dd)

def distances(x, y):
    n = len(x)
    dd = []
    for i in range(n-1):
        for j in range(i+1,n):
            dd.append(hypot(x[i]-x[j], y[i]-y[j]))
    return array(dd)

def differences(x):
    n = len(x)
    dd = []
    for i in range(n-1):
        for j in range(i+1,n):
            dd.append(x[i]-x[j])
    return array(dd)

def nrange(x, n=100):
    """n EQUALLY-SPACED SAMPLES ON THE RANGE OF x"""
    return arange(n) / (n-1.) * (max(x) - min(x)) + min(x)

def range01(n=100):
    """n EQUALLY-SPACED SAMPLES ON THE RANGE OF [0,1]"""
    return arange(n) / (n-1.)

def middle(x):
    return (max(x) + min(x)) / 2.

def within(A, xc, yc, ro, yesorno=0):  # --DC
    """RETURNS WHETHER EACH PIXEL OF AN ARRAY IS WITHIN A CIRCLE
    DEFINED ON THE ARRAY'S COORDINATES.
    FRACTIONAL MEMBERSHIP IS ALSO ESTIMATED
    BY THE FRACTION OF THE BOX CROSSED BY THE CIRCLE AT THAT ANGLE.
    IT'S LIKE ANTI-ALIASING.
    THESE FRACTIONS ARE SLIGHTLY OVERESTIMATED
    BUT ARE AN IMPROVEMENT OVER NOT USING THEM AT ALL!
    TO TURN OFF FRACTIONS AND JUST RETURN True/False, SET yesorno=1"""
    ny, nx = A.shape
    a = ones((ny,nx))
    y = arange(ny)
    x = arange(nx)
    x, y = meshgrid(x, y)
    x = x-xc + 0.
    y = y-yc + 0.
    r = hypot(x,y)
    xy = abs(divsafe(x, y, nan=0))
    yx = abs(divsafe(y, x, nan=0))
    m = min([xy, yx])
    dr = hypot(1, m)  # = 1 ON AXES, sqrt(2) ON DIAGONALS

    if (ro - xc > 0.5) or (ro - yc > 0.5) \
            or (ro + xc > nx - 0.5) or (ro + yc > ny - 0.5):
        print('WARNING: CIRCLE EXTENDS BEYOND BOX IN MLab_coe.within')

    if yesorno:
        v = less_equal(r, ro)  # TRUE OR FALSE, WITHOUT FRACTIONS
    else:
        v = less_equal(r, ro-0.5*dr) * 1
        v = v + between(ro-0.5*dr, r, ro+0.5*dr) * (ro+0.5*dr - r) / dr

    #if showplot:  matplotlib NOT LOADED IN MLab_coe
    if 0:
        matshow(v)
        circle(xc+0.5, yc+0.5, ro, color='k', linewidth=2)

    return v

#def sumwithin(A, xc, yc, ro, showplot=0):
#    return total(A * within(A, xc, yc, ro, showplot=showplot))

def sumwithin(A, xc, yc, ro):
    """RETURNS SUM OF ARRAY WITHIN CIRCLE DEFINED ON ARRAY'S COORDINATES"""
    return total(A * within(A, xc, yc, ro))

def floatin(x, l, ndec=3):
    """IS x IN THE LIST l?
    WHO KNOWS WITH FLOATING POINTS!"""
    x = int(x * 10**ndec + 0.1)
    l = (array(l) * 10**ndec + 0.1).astype(int).tolist()
    return x in l

def floatindex(x, l, ndec=3):
    """IS x IN THE LIST l?
    WHO KNOWS WITH FLOATING POINTS!"""
    x = int(x * 10**ndec + 0.1)
    l = (array(l) * 10**ndec + 0.1).astype(int).tolist()
    return l.index(x)

def integral(f, x1, x2):
    return quad(f, x1, x2)[0]

def magnify(a, n):
    """MAGNIFIES A MATRIX BY n
    YIELDING, FOR EXAMPLE:
    >>> magnify(IndArr(3,3), 2)
    001122
    001122
    334455
    334455
    667788
    667788
    """
    ny, nx = a.shape
    a = repeat(a, n**2)
    a = reshape(a, (ny,nx,n,n))
    a = transpose(a, (0, 2, 1, 3))
    a = reshape(a, (n*ny, n*nx))
    return a

def demagnify(a, n, func='mean'):
    """DEMAGNIFIES A MATRIX BY n
    YIELDING, FOR EXAMPLE:
    >>> demagnify(magnify(IndArr(3,3), 2), 2)
    012
    345
    678
    """
    ny, nx = array(a.shape) / n
    a = a[:ny*8,:nx*8]  # Trim if not even multiples
    a = reshape(a, (ny, n, nx, n))
    a = transpose(a, (0, 2, 1, 3))
    a = reshape(a, (ny, nx, n*n))
    a = transpose(a, (2, 0, 1))
    exec('a = %s(a)' % func)
    return a

# Elementary Matrices

# zeros is from matrixmodule in C
# ones is from Numeric.py

import numpy.random as RandomArray
import math

def listo(x):
    if singlevalue(x):
        x = [x]
    return x

# ~/glens/h0limits/scatterrea.py
def insidepoly1(xp, yp, x, y):
    """DETERMINES WHETHER THE POINT (x, y)
    IS INSIDE THE CONVEX POLYGON DELIMITED BY (xp, yp)"""
    xp, yp = CCWsort(xp, yp)
    xp = xp.tolist()
    yp = yp.tolist()
    if xp[-1] != xp[0]:
        xp.append(xp[0])
        yp.append(yp[0])

    xo = mean(xp)
    yo = mean(yp)
    inpoly = 1
    xa = [xo, x]
    ya = [yo, y]
    for j in range(len(xp)-1):
        xb = xp[j:j+2]
        yb = yp[j:j+2]
        if linescross2(xa, ya, xb, yb):
            inpoly = 0
            break

    return inpoly

# ~/glens/h0limits/scatterrea.py
# ALSO SEE matplotlib.nxutils.pnpoly & points_inside_poly()
# http://matplotlib.sourceforge.net/faq/howto_faq.html
def insidepoly(xp, yp, xx, yy):
    """DETERMINES WHETHER THE POINTS (xx, yy)
    ARE INSIDE THE CONVEX POLYGON DELIMITED BY (xp, yp)"""
    xp, yp = CCWsort(xp, yp)
    xx = ravel(listo(xx))
    yy = ravel(listo(yy))
    inhull = []
    for i in range(len(xx)):
        if i and not (i % 10000):
            print('%d / %d' % (i, len(xx)))
        inhull1 = insidepoly1(xp, yp, xx[i], yy[i])
        inhull.append(inhull1)

    return array(inhull).astype(int)


# TESTED IN ~/glens/lenspoints/optdefl/sourceconstraints/testconvexhull.py
#  testinsidepoly() -- NEVER QUITE GOT THE TEST TO WORK HERE
def insidepolyshwag(xp, yp, xx, yy):
    """DETERMINES WHETHER THE POINTS (xx, yy)
    ARE INSIDE THE CONVEX POLYGON DELIMITED BY (xp, yp)"""
    xp, yp = CCWsort(xp, yp)  # NEEDED
    xp = xp.tolist()
    yp = yp.tolist()
    if xp[-1] != xp[0]:
        xp.append(xp[-1])  # SHOULD BE [0]
        yp.append(yp[-1])  # SHOULD BE [0]

    xo = mean(xp)
    yo = mean(yp)
    xx = ravel(listo(xx))
    yy = ravel(listo(yy))
    inhull = ones(len(xx)).astype(int)
    for i in range(len(xx)):
        if i and not (i % 10000):
            print('%d / %d' % (i, len(xx)))
        xa = [xo, xx[i]]
        ya = [yo, yy[i]]
        for j in range(len(xp)-2):
            xb = xp[j:j+2]
            yb = yp[j:j+2]
            if linescross2(xa, ya, xb, yb):
                inhull[i] = 0
                break

    return inhull

def testinsidepoly():
    #from numpy.random import random
    N = 40
    x = random(50) * N
    y = random(50) * N
    xh, yh = convexhull(x, y)
    zz = arange(N)
    xx, yy = meshgrid(zz, zz)
    xx = ravel(xx)
    yy = ravel(yy)
    inhull = insidepoly(xh, yh, xx, yy)
    figure(11)
    clf()
    plot(xh, yh)
    ioff()
    for i in range(len(XX)):
        color = ['r', 'g'][ininin[i]]
        p = plot([xx[i]], [yy[i]], 'o', mfc=color)

    show()

def p2p(x):  # DEFINED AS ptp IN MLab (BELOW)
    return max(x) - min(x)

def rotate(x, y, ang):
    """ROTATES (x, y) BY ang RADIANS CCW"""
    x2 = x * cos(ang) - y * sin(ang)
    y2 = y * cos(ang) + x * sin(ang)
    return x2, y2

def rotdeg(x, y, ang):
    """ROTATES (x, y) BY ang DEGREES CCW"""
    return rotate(x, y, ang/180.*pi)

def linefit(x1, y1, x2, y2):
    """y = mx + b FIT TO TWO POINTS"""
    if x2 == x1:
        m = Inf
        b = NaN
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
    return m, b

def linescross(xa, ya, xb, yb):
    """
    DO THE LINES CONNECTING A TO B CROSS?
    A: TWO POINTS: (xa[0], ya[0]), (xa[1], ya[1])
    B: TWO POINTS: (xb[0], yb[0]), (xb[1], yb[1])
    DRAW LINE FROM A0 TO B0 
    IF A1 & B1 ARE ON OPPOSITE SIDES OF THIS LINE, 
    AND THE SAME IS TRUE VICE VERSA,
    THEN THE LINES CROSS
    """
    if xa[0] == xb[0]:
        xb = list(xb)
        xb[0] = xb[0] + 1e-10

    if xa[1] == xb[1]:
        xb = list(xb)
        xb[1] = xb[1] + 1e-10

    m0, b0 = linefit(xa[0], ya[0], xb[0], yb[0])
    ya1 = m0 * xa[1] + b0
    yb1 = m0 * xb[1] + b0
    cross1 = (ya1 > ya[1]) != (yb1 > yb[1])

    m1, b1 = linefit(xa[1], ya[1], xb[1], yb[1])
    ya0 = m1 * xa[0] + b1
    yb0 = m1 * xb[0] + b1
    cross0 = (ya0 > ya[0]) != (yb0 > yb[0])

    return cross0 and cross1

def linescross2(xa, ya, xb, yb):
    """
    DO THE LINES A & B CROSS?
    DIFFERENT NOTATION:
    LINE A: (xa[0], ya[0]) -> (xa[1], ya[1])
    LINE B: (xb[0], yb[0]) -> (xb[1], yb[1])
    DRAW LINE A
    IF THE B POINTS ARE ON OPPOSITE SIDES OF THIS LINE, 
    AND THE SAME IS TRUE VICE VERSA,
    THEN THE LINES CROSS
    """
    if xa[0] == xa[1]:
        xa = list(xa)
        xa[1] = xa[1] + 1e-10

    if xb[0] == xb[1]:
        xb = list(xb)
        xb[1] = xb[1] + 1e-10

    ma, ba = linefit(xa[0], ya[0], xa[1], ya[1])
    yb0 = ma * xb[0] + ba
    yb1 = ma * xb[1] + ba
    crossb = (yb0 > yb[0]) != (yb1 > yb[1])

    mb, bb = linefit(xb[0], yb[0], xb[1], yb[1])
    ya0 = mb * xa[0] + bb
    ya1 = mb * xa[1] + bb
    crossa = (ya0 > ya[0]) != (ya1 > ya[1])

    return crossa and crossb

def linescross2test():
    # from numpy.random import random
    xa = random(2)
    ya = random(2)
    xb = random(2)
    yb = random(2)

    figure(1)
    clf()
    plot(xa, ya)
    plot(xb, yb)
    title('%s' % linescross2(xa, ya, xb, yb))
    show()

def linescrosstest():
    # from random import random
    xa = random(), random()
    ya = random(), random()
    xb = random(), random()
    yb = random(), random()

    figure(1)
    clf()
    atobplot(xa, ya, xb, yb, linetype='')
    title('%s' % linescross(xa, ya, xb, yb))
    show()

def outside(x, y, xo, yo):
    """GIVEN 3 POINTS a, b, c OF A POLYGON 
    WITH CENTER xo, yo
    DETERMINE WHETHER b IS OUTSIDE ac,
    THAT IS, WHETHER abc IS CONVEX"""
    # DOES o--b CROSS a--c ?
    #      A--B       A--B
    xa, xb, xc = x
    ya, yb, yc = y
    xA = (xo, xa)
    yA = (yo, ya)
    xB = (xb, xc)
    yB = (yb, yc)
    return linescross(xA, yA, xB, yB)

# TESTED IN ~/glens/lenspoints/optdefl/sourceconstraints/testconvexhull.py
def convexhull(x, y, rep=1, nprev=0):
    """RETURNS THE CONVEX HULL OF x, y
    THAT IS, THE EXTERIOR POINTS"""
    x = x.astype(float)
    y = y.astype(float)
    x, y = CCWsort(x, y)
    xo = mean(x)
    yo = mean(y)
    x = x.tolist()
    y = y.tolist()
    dmax = max([p2p(x), p2p(y)])
    ngood = 0
    while ngood < len(x)+1:
        dx = x[1] - xo
        dy = y[1] - yo
        dr = hypot(dx, dy)
        dx = dx * dmax / dr
        dy = dy * dmax / dr
        x1 = xo - dx
        y1 = yo - dy
        if not outside(x[:3], y[:3], x1, y1):
            del x[1]
            del y[1]
        else: # ROTATE THE COORD LISTS
            x.append(x.pop(0))
            y.append(y.pop(0))
            ngood += 1

    x = array(x)
    y = array(y)

    # REPEAT UNTIL CONVERGENCE
    if (nprev == 0) or (len(x) < nprev):
        x, y = convexhull(x, y, nprev=len(x))

    if rep:
        x = concatenate((x, [x[0]]))
        y = concatenate((y, [y[0]]))

    return x, y

def gauss(r, sig=1., normsum=1):
    """GAUSSIAN NORMALIZED SUCH THAT AREA=1"""
    r = clip(r/float(sig), 0, 10)
    G = exp(-0.5 * r**2)
    G = where(less(r, 10), G, 0)
    if normsum:
        G = G * 0.5 / (pi * sig**2)
    return G

def gauss1(r, sig=1.):
    """GAUSSIAN NORMALIZED SUCH THAT PEAK AMPLITUDE = 1"""
    return gauss(r, sig, 0)

def atanxy(x, y, degrees=0):
    """ANGLE CCW FROM x-axis"""
    theta = arctan(divsafe(y, x, inf=1e30, nan=0))
    theta = where(less(x, 0), theta + pi, theta)
    theta = where(logical_and(greater(x, 0), less(y, 0)), theta + 2*pi, theta)
    if degrees:
        theta = theta * 180. / pi
    return theta


def chebyshev(x,n):
    if n == 0:
        return x ** 0
    elif n == 1:
        return x
    elif n == 2:
        return 2 * x ** 2 - 1
    elif n == 3:
        return 4 * x ** 3 - 3 * x
    elif n == 4:
        return 8 * x ** 4 - 8 * x ** 2
    elif n == 5:
        return 16 * x ** 5 - 20 * x ** 3 + 5 * x
    elif n == 6:
        return 32 * x ** 6 - 48 * x ** 4 + 18 * x ** 2 - 1

def chebyshev2d(x,y,a):
    A = x * 0
    ncy, ncx = a.shape
    for iy in range(ncy):
        for ix in range(ncx):
            if a[iy][ix]:
                A = A + a[iy][ix] * chebyshev(x,ix) * chebyshev(y,iy)
    return A

def crossprod(a, b):
    """CROSS PRODUCT (PROBABLY DEFINED IN SOME BUILT-IN MODULE!)"""
    return a[0] * b[1] - a[1] * b[0]

def dotprod(a, b):
    """DOT PRODUCT (PROBABLY DEFINED IN SOME BUILT-IN MODULE!)"""
    return a[0] * b[0] + a[0] * b[0]

def triarea(x, y, dir=0):
    """RETURNS THE AREA OF A TRIANGLE GIVEN THE COORDINATES OF ITS VERTICES
    A = 0.5 * | u X v |
    where u & v are vectors pointing from one vertex to the other two
    and X is the cross-product
    The dir flag lets you retain the sign (can tell if triangle is flipped)"""
    ux = x[1] - x[0]
    vx = x[2] - x[0]
    uy = y[1] - y[0]
    vy = y[2] - y[0]
    A = 0.5 * (ux * vy - uy * vx)
    if not dir:
        A = abs(A)
    return A

def CCWsort(x, y):
    """FOR A CONVEX SET OF POINTS, 
    SORT THEM SUCH THAT THEY GO AROUND IN ORDER CCW FROM THE x-AXIS"""
    xc = mean(x)
    yc = mean(y)
    ang = atanxy(x-xc, y-yc)
    SI = array(argsort(ang))
    x2 = x.take(SI, 0)
    y2 = y.take(SI, 0)
    return x2, y2

def polyarea(x, y):
    """RETURNS THE AREA OF A CONVEX POLYGON 
    GIVEN ITS COORDINATES (IN ANY ORDER)"""
    A = 0.
    x, y = CCWsort(x, y)
    for i in range(1, len(x)-1):
        xtri = x.take((0, i, i+1), 0)
        ytri = y.take((0, i, i+1), 0)
        A += triarea(xtri, ytri)
    return A

def odd(n):
    """RETURNS WHETHER AN INTEGER IS ODD"""
    return n & 1

def even(n):
    """RETURNS WHETHER AN INTEGER IS EVEN"""
    return 1 - odd(n)

def fpart(x):
    """FRACTIONAL PART OF A REAL NUMBER"""
    if type(x) in [array, list]:
        if len(x) == 1:
            x = x[0]
    return math.modf(x)[0]

def sigrange(x, nsigma=1):
    lo = percentile(gausst(nsigma), x)
    hi = percentile(gaussp(nsigma), x)
    return lo, hi

def sqrtsafe(x):
    """sqrt(x) OR 0 IF x < 0"""
    x = clip2(x, 0, None)
    return sqrt(x)

def sgn(a):
    return where(a, where(greater(a, 0), 1, -1), 0)

def sym8(a):
    """OKAY, SO THIS ISN'T QUITE RADIAL SYMMETRY..."""
    x = a + flipud(a) + fliplr(a) + transpose(a) + rot90(transpose(a),2) + rot90(a,1) + rot90(a,2) + rot90(a,3)
    return x / 8.

#def divsafe(a, b, inf=1e30, nan=0.):
def divsafe(a, b, inf=Inf, nan=NaN):
    """a / b with a / 0 = inf and 0 / 0 = nan"""
    a = array(a).astype(float)
    b = array(b).astype(float)
    asgn = greater_equal(a, 0) * 2 - 1.
    bsgn = greater_equal(b, 0) * 2 - 1.
    xsgn = asgn * bsgn
    sgn = where(b, xsgn, asgn)
    sgn = where(a, xsgn, bsgn)
    babs = clip(abs(b), 1e-200, 1e9999)
    bb = bsgn * babs
    #return where(b, a / bb, where(a, Inf, NaN))
    return where(b, a / bb, where(a, sgn*inf, nan))

def expsafe(x):
    x = array(x)
    y = []
    for xx in x:
        if xx > 708:
            y.append(1e333) # inf
        elif xx < -740:
            y.append(0)
        else:
            y.append(exp(xx))
    if len(y) == 1:
        return y[0]
    else:
        return array(y)

def floorint(x):
    return(int(floor(x)))

def ceilint(x):
    return(int(ceil(x)))

def roundint(x):
    if singlevalue(x):
        return(int(round(x)))
    else:
        return asarray(x).round().astype(int)

intround = roundint

def singlevalue(x):
    """IS x A SINGLE VALUE?  (AS OPPOSED TO AN ARRAY OR LIST)"""
    return type(x) in [type(None), float, float32, float64, int, int0, int8, int16, int32, int64]  # THERE ARE MORE TYPECODES IN Numpy

def roundn(x, ndec=0):
    if singlevalue(x):
        fac = 10.**ndec
        return roundint(x * fac) / fac
    else:
        rr = []
        for xx in x:
            rr.append(roundn(xx, ndec))
        return array(rr)

def percentile(p, x):
    x = sort(x)
    i = p * (len(x) - 1.)
    return interp(i, arange(len(x)), x)

def percentile2(v, x):
    return searchsorted(sort(x), v) / float(len(x))

def logical(x):
    return where(x, 1, 0)

def element_or(*l):
    """l is a list/tuple of arrays
    USAGE: x = element_or(a, b, c)"""
    x = where(l[0], l[0], l[1])
    for i in range(2,len(l)):
        x = where(x, x, l[2])
    return x

def log2(x, loexp=''):
    if loexp != '':
        x = clip2(x, 2**loexp, None)
    return log10(x) / log10(2)

def log10clip(x, loexp, hiexp=None):
    if hiexp==None:
        return log10(clip2(x, 10.**loexp, None))
    else:
        return log10(clip2(x, 10.**loexp, 10.**hiexp))

def lnclip(x, loexp):
    return log(clip2(x, e**loexp, None))

def linreg(X, Y):
    # written by William Park
    # http://www.python.org/topics/scicomp/recipes_in_python.html
    """ Returns coefficients to the regression line 'y=ax+b' from x[] and y[]. 
    Basically, it solves Sxx a + Sx b = Sxy Sx a + N b = Sy 
    where Sxy = \sum_i x_i y_i, Sx = \sum_i x_i, and Sy = \sum_i y_i. 
    The solution is a = (Sxy N - Sy Sx)/det b = (Sxx Sy - Sx Sxy)/det 
    where det = Sxx N - Sx^2. 
    In addition, 
    Var|a| = s^2 |Sxx Sx|^-1 
    = s^2 | N -Sx| / det |b| |Sx N | |-Sx Sxx| s^2
    = {\sum_i (y_i - \hat{y_i})^2 \over N-2} 
    = {\sum_i (y_i - ax_i - b)^2 \over N-2} 
    = residual / (N-2) R^2 
    = 1 - {\sum_i (y_i - \hat{y_i})^2 \over \sum_i (y_i - \mean{y})^2} 
    = 1 - residual/meanerror 
    It also prints to &lt;stdout&gt; 
    few other data, N, a, b, R^2, s^2, 
    which are useful in assessing the confidence of estimation. """
    #from math import sqrt
    if len(X) != len(Y):
        raise ValueError('unequal length')
    N = len(X)
    if N == 2: # --DC
        a = (Y[1] - Y[0]) / (X[1] - X[0])
        b = Y[0] - a * X[0]
    else:
        Sx = Sy = Sxx = Syy = Sxy = 0.0
        for x, y in map(None, X, Y):
            Sx = Sx + x
            Sy = Sy + y
            Sxx = Sxx + x*x
            Syy = Syy + y*y
            Sxy = Sxy + x*y
        det = Sxx * N - Sx * Sx
        a, b = (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det
        meanerror = residual = 0.0
        for x, y in map(None, X, Y):
            meanerror = meanerror + (y - Sy/N)**2
            residual = residual + (y - a * x - b)**2
        RR = 1 - residual/meanerror
        ss = residual / (N-2)
        Var_a, Var_b = ss * N / det, ss * Sxx / det
    print("y=ax+b")
    print("N= %d" % N)
    if N == 2:
        print("a= ", a)
        print("b= ", b)
    else:
        print("a= %g \\pm t_{%d;\\alpha/2} %g" % (a, N-2, sqrt(Var_a)))
        print("b= %g \\pm t_{%d;\\alpha/2} %g" % (b, N-2, sqrt(Var_b)))
        print("R^2= %g" % RR)
        print("s^2= %g" % ss)
    return a, b


def linregrobust(x, y):
    n = len(x)
    a, b = linreg(x, y)
    dy = y - (a * x + b)
    #s = std2(dy)
    s = std(dy)
    good = less(abs(dy), 3*s)
    x, y = compress(good, (x, y))
    ng = len(x)
    if ng < n:
        print('REMOVED %d OUTLIER(S), RECALCULATING linreg' % (n - ng))
        a, b = linreg(x, y)
    return a, b


def close(x, y, rtol=1.e-5, atol=1.e-8):
    """JUST LIKE THE Numeric FUNCTION allclose, BUT FOR SINGLE VALUES.  (WILL IT BE QUICKER?)"""
    return abs(y - x) < (atol + rtol * abs(y))

def wherein(x, vals):
    """RETURNS 1 WHERE x IS IN vals"""
    try:
        good = zeros(len(x), int)
    except:
        good = 0
    for val in vals:
        good = logical_or(good, close(x, val))
    return good

def wherenotin(x, vals):
    """RETURNS 1 WHERE x ISN'T IN vals"""
    return logical_not(wherein(x, vals))

def count(a):
    """RETURNS A DICTIONARY WITH THE NUMBER OF TIMES EACH ID OCCURS"""
    bins = norep(a)
    h = histogram(a, bins)
    d = {}
    for i in range(len(h)):
        d[bins[i]] = h[i]
    return d

def rep(a):
    """RETURNS A DICTIONARY WITH THE NUMBER OF TIMES EACH ID IS REPEATED
    1 INDICATES THE VALUE APPEARED TWICE (WAS REPEATED ONCE)"""
    a = sort(a)
    d = a[1:] - a[:-1]
    c = compress(logical_not(d), a)
    if c.any():
        bins = norep(c)
        h = histogram(c, bins)
        d = {}
        for i in range(len(h)):
            d[bins[i]] = h[i]
        return d
    else:
        return {}

def norep(a):
    """RETURNS a w/o REPETITIONS, i.e. THE MEMBERS OF a"""
    a = sort(a)
    d = a[1:] - a[:-1]
    c = compress(d, a)
    x = concatenate((c, [a[-1]]))
    return x
##     l = []
##     for x in ravel(a):
##         if x not in l:
##             l.append(x)
##     return array(l)

def norepxy(x, y, tol=1e-8):
    """REMOVES REPEATS IN (x,y) LISTS -- WITHIN tol EQUALS MATCH"""
    if type(x) == type(array([])):
        x = x.tolist()
        y = y.tolist()
    else:  # DON'T MODIFY ORIGINAL INPUT LISTS
        x = x[:]
        y = y[:]
    i = 0
    while i < len(x)-1:
        j = i + 1
        while j < len(x):
            dist = hypot(x[i] - x[j], y[i] - y[j])
            if dist < tol:
                del x[j]
                del y[j]
            else:
                j += 1
        i += 1
    return x, y


def isseq(a):
    """TELLS YOU IF a IS SEQUENTIAL, LIKE [3, 4, 5, 6]"""
    return (alltrue(a == arange(len(a)) + a[0]))

def between(lo, x, hi):  # --DC
    # RETURNS 1 WHERE lo < x < hi
    # (can also use that syntax "lo < x < hi")
    if lo in [None, '']:
        try:
            good = ones(len(x)).astype(int)
        except:
            good = 1
    else:
        good = greater(x, lo)
    if hi not in [None, '']:
        good = good * less(x, hi)
    return good

def divisible(x, n): # --DC
    return (x / float(n) - x / n) < (0.2 / n)

def ndec(x, max=3):  # --DC
    """RETURNS # OF DECIMAL PLACES IN A NUMBER"""
    for n in range(max, 0, -1):
        if round(x, n) != round(x, n-1):
            return n
    return 0  # IF ALL ELSE FAILS...  THERE'S NO DECIMALS

def qkfmt(x, max=8):
    n = ndec(x, max=max)
    if n:
        fmt = '%%.%df' % n
    else:
        fmt = '%d'
    return fmt % x

def interp(x, xdata, ydata, silent=0, extrap=0):  # NEW VERSION!
    """DETERMINES y AS LINEAR INTERPOLATION OF 2 NEAREST ydata"""
    SI = argsort(xdata)
    xdata = xdata.take(SI, 0)
    ydata = ydata.take(SI, 0)
    ii = searchsorted(xdata, x)
    if singlevalue(ii):
        ii = array([ii])
    # 0 = before all
    # len(xdata) = after all
    n = len(xdata)
    if extrap:
        i2 = clip(ii,   1, n-1)
        i1 = i2 - 1
    else:
        i2 = clip(ii,   0, n-1)
        i1 = clip(ii-1, 0, n-1)

    x2 = take(xdata, i2)
    x1 = take(xdata, i1)
    y2 = take(ydata, i2)
    y1 = take(ydata, i1)
    # m = (y2 - y1) / (x2 - x1)
    m = divsafe(y2 - y1, x2 - x1, nan=0)
    b = y1 - m * x1
    y = m * x + b
    if len(y) == 0:
        y = y[0]
    return y

interpn = interp

def interp1(x, xdata, ydata, silent=0):  # --DC
    """DETERMINES y AS LINEAR INTERPOLATION OF 2 NEAREST ydata"""
    SI = argsort(xdata)
    # NEW numpy's take IS ACTING FUNNY
    # NO DEFAULT AXIS, MUST BE SET EXPLICITLY TO 0
    xdata = xdata.take(SI, 0).astype(float).tolist()
    ydata = ydata.take(SI, 0).astype(float).tolist()
    if x > xdata[-1]:
        if not silent:
            print(x, 'OUT OF RANGE in interp in MLab_coe.py')
        return ydata[-1]
    elif x < xdata[0]:
        if not silent:
            print(x, 'OUT OF RANGE in interp in MLab_coe.py')
        return ydata[0]
    else:
        # i = bisect(xdata, x)  # SAME UNLESS EQUAL
        i = searchsorted(xdata, x)
        if xdata[i] == x:
            return ydata[i]
        else:
            [xlo, xhi] = xdata[i-1:i+1]
            [ylo, yhi] = ydata[i-1:i+1]
            return ((x - xlo) * yhi + (xhi - x) * ylo) / (xhi - xlo)

def interpn1(x, xdata, ydata, silent=0):  # --DC
    """DETERMINES y AS LINEAR INTERPOLATION OF 2 NEAREST ydata
    interpn TAKES AN ARRAY AS INPUT"""
    yout = []
    for x1 in x:
        yout.append(interp(x1, xdata, ydata, silent=silent))
    return array(yout)

def interp2(x, xdata, ydata):  # --DC
    """LINEAR INTERPOLATION/EXTRAPOLATION GIVEN TWO DATA POINTS"""
    m = (ydata[1] - ydata[0]) / (xdata[1] - xdata[0])
    b = ydata[1] - m * xdata[1]
    y = m * x + b
    return y

def bilin(x, y, data, datax, datay):  # --DC
    """ x, y ARE COORDS OF INTEREST
    data IS 2x2 ARRAY CONTAINING NEARBY DATA
    datax, datay CONTAINS x & y COORDS OF NEARBY DATA"""
    lavg = ( (y - datay[0]) * data[1,0] + (datay[1] - y) * data[0,0] ) / (datay[1] - datay[0])
    ravg = ( (y - datay[0]) * data[1,1] + (datay[1] - y) * data[0,1] ) / (datay[1] - datay[0])
    return ( (x - datax[0]) * ravg + (datax[1] - x) * lavg ) / (datax[1] - datax[0])

def bilin2(x, y, data):  # --DC
    """ x, y ARE COORDS OF INTEREST, IN FRAME OF data - THE ENTIRE ARRAY"""
    # SHOULD BE CHECKS FOR IF x, y ARE AT EDGE OF data
    ny, nx = data.shape
    ix = int(x)
    iy = int(y)
    if ix == nx-1:
        x -= 1e-7
        ix -= 1
    if iy == ny-1:
        y -= 1e-7
        iy -= 1
    if not ((0 <= ix < nx-1) and (0 <= iy < ny-1)):
        val = 0
    else:
        stamp = data[iy:iy+2, ix:ix+2]
        datax = [ix, ix+1]
        datay = [iy, iy+1]
        # print x, y, stamp, datax, datay
        val = bilin(x, y, stamp, datax, datay)
    return val

def rand(*args):
    """rand(d1,...,dn) returns a matrix of the given dimensions
    which is initialized to random numbers from a uniform distribution
    in the range [0,1).
    """
    return RandomArray.random(args)

def eye(N, M=None, k=0, dtype=None):
    """eye(N, M=N, k=0, dtype=None) returns a N-by-M matrix where the 
    k-th diagonal is all ones, and everything else is zeros.
    """
    if M == None: M = N
    if type(M) == type('d'): 
        typecode = M
        M = N
    m = equal(subtract.outer(arange(N), arange(M)),-k)
    return asarray(m,dtype=typecode)

def tri(N, M=None, k=0, dtype=None):
    """tri(N, M=N, k=0, dtype=None) returns a N-by-M matrix where all
    the diagonals starting from lower left corner up to the k-th are all ones.
    """
    if M == None: M = N
    if type(M) == type('d'): 
        typecode = M
        M = N
    m = greater_equal(subtract.outer(arange(N), arange(M)),-k)
    return m.astype(typecode)

# Matrix manipulation

def diag(v, k=0):
    """diag(v,k=0) returns the k-th diagonal if v is a matrix or
    returns a matrix with v as the k-th diagonal if v is a vector.
    """
    v = asarray(v)
    s = v.shape
    if len(s)==1:
        n = s[0]+abs(k)
        if k > 0:
            v = concatenate((zeros(k, v.dtype.char),v))
        elif k < 0:
            v = concatenate((v,zeros(-k, v.dtype.char)))
        return eye(n, k=k)*v
    elif len(s)==2:
        v = add.reduce(eye(s[0], s[1], k=k)*v)
        if k > 0: return v[k:]
        elif k < 0: return v[:k]
        else: return v
    else:
        raise ValueError("Input must be 1- or 2-D.")


def fliplr(m):
    """fliplr(m) returns a 2-D matrix m with the rows preserved and
    columns flipped in the left/right direction.  Only works with 2-D
    arrays.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError("Input must be 2-D.")
    return m[:, ::-1]

def flipud(m):
    """flipud(m) returns a 2-D matrix with the columns preserved and
    rows flipped in the up/down direction.  Only works with 2-D arrays.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError("Input must be 2-D.")
    return m[::-1]

# reshape(x, m, n) is not used, instead use reshape(x, (m, n))

def rot90(m, k=1):
    """rot90(m,k=1) returns the matrix found by rotating m by k*90 degrees
    in the counterclockwise direction.
    """
    m = asarray(m)
    if len(m.shape) != 2:
        raise ValueError("Input must be 2-D.")
    k = k % 4
    if k == 0: return m
    elif k == 1: return transpose(fliplr(m))
    elif k == 2: return fliplr(flipud(m))
    elif k == 3: return fliplr(transpose(m))

def rot180(m):
    return rot90(m, 2)

def rot270(m):
    return rot90(m, 3)

def tril(m, k=0):
    """tril(m,k=0) returns the elements on and below the k-th diagonal of
    m.  k=0 is the main diagonal, k > 0 is above and k < 0 is below the main
    diagonal.
    """
    m = asarray(m)
    return tri(m.shape[0], m.shape[1], k=k, dtype=m.dtype.char)*m

def triu(m, k=0):
    """triu(m,k=0) returns the elements on and above the k-th diagonal of
    m.  k=0 is the main diagonal, k > 0 is above and k < 0 is below the main
    diagonal.
    """     
    m = asarray(m)
    return (1-tri(m.shape[0], m.shape[1], k-1, m.dtype.char))*m 

# Data analysis

# Basic operations
def max(m):
    """max(m) returns the maximum along the first dimension of m.
    """
    return maximum.reduce(m)

def min(m):
    """min(m) returns the minimum along the first dimension of m.
    """
    return minimum.reduce(m)

# Actually from BASIS, but it fits in so naturally here...

def ptp(m):
    """ptp(m) returns the maximum - minimum along the first dimension of m.
    """
    return max(m)-min(m)

def mean1(m):
    """mean(m) returns the mean along the first dimension of m.  Note:  if m is
    an integer array, integer division will occur.
    """
    return add.reduce(m)/len(m)

def mean(m, axis=0):
    """mean(m) returns the mean along the first dimension of m.  Note:  if m is
    an integer array, integer division will occur.
    """
    m = asarray(m)
    return add.reduce(m, axis=axis) / m.shape[axis]

def meangeom(m):
    return product(m) ** (1. / len(m))

# sort is done in C but is done row-wise rather than column-wise
def msort(m):
    """msort(m) returns a sort along the first dimension of m as in MATLAB.
    """
    return transpose(sort(transpose(m)))

def median(m):
    """median(m) returns the median of m along the first dimension of m.
    """
    m = asarray(m)
    if m.shape[0] & 1:
        return msort(m)[m.shape[0]/2]  # ODD # OF ELEMENTS
    else:
        return (msort(m)[m.shape[0]/2] + msort(m)[m.shape[0]/2-1]) / 2.0  # EVEN # OF ELEMENTS


def rms(m):
    """Root-Mean-Squared, as advertised.
    std (below) first subtracts by the mean
    and later divides by N-1 instead of N"""
    return sqrt(mean(m**2))

def std(m):
    """std(m) returns the standard deviation along the first
    dimension of m.  The result is unbiased meaning division by len(m)-1.
    """
    mu = mean(m)
    return sqrt(add.reduce(pow(m-mu,2)))/sqrt(len(m)-1.0)

stddev = std

def meanstd(m):
    """meanstd(m) returns the mean and uncertainty = std / sqrt(N-1)
    """
    mu = mean(m)
    dmu = sqrt(add.reduce(pow(m-mu,2)))/(len(m)-1.0)
    return mu, dmu

def avgstd2(m): # --DC
    """avgstd2(m) returns the average & standard deviation along the first dimension of m.
    avgstd2 ELIMINATES OUTLIERS
    The result is unbiased meaning division by len(m)-1.
    """
    done = ''
    while not done:
        n = len(m)
        mu = mean(m)
        sig = sqrt(add.reduce(pow(m-mu,2)))/sqrt(n-1.0)
        good = greater(m, mu-3*sig) * less(m, mu+3*sig)
        m = compress(good, m)
        done = sum(good) == n

    return [mu, sqrt(add.reduce(pow(m-mu,2)))/sqrt(len(m)-1.0)]

def std2(m): # --DC
    """std2(m) returns the standard deviation along the first dimension of m.
    std2 ELIMINATES OUTLIERS
    The result is unbiased meaning division by len(m)-1.
    """
    [a, s] = avgstd2(m)
    return s

stddev = std

def weightedavg(x, w):
    return sum(x * w) / sum(w)

weightedmean = weightedavg

## def thetaavgstd1(theta):
##     """SHWAG VERSION: WON'T WORK IF THETA SPANS A RANGE > pi
##     CALCULATES THE AVERAGE & STANDARD DEVIATION IN A LIST (OR 1-D ARRAY) OF THETA (ANGLE) MEASUREMENTS
##     RETURNS THE LIST [avg, std]    
##     NEED A NEW CODE TO HANDLE THAT: ?INCREASING WEIGHTED AVERAGES (2 POINTS AT A TIME)?"""
##     if len(theta) == 1:
##      return([theta[0], 999])
##     else:
##      # PUT ALL theta IN [0, 2 * pi]
##      for i in range(len(theta)):
##          if theta[i] < 0:
##              theta[i] = theta[i] + 2 * pi
##      if max(theta) - min(theta) > pi:
##          # "PUT ALL THETA IN [-pi, pi]"
##          for i in range(len(theta)):
##              if theta[i] > pi:
##                  theta[i] = theta[i] - 2 * pi
##      #print theta
##      if max(theta) - min(theta) > pi:
##          print "THETA RANGE TOO BIG FOR thetaavg"
##          return([999, 999])
##         else:
##          thavg = mean(theta)
##          thstd = sqrt( sum( (theta - thavg) ** 2 ) / (len(theta) - 1.) )
##          return([thavg, thstd])

def thetaavgstd(theta):
    """CALCULATES THE AVERAGE & STANDARD DEVIATION IN A LIST (OR 1-D ARRAY) OF THETA (ANGLE) MEASUREMENTS
    RETURNS THE LIST [avg, std]
    CAN HANDLE ANY RANGE OF theta
    USES INCREASING WEIGHTED AVERAGES (2 POINTS AT A TIME)"""
    n = len(theta)
    if n == 1:
        return([theta[0], 999])
    else:
        thavg = theta[0]
        for i in range(1,n):
            th = theta[i]
            if thavg - th > pi:
                thavg = thavg - 2 * pi
            elif th - thavg > pi:
                th = th - 2 * pi
            thavg = ( i * thavg + th ) / (i+1)
        for i in range(n):
            if theta[i] > thavg + pi:
                theta[i] = theta[i] - 2 * pi
        thstd = std(theta)
        return([thavg, thstd])



def clip2(m, m_min=None, m_max=None):
    if m_min == None:
        m_min = min(m)
    if m_max == None:
        m_max = max(m)
    return clip(m, m_min, m_max)


## def sum(m):
##      """sum(m) returns the sum of the elements along the first
##      dimension of m.
##      """
##      return add.reduce(m)
sum = add.reduce  # ALLOWS FOR AXIS TO BE INPUT --DC

def total(m):
    """RETURNS THE TOTAL OF THE ENTIRE ARRAY --DC"""
##     t = m
##     while not(type(t) in [type(1), type(1.)]):
##      t = sum(t)
##     return t
    return sum(ravel(m))

def size(m):
    """RETURNS THE TOTAL SIZE OF THE ARRAY --DC"""
    s = m.shape
    x = 1
    for n in s:
        x = x * n
    return x

def cumsum(m, axis=0):
    """cumsum(m) returns the cumulative sum of the elements along the
    first dimension of m.
    """
    return add.accumulate(m, axis=axis)

def prod(m):
    """prod(m) returns the product of the elements along the first
    dimension of m.
    """
    return multiply.reduce(m)

def cumprod(m):
    """cumprod(m) returns the cumulative product of the elments along the
    first dimension of m.
    """
    return multiply.accumulate(m)

def trapz(y, x=None):
    """trapz(y,x=None) integrates y = f(x) using the trapezoidal rule.
    """
    if x == None: d = 1
    else: d = diff(x)
    return sum(d * (y[1:]+y[0:-1])/2.0)

def cumtrapz(y, x=None, axis=0):
    """trapz(y,x=None) integrates y = f(x) using the trapezoidal rule. --DC"""
    if x == None: d = 1
    else: d = diff(x)
    if axis == 0:
        return cumsum(d * (y[1:]+y[0:-1])/2.0)
    elif axis == 1:
        return cumsum(d * (y[:,1:]+y[:,0:-1])/2.0, axis=1)
    else:
        print('YOUR VALUE OF axis = %d IS NO GOOD IN MLab_coe.cumtrapz' % axis)

def xbins(x):
    """[-0.5, 0.5, 1] --> [-1, 0, 0.75, 1.25]"""
    d = shorten(x)
    da = x[1] - x[0]
    db = x[-1] - x[-2]
    d = concatenate(([x[0] - da/2.], d, [x[-1] + db/2.]))
    return d

def diff(x, n=1):
    """diff(x,n=1) calculates the first-order, discrete difference
    approximation to the derivative.
    """
    if n > 1:
        return diff(x[1:]-x[:-1], n-1)
    else:
        return x[1:]-x[:-1]

def shorten(x, n=1):  # shrink
    """shorten(x,n=1) 
    SHORTENS x, TAKING AVG OF NEIGHBORS, RECURSIVELY IF n > 1
    """
    a = (x[1:] + x[:-1]) / 2.
    if n > 1:
        return shorten(a, n-1)
    else:
        return a

def lengthen(x, n):  # expand
    """lengthen([0, 1, 5], 4) ==> 0, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5"""
    x = array(x)
    d = diff(x)
    i = arange(n) / float(n)
    o = outer(i, d)
    o = o + x[:-1]
    o = ravel(transpose(o))
    o = concatenate((o, [x[-1]]))
    return o

def powerlaw(x, y):
    """RETURNS EXPONENT n TO POWER LAW FIT y ~ x^n
    AT POINTS ON AVERAGED x"""
    # d log y / d log x = (dy / dx) * (x / y)
    # d log x = dx / x
    logx = log10(x)
    logy = log10(y)

    dlogx = diff(logx)
    dlogy = diff(logy)

    dd = dlogy / dlogx
    #x2 = (x[1:] + x[:-1]) / 2
    logx2 = shorten(logx)
    x2 = 10**logx2

    return x2, dd


def grad(m):
    """Calculates the gradient of the matrix m using the finite difference method
    The result will be 2 arrays, one for each of the axes x & y, respectively,
    with each having dimension (N-2, N-2), where m was (N, N).
    The coordinates will be in between of those of m.  --DC"""
    ay = (m[2:]   - m[:-2]) / 2.       # (N-2, N)
    ax = (m[:,2:] - m[:,:-2]) / 2.     # (N,   N-2)
    ay = ay[:,1:-1]                    # (N-2, N-2)
    ax = ax[1:-1,:]
    return array([ax, ay])

def laplacian(m):
    """Calculates the laplacian of the matrix m
    using the finite differencing method.
    The result will have dimension (ny-2, nx-2) where m had (ny, nx).
    see Fig. 2 of Bradac & Schneider 2005
    (Strong & Weak Lensing United I)
    although theirs is a factor of 1/2 too low.
    """
    ny, nx = m.shape
    center = m[1:-1,1:-1]

    sides = zeros(center.shape, float)
    for dx,dy in [(-1,0), (0,1), (1,0), (0,-1)]:
        sides = sides + m[1+dy:ny-1+dy, 1+dx:nx-1+dx]

    corners = zeros(center.shape, float)
    for dx,dy in [(-1,-1), (-1,1), (1,1), (1,-1)]:
        corners = corners + m[1+dy:ny-1+dy, 1+dx:nx-1+dx]

    return (2*corners - sides - 4*center) / 3.

def corrcoef(x, y=None):
    """The correlation coefficients
    """
    c = cov(x, y)
    d = diag(c)
    return c/sqrt(multiply.outer(d,d))

def cov(m,y=None):
    m = asarray(m)
    mu = mean(m)
    if y != None: m = concatenate((m,y))
    sum_cov = 0.0
    for v in m:
        sum_cov = sum_cov+multiply.outer(v,v)
    return (sum_cov-len(m)*multiply.outer(mu,mu))/(len(m)-1.0)

# Added functions supplied by Travis Oliphant
#import numpy.linalg.old as LinearAlgebra
def squeeze(a):
    "squeeze(a) removes any ones from the shape of a"
    b = asarray(a.shape)
    reshape (a, tuple (compress (not_equal (b, 1), b)))
    return

def kaiser(M,beta):
    """kaiser(M, beta) returns a Kaiser window of length M with shape parameter
    beta. It depends on the cephes module for the modified bessel function i0.
    """
    import cephes
    n = arange(0,M)
    alpha = (M-1)/2.0
    return cephes.i0(beta * sqrt(1-((n-alpha)/alpha)**2.0))/cephes.i0(beta)

def blackman(M):
    """blackman(M) returns the M-point Blackman window.
    """
    n = arange(0,M)
    return 0.42-0.5*cos(2.0*pi*n/M) + 0.08*cos(4.0*pi*n/M)


def bartlett(M):
    """bartlett(M) returns the M-point Bartlett window.
    """
    n = arange(0,M)
    return where(less_equal(n,M/2.0),2.0*n/M,2.0-2.0*n/M)

def hanning(M):
    """hanning(M) returns the M-point Hanning window.
    """
    n = arange(0,M)
    return 0.5-0.5*cos(2.0*pi*n/M)

def hamming(M):
    """hamming(M) returns the M-point Hamming window.
    """
    n = arange(0,M)
    return 0.54-0.46*cos(2.0*pi*n/M)

def sinc(x):
    """sinc(x) returns sin(pi*x)/(pi*x) at all points of array x.
    """
    return where(equal(x,0.0),1.0,sin(pi*x)/(pi*x))

from numpy.linalg import eig, svd
#def eig(v):
#    """[x,v] = eig(m) returns the the eigenvalues of m in x and the corresponding
#    eigenvectors in the rows of v.
#    """
#    return LinearAlgebra.eigenvectors(v)

#def svd(v):
#    """[u,x,v] = svd(m) return the singular value decomposition of m.
#    """
#    return LinearAlgebra.singular_value_decomposition(v)


def histogram(a, bins):
    n = searchsorted(sort(a), bins)
    n = concatenate([n, [len(a)]])
    return n[1:]-n[:-1]

def cumhisto(a,da=1.,amin=[],amax=[]): # --DC
    """
    Histogram of 'a' defined on the bin grid 'bins'
       Usage: h=histogram(p,xp)
    """
    if amin == []:
        amin = min(a)
    if amax == []:
        amax = max(a)
    nnn = (amax - amin) / da
    if less(nnn - int(nnn), 1e-4):
        amax = amax + da
    bins = arange(amin,amax+da,da)
    n=searchsorted(sort(a),bins)
    n=array(list(map(float,n)))
    return n[1:]

def cumHisto(a,da=1.,amin=[],amax=[]): # --DC
    if amin == []:
        amin = min(a)
    if amax == []:
        amax = max(a)
    h = cumhisto(a, da, amin, amax)
    return Histogram(h, amin, da)

def plotcumhisto(a,da=1.,amin=[],amax=[]): # --DC
    p = FramedPlot()
    p.add(cumHisto(a, da, amin, amax))
    p.show()
    return p

# from useful_coe.py
def histo(a,da=1.,amin=[],amax=[]): # --DC
    """
    Histogram of 'a' defined on the bin grid 'bins'
       Usage: h=histogram(p,xp)
    """
    if amin == []:
        amin = min(a)
    if amax == []:
        amax = max(a)
    nnn = (amax - amin) / da
    if less(nnn - int(nnn), 1e-4):
        amax = amax + da
    bins = arange(amin,amax+da,da)
    n=searchsorted(sort(a),bins)
#    n=concatenate([n,[len(a)]])
    n=array(list(map(float,n)))
##     print a
##     print bins
##     print n
    return n[1:]-n[:-1]
#    return hist(a, bins)

def Histo(a,da=1.,amin=[],amax=[], **other): # --DC
    if amin == []:
        amin = min(a)
    if amax == []:
        amax = max(a)
    try:
        amin = amin[0]
    except:
        pass
##     print 'hi'
##     print da
##     print amin
##     print amax
    h = histo(a, da, amin, amax)
##     print h
    return Histogram(h, amin, da, **other)

def plothisto(a,da=1.,amin=[],amax=[]): # --DC
    p = FramedPlot()
    p.add(Histo(a, da, amin, amax))
    p.show()

def bargraphbiggles(x, y, fill=1, color='black', **other):
    n = len(x)
    xx = repeat(x, 2)
    y = y.astype(float)
    z = array([0.])
    yy = concatenate([z, repeat(y, 2), z])
    zz = yy*0

    p = FramedPlot()
    if fill:
        p.add(FillBetween(xx, yy, xx, zz, color=color))
    else:
        p.add(Curve(xx, yy, color=color, **other))
    p.show()

def BarGraph(x, y, fill=1, color='black', bottom=0, **other):
    n = len(x)
    xx = repeat(x, 2)
    y = y.astype(float)
    z = array([0.])
    yy = concatenate([z, repeat(y, 2), z])
    zz = yy*0 + bottom
    if fill:
        return FillBetween(xx, yy, xx, zz, color=color)
    else:
        return Curve(xx, yy, color=color, **other)

def histob(a,da=1.,amin=[],amax=[]): # --DC
    # NOTE searchsorted can't be counted on to act consistently
    #   when bin values are equal to data values
    # for example, neither 0.04 or 0.05 gets put in the 0.04-0.05 bin
    #   0.04 gets put in the bin below, but 0.05 gets put in the bin above
    # So it's good to stagger your bins values when necessary (0.035, 0.045, 0.055)
    """
    Histogram of 'a' defined on the bin grid 'bins'
       Usage: h=histogram(p,xp)
    """
    if amin == []:
        amin = min(a)
    if amax == []:
        amax = max(a)
    # MAKE SURE 18 GOES IN THE 18-18.9999 bin (for da=1 anyway)
    amin = amin - 1e-4
    amax = amax + 1e-4
    #if less(abs(amax - a[-1]), da*1e-4):
    nnn = (amax - amin) / da
    if less(nnn - int(nnn), 1e-4):
        amax = amax + da
    #bins = arange(amin,amax+da,da)
    bins = arange(amin,amax+da,da)
    n=searchsorted(sort(a),bins)
    n=array(list(map(float,n)))
    n = n[1:]-n[:-1]
    return (bins, n)

def Histob(a, da=1., amin=[], amax=[], fill=1, color='black', bottom=0):
    bins, n = histob(a, da, amin, amax)
    return BarGraph(bins, n, fill=fill, color=color, bottom=bottom)

def histov(a, bins, v, presorted=0):
    """Total of values (v) in bins
    (other historgrams just count number of elements in bins)"""
    if not presorted:
        SI = argsort(a)
        a = take(a, SI)
        v = take(v, SI)
    vcum = cumsum(v)
    i = searchsorted(a, bins)
    i = i[1:] - 1
    vcumi = vcum.take(i)
    vcumi = concatenate([[0], vcumi])
    vb = vcumi[1:] - vcumi[:-1]
    return vb

#def isNaN(x):
#    return (x == 1) and (x == 0)

def isNaN(x):
    return not (x < 0) and not (x > 0) and (x != 0)

def isnan(x):
    l = less(x, 0)
    g = greater(x, 0)
    e = equal(x, 0)
    n = logical_and(logical_not(l), logical_not(g))
    n = logical_and(n, logical_not(e))
    return n

#from coeplot2a import *
#testinsidepoly()

