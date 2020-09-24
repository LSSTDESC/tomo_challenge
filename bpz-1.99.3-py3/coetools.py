## Automatically adapted for numpy Jun 08, 2006 by 

## Automatically adapted for numpy Jun 08, 2006 by 

# PYTHON TOOLS
# Dan Coe

import os
import sys
#import shutil
#from glob import glob
#import fitsio
#import math
#from Numeric import *
from numpy import *
sys.float_output_precision = 5  # PRINTING ARRAYS: # OF DECIMALS
from types import *  # TO TELL WHAT TYPE A VARIABLE IS
from time import *
from MLab_coe import * #median, std, mean
#from bisect import bisect  # WHERE ITEM WOULD FIT IN SORTED LIST
#from RandomArray import *
#from biggles import *
#from useful import points, connect, mark_faroutliers
#from sortcat import sortcat
#from NumTut import *  # view ARRAY VIEWER
#from colormap import colormap, Colormap, addColormap
# plotconfig() CALLED BELOW
#import numarray
#import pyfits
from compress2 import compress2 as compress
#import popen2
import subprocess
import string  # LOAD THIS AFTER numpy, BECAUSE numpy HAS ITS OWN string

# ORIGINALLY ksbtools.py
# NOW coetools.py, BROKEN INTO coeio, smooth

# NOTE PLACEMENT OF THIS LINE IS IMPORTANT
# coeio ALSO IMPORTS FROM coetools (THIS MODULE)
# SO TO AVOID AN INFINITE LOOP, coeio ONLY LOADS FROM coetools
#  THOSE FUNCTIONS DEFINED BEFORE coeio IS LOADED
# SO, I'M LOADING THESE AFTER EVERYTHING DEFINED HERE!
#from coeio import *
#from smooth import *

numerix = os.environ.get('NUMERIX', '')

pwd = os.getcwd
die = sys.exit

def color1to255(color):
    return tuple((array(color) * 255. + 0.49).astype(int).tolist())  # CONVERT TO 0-255 SCALE

def color255to1(color):
    return tuple((array(color) / 255.).tolist())  # CONVERT TO 0-255 SCALE

def color2hex(color):
    if 0:  # 0 < max(color) <= 1:  # 0-1 SCALE
        # BUT EVERY ONCE IN A WHILE, YOU'LL GET A (0,0,1) OUT OF 255...
        color = color1to255(color)
    colorhex = '#'
    for val in color:
        h = hex(val)[2:]
        if len(h) == 1:
            h = '0'+h
        colorhex += h
    return colorhex

###

def keyvals(k, keys, vals):
    """GIVEN {keys: vals}, RETURNS VALUES FOR k
    THERE MUST BE A BUILT-IN WAY OF DOING THIS!"""
    d = dict(list(zip(keys, vals)))
    d[0] = 0
    f = lambda x: d[x]
    v = list(map(f, ravel(k)))
    if type(k) == type(array([])):
        v = array(v)
        v.shape = k.shape
    return v

def printmult(x, n):
    if not (x % n):
        print(x)

def cd(dir):
    if len(dir) > 2:
        if dir[0:2] == '~/':
            dir = os.path.join(home, dir[2:])
    os.chdir(dir)

def cdmk(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    os.chdir(dir)

def splitparagraphs(txt):
    paragraphs = ['']
    for line in txt:
        line = string.strip(line)
        if not line:
            line = '\n'
        if line[-1] != '\n':
            line += '\n'
        if line == '\n':
            paragraphs.append('')
        else:
            #paragraphs[-1].append(line)
            paragraphs[-1] += line
    if paragraphs[-1] == '':
        paragraphs = paragraphs[:-1]
    return paragraphs

def echo(word):
    cmd = 'echo ' + word
    subproc = subprocess.Popen(cmd)
    out = subproc.fromchild.readlines()  # SExtractor output
    out = out[0][:-1]  # LIST OF 1 STRING WITH \n AT END
    return out

home = os.environ.get('HOME', '')

def singlevalue(x):
    """IS x A SINGLE VALUE?  (AS OPPOSED TO AN ARRAY OR LIST)"""
    # return type(x) in [float, int]  THERE ARE MORE TYPECODES IN Numpy
    return type(x) in [float, float32, float64, int, int0, int8, int16, int32, int64]  # THERE ARE MORE TYPECODES IN Numpy
##     try:
##         a = x[0]
##         singleval = False
##     except:
##         singleval = True
##     return singleval

def comma(x, ndec=0):
    if ndec:
        format = '%%.%df' % ndec
        s = format % x
        si, sf = string.split(s, '.')
        sf = '.' + sf
    else:
        s = '%d' % x
        si = s
        sf = ''
    ss = ''
    while len(si) > 3:
        ss = ',' + si[-3:] + ss
        si = si[:-3]
    ss = si + ss + sf
    return ss

# print comma(9812345.67)

def th(n):
    """RETURNS 0th, 1st, 2nd, 3rd, 4th, 5th, etc."""
    if n == 1:
        return '1st'
    elif n == 2:
        return '2nd'
    elif n == 3:
        return '3rd'
    else:
        return '%dth' % n

nth = th

def num2str(x, max=3):
    try:
        n = ndec(x, max)
        if n:
            return "%%.%df" % n % x
        else:
            return "%d" % x
    except:
        return x

def str2num(str, rf=0):
    """CONVERTS A STRING TO A NUMBER (INT OR FLOAT) IF POSSIBLE
    ALSO RETURNS FORMAT IF rf=1"""
    try:
        num = string.atoi(str)
        format = 'd'
    except:
        try:
            num = string.atof(str)
            format = 'f'
        except:
            if not string.strip(str):
                num = None
                format = ''
            else:
                num = str
                format = 's'
    if rf:
        return (num, format)
    else:
        return num

def minmax(x, range=None):
    if range:
        lo, hi = range
        good = between(lo, x, hi)
        x = compress(good, x)
    return min(x), max(x)

#############################################################################
# ARRAYS
#
# PYTHON USES BACKWARDS NOTATION: a[row,column] OR a[iy,ix] OR a[iy][ix]
# NEED TO MAKE size GLOBAL (I THINK) OTHERWISE, YOU CAN'T CHANGE IT!
# COULD HAVE ALSO USED get_data IN ~txitxo/Python/useful.py

def FltArr(n0,n1):
    """MAKES A 2-D FLOAT ARRAY"""
    #a = ones([n0,n1], dtype=float32)
    #a = ones([n0,n1], float32)  # DATA READ IN LESS ACCURATELY IN loaddata !!
    # float32 can't handle more than 8 significant digits
    a = ones([n0,n1], float)
    return(a[:])


def IndArr(n0,n1):
    """MAKES A 2-D INTEGER ARRAY WITH INCREASING INDEX"""
    a = arange(n0*n1)
    return resize(a, [n0,n1])

#################################
# STRINGS, INPUT

def striskey(str):
    """IS str AN OPTION LIKE -C or -ker
    (IT'S NOT IF IT'S -2 or -.9)"""
    iskey = 0
    if str:
        if str[0] == '-':
            iskey = 1
            if len(str) > 1:
                iskey = str[1] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']
    return iskey


def pause(text=''):
    inp = input(text)

def wait(seconds):
    t0 = time()
    t1 = time()
    while (t1 - t0) < seconds:
        t1 = time()

def inputnum(question = ''):
    done = 0
    while not done:
        rinp = input(question)
        try: 
            x = string.atof(rinp)
            done = 1
        except: 
            pass
    try: x = string.atoi(rinp)
    except: pass
    return x

def stringsplitatoi(str, separator=''):
    if separator:
        words = string.split(str, separator)
    else:
        words = string.split(str)
    vals = []
    for word in words:
        vals.append(string.atoi(word))
    return vals

def stringsplitatof(str, separator=''):
    if separator:
        words = string.split(str, separator)
    else:
        words = string.split(str)
    vals = []
    for word in words:
        vals.append(string.atof(word))
    return vals

def stringsplitstrip(str, separator=''):
    # SPLITS BUT ALSO STRIPS EACH ITEM OF WHITESPACE
    if separator:
        words = string.split(str, separator)
    else:
        words = string.split(str)
    vals = []
    for word in words:
        vals.append(string.strip(word))
    return vals

def strbegin(str, phr):
    return str[:len(phr)] == phr

def strend(str, phr):
    return str[-len(phr):] == phr

def strfindall(str, phr):
    """FIND ALL INSTANCES OF phr IN str
    RETURN LIST OF POSITIONS WHERE phr IS FOUND
    (OR RETURN [] IF NOT FOUND)"""
    pos = []
    start = -1
    while 1:
        start = string.find(str, phr, start+1)
        if start > -1:
            pos.append(start)
        else:
            break
    return pos

def strbtw1(s, left, right=None):
    """RETURNS THE PART OF STRING s BETWEEN left & right
    EXAMPLE strbtw('det_lab.reg', '_', '.') RETURNS 'lab'
    EXAMPLE strbtw('det_{a}.reg', '{}') RETURNS 'a'"""
    out = None
    if right == None:
        if len(left) == 1:
            right = left
        elif len(left) == 2:
            left, right = left
    i1 = string.find(s, left)
    if (i1 > -1):
        i1 += len(left) - 1
        i2 = string.find(s, right, i1+1)
        if (i2 > i1):
            out = s[i1+1:i2]
    #out = string.split(s, left)[1]
    #out = string.split(out, right)[0]
    return out

def strbtw(s, left, right=None, r=False):
    """RETURNS THE PART OF STRING s BETWEEN left & right
    EXAMPLE strbtw('det_lab.reg', '_', '.') RETURNS 'lab'
    EXAMPLE strbtw('det_{a}.reg', '{}') RETURNS 'a'
    EXAMPLE strbtw('det_{{a}, b}.reg', '{}', r=1) RETURNS '{a}, b'"""
    out = None
    if right == None:
        if len(left) == 1:
            right = left
        elif len(left) == 2:
            left, right = left
    i1 = string.find(s, left)
    if (i1 > -1):
        i1 += len(left) - 1
        if r:  # search from the right
            i2 = string.rfind(s, right, i1+1)
        else:
            i2 = string.find(s, right, i1+1)
        if (i2 > i1):
            out = s[i1+1:i2]
    #out = string.split(s, left)[1]
    #out = string.split(out, right)[0]
    return out

def getanswer(question=''):
    ans = -1
    while ans == -1:
        inp = input(question)
        if inp:
            if string.upper(inp[0]) == 'Y':
                ans = 1
            if string.upper(inp[0]) == 'N':
                ans = 0
    return ans

ask = getanswer

#################################
# LISTS

def putids(selfvalues, selfids, ids, values):
    """ selfvalues = INITIAL ARRAY -OR- A DEFAULT VALUE FOR UNput ELEMENTS """
    try:
        n = len(selfvalues)
    except:
        n = len(selfids)
        selfvalues = zeros(n, int) + selfvalues
    indexlist = zeros(max(selfids)+1, int) - 1
    put(indexlist, array(selfids).astype(int), arange(len(selfids)))
    indices = take(indexlist, array(ids).astype(int))
    put(selfvalues, indices, values)
    return selfvalues

def takelist(a, ind):
    l = []
    for i in ind:
        l.append(a[i])
    return l

def common(id1, id2):
    # ASSUME NO IDS ARE NEGATIVE
    id1 = array(id1).astype(int)
    id2 = array(id2).astype(int)
    n = max((max(id1), max(id2)))
    in1 = zeros(n+1, int)
    in2 = zeros(n+1, int)
    put(in1, id1, 1)
    put(in2, id2, 1)
    inboth = in1 * in2
    ids = arange(n+1)
    ids = compress(inboth, ids)
    return ids

# FROM sparse.py ("sparse3")
def census(a, returndict=1):
    a = sort(ravel(a))
    if returndict:
        i = arange(min(a), max(a)+2)
    else:
        i = arange(max(a)+2)
    s = searchsorted(a, i)
    s = s[1:] - s[:-1]
    i = i[:-1]
    if returndict:
        print(i)
        print(s)
        #i, s = compress(s, (i, s))
        i = compress(s, i)
        s = compress(s, s)
        print('is')
        print(i)
        print(s)
        d = {}
        for ii in range(len(i)):
            d[i[ii]] = s[ii]
        return d
    else:
        return s

# ALSO CONSIDER: set(all) - set(ids)
def invertselection(ids, all):
    if type(all) == int:  # size input
        all = arange(all) + 1
        put(all, array(ids)-1, 0)
        all = compress(all, all)
        return all
    else:
        out = []
        for val in all:
            #if val not in ids:
            if not floatin(val, ids):
                out.append(val)
        return out

def mergeids(id1, id2):
    # ASSUME NO IDS ARE NEGATIVE
    id1 = array(id1).astype(int)
    id2 = array(id2).astype(int)
    idc = common(id1, id2)
    id3 = invertselection(idc, id2)
    return concatenate((id1, id3))


def findmatch1(x, xsearch, tol=1e-4):
    """RETURNS THE INDEX OF x WHERE xsearch IS FOUND"""
    i = argmin(abs(x - xsearch))
    if tol:
        if abs(x[i] - xsearch) > tol:
            print(xsearch, 'NOT FOUND IN findmatch1')
            i = -1
    return i

def findmatch(x, y, xsearch, ysearch, dtol=4, silent=0, returndist=0, xsorted=0):
    """FINDS AN OBJECT GIVEN A LIST OF POSITIONS AND SEARCH COORDINATE
    RETURNS INDEX OF THE OBJECT OR n IF NOT FOUND"""

    n = len(x)
    if silent < 0:
        print('n=', n)
    if not xsorted:
        SI = argsort(x)
        x = take(x, SI)
        y = take(y, SI)
    else:
        SI = arange(n)

    dist = 99  # IN CASE NO MATCH IS FOUND

    # SKIP AHEAD IN CATALOG TO x[i] = xsearch - dtol
    #print "SEARCHING..."
    if xsearch > dtol + max(x):
        done = 'too far'
    else:
        done = ''
        i = 0
        while xsearch - x[i] > dtol:
            if silent < 0:
                print(i, xsearch, x[i])
            i = i + 1

    while not done:
        if silent < 0:
            print(i, x[i], xsearch)
        if x[i] - xsearch > dtol:
            done = 'too far'
        else:
            dist = sqrt( (x[i] - xsearch) ** 2 + (y[i] - ysearch) ** 2)
            if dist < dtol:
                done = 'found'
            elif i == n - 1:
                done = 'last gal'
            else:
                i = i + 1
        if silent < 0:
            print(done)

    if done == 'found':
        if not silent:
            print('MATCH FOUND %1.f PIXELS AWAY AT (%.1f, %.1f)' % (dist, x[i], y[i]))
        ii = SI[i]
    else:
        if not silent:
            print('MATCH NOT FOUND')
        ii = n
    if returndist:
        return ii, dist
    else:
        return ii

def findmatches2(x1, y1, x2, y2):
    """MEASURES ALL DISTANCES, FINDS MINIMA
    SEARCHES FOR 2 IN 1
    RETURNS INDICES AND DISTANCES"""
    dx = subtract.outer(x1, x2)
    dy = subtract.outer(y1, y2)
    d = sqrt(dx**2 + dy**2)
    i = argmin(d,0)

    n1 = len(x1)
    n2 = len(x2)
    j = arange(n2)
    di = n2*i + j
    dmin = take(d,di)
    return i, dmin


def xref(data, ids, idcol=0, notfoundval=None):
    """CROSS-REFERENCES 2 DATA COLUMNS
    data MAY EITHER BE A 2-COLUMN ARRAY, OR A FILENAME CONTAINING THAT DATA
    ids ARE THE KEYS -- THE VALUES CORRESPONDING TO THESE (IN data's OTHER COLUMN) ARE RETURNED
    idcol TELLS WHICH COLUMN THE ids ARE IN (0 OR 1)"""
    if type(data) == str:
        data = transpose(loaddata(data))
    iddata = data[idcol].astype(int)
    xrefdata = data[not idcol].astype(int)

    dict = {}
    for i in range(len(iddata)):
        dict[iddata[i]] = xrefdata[i]

    xrefs = []
    for id in ids:
        xrefs.append(dict.get(id, notfoundval))

    return array(xrefs)


def takeid(data, id):
    """TAKES data COLUMNS CORRESPONDING TO id.
    data's ID's ARE IN ITS FIRST ROW"""
    dataids = data[0].astype(int)
    id = int(id)
    outdata = []
    i = 0
    while id != dataids[i]:
        i += 1
    return data[:,i]

def takeids(data, ids, idrow=0, keepzeros=0):
    """TAKES data COLUMNS CORRESPONDING TO ids.
    data's ID's ARE IN idrow, ITS FIRST ROW BY DEFAULT"""
    dataids = data[idrow].astype(int)
    ids = ids.astype(int)
    outdata = []
    n = data.shape[1]
    for id in ids:
        gotit = 0
        for i in range(n):
            if id == dataids[i]:
                gotit = 1
                break
        if gotit:
            outdata.append(data[:,i])
        elif keepzeros:
            outdata.append(0. * data[:,0])
    return transpose(array(outdata))


#################################
# FLUX, BPZ

bpzpath = os.environ.get('BPZPATH', '')

def bpzsedname(tb, seds, interp=2):
    if type(seds) == str:
        seds = loadfile(bpzpath + '/SED/' + seds)
    rb = roundint(tb)
    name = seds[rb-1]
    if abs(rb - tb) > 0.1:
        rb2 = roundint((tb - rb) * 3 + rb)
        name = name[:-4] + '-' + seds[rb2-1]
    return name

def bpztypename(tb, tbs, interp=2):
    rb = roundint(tb)
    name = tbs[rb-1]
    if abs(rb - tb) > 0.1:
        rb2 = roundint((tb - rb) * 3 + rb)
        name += '-' + tbs[rb2-1]
    return name

def addmags(m1, m2, dm1=0, dm2=0):
    # F = 10 ** (-0.4 * m)
    # dF = -0.4 * ln(10) * 10 ** (-0.4 * m) * dm = -0.921034 * F * dm
    # somehow this is wrong, should be:
    # dF / F = 10 ** (0.4 * dm) - 1  (as in bpz_tools.e_frac2mag)
    if (m1 >= 99 and m2 >= 99) or (dm1 >= 99 and dm2 >= 99):
        m = 99
        dm = 99
    elif m1 >= 99 or dm1 >= 99:
        m = m2
        dm = dm2
    elif m2 >= 99 or dm2 >= 99:
        m = m1
        dm = dm1
    else:  # NORMAL SITUATION
        F1 = 10 ** (-0.4 * m1)
        F2 = 10 ** (-0.4 * m2)
        F = F1 + F2
        m = -2.5 * log10(F)
        #dF1 = 0.921034 * F1 * dm1
        #dF2 = 0.921034 * F2 * dm2
        #dF = sqrt(dF1 ** 2 + dF2 ** 2)
        #dm = dF / F / 0.921034
        dm = sqrt( (F1 * dm1) ** 2 + (F2 * dm2) ** 2 ) / F
    output = (m, dm)

    return output

def addfluxes(F1, F2, dF1=0, dF2=0):
    F = F1 + F2
    dF = sqrt(dF1 ** 2 + dF2 ** 2)
    output = (F, dF)

    return output



#################################
# FROM Txitxo's bpz_tools.py

def sex2bpzmags(f,ef,zp=0.,sn_min=1.):
    """
    This function converts a pair of flux, error flux measurements from SExtractor
    into a pair of magnitude, magnitude error which conform to BPZ input standards:
    - Nondetections are characterized as mag=99, errormag=+m_1sigma
      - corrected error in previous version: was errormag=-m_1sigma
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

    m=zeros(len(f), float)
    em=zeros(len(ef), float)

    m = where(detected,-2.5*log10(f)+zp,m)
    m = where(nondetected,99.,m)
    m = where(nonobserved,-99.,m)

    em = where(detected,2.5*log10(1.+ef/f),em)
    #em = where(nondetected,2.5*log10(ef)-zp,em)
    em = where(nondetected,zp-2.5*log10(ef),em)
    #print "NOW WITH CORRECT SIGN FOR em"
    em = where(nonobserved,0.,em)
    return m,em


# NOTE PLACEMENT OF THIS LINE IS IMPORTANT
# coeio ALSO IMPORTS FROM coetools (THIS MODULE)
# SO TO AVOID AN INFINITE LOOP, coeio ONLY LOADS FROM coetools
#  THOSE FUNCTIONS DEFINED BEFORE coeio IS LOADED
from coeio import *
#from smooth import *
import string  # LOAD THIS AFTER numpy, BECAUSE numpy HAS ITS OWN string
from numpy.random import *
from compress2 import compress2 as compress
from MLab_coe import * #sum should be add.reduce
