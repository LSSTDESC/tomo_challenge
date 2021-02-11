## Automatically adapted for numpy Jun 08, 2006 by convertcode.py

#Useful functions and definitions

import os,sys
from types import *
#from Numeric import *
from numpy import *
#from MLab import *
from MLab_coe import *
from time import *
from spline import *
from string import *  # AFTER numpy, WHICH HAS ITS OWN split, (and more?)
#import whrandom
import random
#To use the astrometrical functions defined below
#try: import ephem
#except: pass


#If biggles installed allow the plot options in the tests
plots=1
try: 
    from biggles import *
except: plots=0

pi=3.141592653

def ejecuta(command=None,verbose=1):
    import os
    if verbose: print(command)
    os.system(command)
    

def ask(what="?"):
    """
    Usage:
    ans=ask(pregunta)
    This function prints the string what, 
    (usually a question) and asks for input 
    from the user. It returns the value 0 if the 
    answer starts by 'n' and 1 otherwise, even 
    if the input is just hitting 'enter'
    """
    if what[-1]!='\n': what=what+'\n'
    ans=input(what)
    try:
        if ans[0]=='n': return 0
    except:
        pass
    return 1
    
#Input/Output subroutines

#Read/write headers

def get_header(file):
    """ Returns a string containing all the lines 
    at the top of a file which start by '#'"""
    buffer=''
    for line in open(file).readlines():
        if line[0]=='#': buffer=buffer+line
        else: break
    return buffer

def put_header(file,text,comment=1):
    """Adds text (starting by '#' and ending by '\n')
    to the top of a file."""
    if len(text)==0: return
    if text[0]!='#' and comment: text='#'+text
    if text[-1]!='\n':text=text+'\n'
    buffer=text+open(file).read()
    open(file,'w').write(buffer)

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

def put_str(file,tupla):
    """ Writes tuple of string lists to a file
        Usage:
	  put_str(file,(x,y,z))
    """    
    if type(tupla)!=type((2,)):
        raise 'Need a tuple of variables'
    f=open(file,'w')    
    for i in range(1,len(tupla)):
        if len(tupla[i])!=len(tupla[0]):
            raise 'Variable lists have different lenght'
    for i in range(len(tupla[0])):
        cosas=[]
        for j in range(len(tupla)):cosas.append(str(tupla[j][i]))
        f.write(join(cosas)+'\n')
    f.close()

#Files containing data

def get_data(file,cols=0,nrows='all'):
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
        return tuple(data) 

def write(file,variables,header='',format='',append='no'):
    """ Writes tuple of list/arrays to a file 
        Usage:
	  put_data(file,(x,y,z),header,format)
	where header is any string  
        and format is a string of the type:
           '%f %f %i ' 
	The default format is all strings
    """    
    if type(variables)!=type((2,)):
        raise 'Need a tuple of variables'
    if format=='': format='%s  '*len(variables)
    if append=='yes': f=open(file,'a')
    else: f=open(file,'w')
    if header!="":
        if header[0]!='#': header='#'+header
        if header[-1]!='\n': header=header+'\n'
        f.write(header)
    for i in range(len(variables[0])):
        cosas=[]
        for j in range(len(variables)):
            cosas.append(variables[j][i])
        line=format % tuple(cosas)             
        f.write(line+'\n')
    f.close()

put_data=write

#Read/write 2D arrays

def get_2Darray(file,cols='all',nrows='all',verbose='no'):
    """Read the data on the defined columns of a file 
    to an 2 array
    Usage:
    x=get_2Darray(file)
    x=get_2Darray(file,range(len(p))
    x=get_2Darray(file,range(0,10,2),nrows=5000)
    Returns x(nrows,ncols)
    """
    if cols=='all':
        #Get the number of columns in the file
        for line in open(file).readlines():
            pieces=split(line)
            if len(pieces)==0: continue
            if line[0]=='#':continue
            nc=len(pieces)
            cols=list(range(nc))
            if verbose=='yes': print('cols=',cols)
            break
    else:
        nc=len(cols)
    
    lista=get_data(file,cols,nrows)
    nl=len(lista[0])
    x=zeros((nl,nc),float)
    for i in range(nc):x[:,i]=lista[i]
    return x

def put_2Darray(file,array,header='',format='',append='no'):
    """ Writes a 2D array to a file, where the first 
    index changes along the lines and the second along
    the columns
    Usage: put_2Darray(file,a,header,format)
	where header is any string  
        and format is a string of the type:
           '%f %f %i ' 
    """
    lista=[]
    for i in range(array.shape[1]):lista.append(array[:,i])
    lista=tuple(lista)
    put_data(file,lista,header,format,append)
       

class watch:
    def set(self):
        self.time0=time()
        print('')
        print('Current time ',ctime(self.time0))
        print()
    def check(self):
        if self.time0:
            print()
            print("Elapsed time", strftime('%H:%M:%S',gmtime(time()-self.time0)))
            print()
        else:
            print()
            print('You have not set the initial time')
            print()

def params_file(file):
    """ 
	Read a input file containing the name of several parameters 
        and their values with the following format:
        
	KEY1   value1,value2,value3   # comment
        KEY2   value

        Returns the dictionary
        dict['KEY1']=(value1,value2,value3)
        dict['KEY2']=value
    """
    dict={}
    for line in open(file,'r').readlines():
        if line[0]==' ' or line[0]=='#': continue 
        halves=line.split('#')
	#replace commas in case they're present
        halves[0]=halves[0].replace(',',' ') 	
        pieces=halves[0].split()
        if len(pieces)==0: continue
        key=pieces[0]
        #if type(key)<>type(''):
        #     raise 'Keyword not string!'
        if len(pieces)<2:
            mensaje='No value(s) for parameter  '+key
            raise ValueError(mensaje)
        dict[key]=tuple(pieces[1:]) 
        if len(dict[key])==1: dict[key]=dict[key][0]
    return dict

def params_commandline(lista):
    """ Read an input list (e.g. command line) 
	containing the name of several parameters 
        and their values with the following format:
        
        ['-KEY1','value1,value2,value3','-KEY2','value',etc.] 
          
         Returns a dictionary containing 
        dict['KEY1']=(value1,value2,value3)
        dict['KEY2']=value 
	etc.
    """
    if len(lista)%2!=0:
        print('Error: The number of parameter names and values does not match')
        sys.exit()
    dict={}
    for i in range(0,len(lista),2):
        key=lista[i]
        if type(key)!=type(''):
            raise 'Keyword not string!'
        #replace commas in case they're present
        if key[0]=='-':key=key[1:]
        lista[i+1]=replace(lista[i+1],',',' ')
        values=tuple(split(lista[i+1]))
        if len(values)<1:
            mensaje='No value(s) for parameter  '+key
            raise mensaje
        dict[key]=values
        if len(dict[key])==1: dict[key]=dict[key][0]
    return dict

def view_keys(dict):
    """Prints sorted dictionary keys"""
    claves=list(dict.keys())
    claves.sort()
    for line in claves:
        print(line.upper(),'  =  ',dict[line])

class params:
    """This class defines and manages a parameter dictionary"""
    def __init__(self,d=None):
        if d==None:self.d={}
        else: self.d=d

    # Define a few useful methods:

    def fromfile(self,file):
        """Update the parameter dictionary with a file"""
        self.d.update(params_file(file)) 
    
    def fromcommandline(self,command_line):
        """Update the parameter dictionary with command line options (sys.argv[i:])"""
        self.d.update(params_commandline(command_line))

    def update(self,dict):
        """Update the parameter information with a dictionary"""
        for key in list(dict.keys()):
            self.d[key]=dict[key]
                   
    def check(self):
        """Interactively check the values of the parameters"""
        view_keys(self.d)
        paso1=input('Do you want to change any parameter?(y/n)\n')
        while paso1[0] == 'y':
            key=input('Which one?\n')
            if key not in self.d:
                paso2=input("This parameter is not in the dictionary.\
Do you want to include it?(y/n)\n")
                if paso2[0]=='y':
                    value=input('value(s) of '+key+'?= ')
                    self.d[key]=tuple(split(replace(value,',',' ')))
                else:continue
            else:
                value=input('New value(s) of '+key+'?= ')
                self.d[key]=tuple(split(replace(value,',',' ')))
            view_keys(self.d)
            paso1=input('Anything else?(y/n)\n')            

    def write(self,file):
        claves=list(self.d.keys())
        claves.sort()
        buffer=''
        for key in claves:
            if type(self.d[key])==type((2,)):
                values=list(map(str,self.d[key]))
                line=key+' '+string.join(values,',')
            else:
                line=key+' '+str(self.d[key])
            buffer=buffer+line+'\n'
        print(line)
        open(file,'w').write(buffer)

#List of colors from biggles:
def biggles_colors():
    try: import biggles
    except: pass
    return get_str('/home/txitxo/Python/biggles_colors.txt',0)


#Some miscellaneous numerical functions

def ascend(x):
    """True if vector x is monotonically ascendent, false otherwise 
       Recommended usage: 
       if not ascend(x): sort(x) 
    """
    return alltrue(greater_equal(x[1:],x[0:-1]))


#def match_resol(xg,yg,xf,method="linear"):
#    """ 
#    Interpolates and/or extrapolate yg, defined on xg, onto the xf coordinate set.
#    Options are 'lineal' or 'spline' (uses spline.py from Johan Hibscham)
#    Usage:
#    ygn=match_resol(xg,yg,xf,'spline')
#    """
#    if method<>"spline":
#	if type(xf)==type(1.): xf=array([xf])
#	ng=len(xg)
#	d=(yg[1:]-yg[0:-1])/(xg[1:]-xg[0:-1])
#	#Get positions of the new x coordinates
#	ind=clip(searchsorted(xg,xf)-1,0,ng-2)
#	ygn=take(yg,ind)+take(d,ind)*(xf-take(xg,ind))
#	if len(ygn)==1: ygn=ygn[0]
#	return ygn
#    else:
#	low_slope=(yg[1]-yg[0])/(xg[1]-xg[0])
#	high_slope=(yg[-1]-yg[-2])/(xg[-1]-xg[-2])
#	sp=Spline(xg,yg,low_slope,high_slope)
#	return sp(xf)	


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
	


def overlap(x,y):
    """Returns 1 if vectors x and y overlap, 0 otherwise"""
    if (x[0]<=y[-1] and x[-1]>y[0]) or (y[0]<=x[-1] and y[-1]>x[0]):
        return 1
    else: return 0


def match_objects(coords1,coords2,tail1=(),tail2=(),accuracy=1.):
    """
    where coords1 and coords2 are tuples containing 1-D arrays,
    and tail1 and tail2 are tuples containing sequences of 
    arbitrary types
    Usage:
    results=match_objects((x1,y1),(x2,y2),(a1,b1,c1),(d2,e2),accuracy=.5)
    It returns the sequence x1,y1,a1,b1,c1,d2,e2 for those objects 
    which have dist(x1,y1-x2,y2)< accuracy
    """
    acc2=accuracy**2
    nc=len(coords1)
    np1=len(coords1[0])
    np2=len(coords2[0])
    a1=array(coords1)
    a2=array(coords2)
    nt1=len(tail1)
    for i in range(nt1): 
        if len(tail1[i])!= np1: raise 'Not the same lenght as coordinates 1'
    nt2=len(tail2)
    for i in range(nt2): 
        if len(tail2[i])!= np2: raise 'Not the same lenght as coordinates 2'
    match=zeros(np1, int)-1
    for j in range(np1):
        #dist=add.reduce((a1[:,j,NewAxis]-a2[:,:])**2)
        a1j = a1[:,j]
        dist=add.reduce((reshape(a1j, (len(a1j), 1)) - a2)**2)
        i_min=argmin(dist)
        if dist[i_min]<acc2:match[j]=i_min
    good=greater_equal(match,0)
    n1=compress(good,list(range(np1)))    
    match=compress(good,match)
    a1=compress(good,a1)
    salida=list(a1)
    for i in range(nt1):
        if type(tail1[i][0])==type('si'):
            t=[]
            for j in n1: t.append(tail1[i][j])
        else:
            t=take(tail1[i],n1)
        salida.append(t)
    for i in range(nt2):
        if type(tail2[i][0])==type('si'):
            t=[]
            for j in match: t.append(tail2[i][j])
        else:
            t=take(tail2[i],match)
        salida.append(t)
    return salida


def match_min(coords1,coords2,tail1=(),tail2=()):
    """
    where coords1 and coords2 are tuples containing 1-D arrays,
    and tail1 and tail2 are tuples containing sequences of 
    arbitrary types

    Usage:

    results=match_min((x1,y1),(x2,y2),(a1,b1,c1),(d2,e2))
    It returns the sequence x1,y1,a1,b1,c1,d2,e2, dist_min 
    where dist_min is the minimal value of dist(x1,y1-x2,y2)
    The match looks for the objects with minimal distance
    """
    nc=len(coords1)
    np1=len(coords1[0])
    np2=len(coords2[0])
    a1=array(coords1)
    a2=array(coords2)
    nt1=len(tail1)
    for i in range(nt1): 
        if len(tail1[i])!= np1: raise 'Not the same lenght as coordinates 1'
    nt2=len(tail2)
    for i in range(nt2): 
        if len(tail2[i])!= np2: raise 'Not the same lenght as coordinates 2'
    match=zeros(np1, int)-1

    dist_min=zeros(np1)*1.

    for j in range(np1):
        #dist=sqrt(add.reduce((a1[:,j,NewAxis]-a2[:,:])**2))
        a1j = a1[:,j]
        dist=add.reduce((reshape(a1j, (len(a1j), 1)) - a2)**2)
        i_min=argmin(dist)
        dist_min[j]=dist[i_min]
        match[j]=i_min

    salida=list(a1)
    for i in range(nt1):salida.append(tail1[i])
    
    for i in range(nt2):
        if type(tail2[i][0])==type('si'):
            t=[]
            for j in match: t.append(tail2[i][j])
        else:
            t=take(tail2[i],match)
        salida.append(t)

    salida.append(dist_min)
    return tuple(salida)


def match_min2(coords1,coords2,tail1=(),tail2=()):
    """
    where coords1 and coords2 are tuples containing 1-D arrays,
    and tail1 and tail2 are tuples containing sequences of 
    arbitrary types

    Usage:

    results=match_min((x1,y1),(x2,y2),(a1,b1,c1),(d2,e2))
    It returns the sequence x1,y1,x2,y2,a1,b1,c1,d2,e2, dist_min 
    where dist_min is the minimal value of dist(x1,y1-x2,y2)
    The match looks for the objects with minimal distance
    """
    nc=len(coords1)
    np1=len(coords1[0])
    np2=len(coords2[0])
    a1=array(coords1)
    a2=array(coords2)
    nt1=len(tail1)
    for i in range(nt1): 
        if len(tail1[i])!= np1: raise 'Not the same lenght as coordinates 1'
    nt2=len(tail2)
    for i in range(nt2): 
        if len(tail2[i])!= np2: raise 'Not the same lenght as coordinates 2'
    match=zeros(np1, int)-1
    dist_min=zeros(np1)*1.
    x2=zeros(np1)*1.
    y2=zeros(np1)*1.
    for j in range(np1):
        #dist=add.reduce((a1[:,j,NewAxis]-a2[:,:])**2)
        a1j = a1[:,j]
        dist=add.reduce((reshape(a1j, (len(a1j), 1)) - a2)**2)
        i_min=argmin(dist)
        dist_min[j]=dist[i_min]
        x2[j],y2[j]=a2[0,i_min],a2[1,i_min]
        match[j]=i_min
        
    salida=list(a1)
    salida.append(x2)
    salida.append(y2)

    for i in range(nt1):salida.append(tail1[i])
    
    for i in range(nt2):
        if type(tail2[i][0])==type('si'):
            t=[]
            for j in match: t.append(tail2[i][j])
        else:
            t=take(tail2[i],match)
        salida.append(t)

    salida.append(dist_min)
    return tuple(salida)

def dist(x,y,xc=0.,yc=0.):
    """Distance between point (x,y) and a center (xc,yc)"""
    return sqrt((x-xc)**2+(y-yc)**2)

def loc2d(a,extremum='max'):
    """ Locates the maximum of an 2D array
        Usage:
	max_vec=max_loc2d(a)
    """
    forma=a.shape
    if len(forma)>2:raise "Array dimension > 2"
    if extremum!='min' and extremum!='max':
        raise 'Which extremum are you looking for?'
    x=ravel(a)
    if extremum=='min': i=argmin(x)
    else: i=argmax(x)
    i1=i/forma[1]
    i2=i%forma[1]
    return i1,i2
    
def hist(a,bins):
    """
    Histogram of 'a' defined on the bin grid 'bins'
       Usage: h=hist(p,xp)
    """
    n=searchsorted(sort(a),bins)
    n=concatenate([n,[len(a)]])
    n=array(list(map(float,n)))
#    n=array(n)
    return n[1:]-n[:-1]

#def hist2D(a,xbins,ybins):
#    """
#    Histogram of 'a' defined on the grid xbins X ybins
#       Usage: h=hist2D(p,xp,yp)
#       Points larger than xbins[-1],ybins[-1] are asigned to
#       the 'last' bin
#    """   
#    nx=len(xbins)
#    ny=len(ybins)
#    #We use searchsorted differenty from the 1-D case
#    hx=searchsorted(xbins,a)
#    hy=searchsorted(ybins,a)        
#    h=zeros((nx,ny))
#    for i in range(len(hx)):
#        for j in range(len(hy)):
#            h[hx[i],hy[i]]=+1
#    for k in range(len(a)):
#        for i in range(len(xbins)):
#            for j in range(len(ybins)):
#                if a[k]>xbins[i] and a[k]<xbins[i+1] \
#                   and a[k]>ybins[i] and a[k]< ybins[i+1]:
#                    h[i,j]=h[i,j]+1
#                    break
#                else:
                                        

def bin_stats(x,y,xbins,stat='average'):
    """Given the variable y=f(x), and 
    the bins limits xbins, return the 
    corresponding statistics, e.g. <y(xbins)>
    Options are rms, median y average
    """
    nbins=len(xbins)
    if   stat=='average' or stat=='mean':          func=mean
    elif stat=='median':                           func=median
    elif stat=='rms' or stat=='std'              : func=std
    elif stat=='std_robust' or stat=='rms_robust': func=std_robust
    elif stat=='mean_robust':                      func=mean_robust
    elif stat=='median_robust':                    func=median_robust
    elif stat=='sum':                              func=sum
    results=[]
    for i in range(nbins):
        if i<nbins-1:
            good=(greater_equal(x,xbins[i])
                  *less(x,xbins[i+1]))
        else: good=(greater_equal(x,xbins[-1]))
        if sum(good)>1.: results.append(func(compress(good,y)))
        else:
            results.append(0.)
            print('Bin starting at xbins[%i] has %i points' % (i,sum(good)))
    return array(results)


def bin_aver(x,y,xbins):
    return bin_stats(x,y,xbins,stat='average')

def p2p(x):
    return max(x) - min(x)

def autobin_stats(x,y,n_bins=8,stat='average',n_points=None):
    """
    Given the variable y=f(x), form n_bins, distributing the 
    points equally among them. Return the average x position 
    of the points in each bin, and the corresponding statistic stat(y).
    n_points supersedes the value of n_bins and makes the bins 
    have exactly n_points each
    Usage:
      xb,yb=autobin_stats(x,y,n_bins=8,'median')
      xb,yb=autobin_stats(x,y,n_points=5)
    """
    
    if not ascend(x):
        ix=argsort(x)
        x=take(x,ix)
        y=take(y,ix)
    n=len(x)
    if n_points==None: 
        #This throws out some points
        n_points=n/n_bins
    else: 
        n_bins=n/n_points
        #if there are more that 2 points in the last bin, add another bin
        if n%n_points>2: n_bins=n_bins+1
    
    if n_points<=1:
        print('Only 1 or less points per bin, output will be sorted input vector with rms==y')
        return x,y
    xb,yb=[],[]
    
    #print 'stat', stat
    if   stat=='average' or stat=='mean':          func=mean
    elif stat=='median':                           func=median
    elif stat=='rms' or stat=='std'              : func=std
    elif stat=='std_robust' or stat=='rms_robust': func=std_robust
    elif stat=='mean_robust':                      func=mean_robust
    elif stat=='median_robust':                    func=median_robust
    elif stat=='p2p':                              func=p2p  # --DC
    elif stat=='min':                              func=min  # --DC
    elif stat=='max':                              func=max  # --DC
    
    for i in range(n_bins):
        xb.append(mean(x[i*n_points:(i+1)*n_points]))
        if func==std and n_points==2:
            print('n_points==2; too few points to determine rms')
            print('Returning abs(y1-y2)/2. in each bin as rms')
            yb.append(abs(y[i*n_points]-y[i*n_points+1])/2.)
        else:
            yb.append(func(y[i*n_points:(i+1)*n_points]))
        if i>2 and xb[-1]==xb[-2]: 
            yb[-2]=(yb[-2]+yb[-1])/2.
            xb=xb[:-1]
            yb=yb[:-1]
    return array(xb),array(yb)


def purge_outliers(x,n_sigma=3.,n=5):
    #Experimental yet. Only 1 dimension
    for i in range(n):
        med=median(x)
        #rms=std_log(x)
        rms=std(x)
        x=compress(less_equal(abs(x-med),n_sigma*rms),x)
    return x        

class stat_robust:
    #Generates robust statistics using a sigma clipping
    #algorithm. It is controlled by the parameters n_sigma
    #and n, the number of iterations
    def __init__(self,x,n_sigma=3,n=5,reject_fraction=None):
        self.x=x
        self.n_sigma=n_sigma
        self.n=n
        self.reject_fraction=reject_fraction

    def run(self):
        good=ones(len(self.x))
        nx=sum(good)
        if self.reject_fraction==None:
            for i in range(self.n):
                if i>0: xs=compress(good,self.x)
                else: xs=self.x
                #            aver=mean(xs)
                aver=median(xs)
                std1=std(xs)
                good=good*less_equal(abs(self.x-aver),self.n_sigma*std1)
                nnx=sum(good)
                if nnx==nx: break
                else: nx=nnx
        else:
            np=float(len(self.x))
            nmin=int((0.5*self.reject_fraction)*np)
            nmax=int((1.-0.5*self.reject_fraction)*np)
            orden=argsort(self.x)
            connect(arange(len(self.x)),sort(self.x))
            good=greater(orden,nmin)*less(orden,nmax)

        self.remaining=compress(good,self.x)
        self.max=max(self.remaining)
        self.min=min(self.remaining)
        self.mean=mean(self.remaining)
        self.rms=std(self.remaining)
        self.rms0=rms(self.remaining)  # --DC
        self.median=median(self.remaining)
        self.outliers=compress(logical_not(good),self.x)
        self.n_remaining=len(self.remaining)
        self.n_outliers=len(self.outliers)
        self.fraction=1.-(float(self.n_remaining)/float(len(self.x)))
    
def std_robust(x,n_sigma=3.,n=5):
    x=purge_outliers(x,n_sigma,n)
    return std(x-mean(x))

def mean_robust(x,n_sigma=3.,n=5):
    x=purge_outliers(x,n_sigma,n)
    return mean(x)

def median_robust(x,n_sigma=3.,n=5):
    x=purge_outliers(x,n_sigma,n)
    return median(x)

def std_log(x,fa=sqrt(20.)):
    dx=std(x)
    #print "std(x)",dx,
    #if abs(dx)<1e-100:dx=mean(abs(x))
    a=fa*dx
    #print sqrt(average(a*a*log(1.+x*x/(a*a)))),
    #print std_robust(x,3,3),
    #print len(x)
    return sqrt(average(a*a*log(1.+x*x/(a*a))))


#def std_log(x,fa=20.):
#    dx=median(abs(x))
#    if abs(dx)<1e-100:dx=mean(abs(x))
#    a=fa*dx
#    return sqrt(average(a*a*log10(1.+x*x/(a*a))))

def med_thr(x,thr=0.2,max_it=10):
    xm=median(x)
    xm0=xm+thr
    for i in range(max_it):
        good=less_equal(x-xm,thr)*greater_equal(x-xm,-thr)
        xm=median(compress(good,x))
        if abs(xm-xm0)<thr/1000.: break
        xm0=xm
        # print xm
    return xm

def std_thr(x,thr=0.2,max_it=10):
    xm=med_thr(x,thr,max_it)
    good=less_equal(x-xm,thr)*greater_equal(x-xm,-thr)
    return std(compress(good,x))

def out_thr(x,thr=0.2,max_it=10):
    xm=med_thr(x,thr,max_it)
    good=less_equal(x-xm,thr)*greater_equal(x-xm,-thr)
    return len(x)-sum(good)



#def bin_aver(x,y,xbins):
#    """Given the variable y=f(x), and 
#    the bins limits xbins, return the 
#    average <y(xbins)>"""
#    a=argsort(x)
#    nbins=len(xbins)
#    y=take(y,a)
#    x=take(x,a)
#    n=searchsorted(x,xbins)
#    results=xbins*0.
#    num=hist(x,xbins)
#    for i in range(nbins):
#	if i< nbins-1:
#	    results[i]=add.reduce(y[n[i]:n[i+1]])
#	else:
#	    results[i]=add.reduce(y[n[i]:])
#	if num[i]>0:
#	    results[i]=results[i]/num[i]
#    return results



def multicompress(condition,variables):
    lista=list(variables)
    n=len(lista)
    for i in range(n):	lista[i]=compress(condition,lista[i])
    return tuple(lista)

def multisort(first,followers):
    #sorts the vector first and matches the ordering 
    # of followers to it
    #Usage:
    # new_followers=multi_sort(first,followers)
    order=argsort(first)
    if type(followers)!= type((1,)):
        return take(followers,order)
    else:
        nvectors=len(followers)
        lista=[]
        for i in range(nvectors):
            lista.append(take(followers[i],order))
        return tuple(lista)
	
def erfc(x):
    """
    Returns the complementary error function erfc(x)
    erfc(x)=1-erf(x)=2/sqrt(pi)*\int_x^\inf e^-t^2 dt   
    """
    try: x.shape
    except: x=array([x])
    z=abs(x)
    t=1./(1.+0.5*z)
    erfcc=t*exp(-z*z-
        1.26551223+t*(
        1.00002368+t*(
        0.37409196+t*(
        0.09678418+t*(
       -0.18628806+t*(
        0.27886807+t*(
       -1.13520398+t*(
        1.48851587+t*(
       -0.82215223+t*0.17087277)
        ))))))))
    erfcc=where(less(x,0.),2.-erfcc,erfcc)
    return erfcc

def erf(x):
    """
    Returns the error function erf(x)
    erf(x)=2/sqrt(pi)\int_0^x \int e^-t^2 dt
    """
    return 1.-erfc(x)

def erf_brute(x):
    step=0.00001
    t=arange(0.,x+step,step)
    f=2./sqrt(pi)*exp(-t*t)
    return sum(f)*step

def erfc_brute(x):
    return 1.-erf_brute(x)


def gauss_int_brute(x=arange(0.,3.,.01),average=0.,sigma=1.):
    step=x[1]-x[0]
    gn=1./sqrt(2.*pi)/sigma*exp(-(x-average)**2/2./sigma**2)
    return add.accumulate(gn)*step

def gauss_int_erf(x=(0.,1.),average=0.,sigma=1.):
    """
    Returns integral (x) of p=int_{-x1}^{+x} 1/sqrt(2 pi)/sigma exp(-(t-a)/2sigma^2) dt
    """
    x=(x-average)/sqrt(2.)/sigma
    return (erf(x)-erf(x[0]))*.5

gauss_int=gauss_int_erf

def inv_gauss_int(p):
    #Brute force approach. Limited accuracy for >3sigma
    #find something better 
    #DO NOT USE IN LOOPS (very slow)
    """
    Calculates the x sigma value corresponding to p
    p=int_{-x}^{+x} g(x) dx
    """
    if p<0. or p>1.:
        print('Wrong value for p(',p,')!')
        sys.exit()
    step=.00001
    xn=arange(0.,4.+step,step)
    gn=1./sqrt(2.*pi)*exp(-xn**2/2.)
    cgn=add.accumulate(gn)*step
    p=p/2.
    ind=searchsorted(cgn,p)
    return xn[ind]

def points(x,y,limits=(None,None,None,None),title='Plot'):
    #Quickly plot two vectors using biggles
    p=FramedPlot()
    if limits[0]!=None and limits[1]!=None:
        p.xrange=limits[0],limits[1]
    if limits[2]!=None and limits[3]!=None:
        p.yrange=limits[2],limits[3]
    p.add(Points(x,y))
    p.add(Slope(0.))
    p.title=title
    p.show()

def pointswriteimg(x,y,limits=(None,None,None,None),title='Plot', xsize=600, ysize=600, name=''):
    #Quickly plot two vectors using biggles
    p=FramedPlot()
    if limits[0]!=None and limits[1]!=None:
        p.xrange=limits[0],limits[1]
    if limits[2]!=None and limits[3]!=None:
        p.yrange=limits[2],limits[3]
    p.add(Points(x,y))
    p.add(Slope(0.))
    p.title=title
    p.show()
    if not name:
        name = title + '.png'
    p.write_img(xsize, ysize, name)

def connect(x,y,limits=(None,None,None,None)):
    #Quickly plot two vectors using biggles
    p=FramedPlot()
    if limits[0]!=None and limits[1]!=None:
        p.xrange=limits[0],limits[1]
    if limits[2]!=None and limits[3]!=None:
        p.yrange=limits[2],limits[3]
    p.add(Curve(x,y))
    p.show()

def mark_outliers(x,n_sigma=3.,n=5): # --DC
    # from purge_outliers
    #Experimental yet. Only 1 dimension
    nx = len(x)
    ii = list(range(nx))
    for i in range(n):
        med=median(x)
        rms=std(x)
        ii,x=compress(less_equal(abs(x-med),n_sigma*rms), (ii,x))
    outliers = ones(nx)
    put(outliers, ii.astype(int), 0)
    return outliers

def mark_faroutliers(x,n_sigma=3.,n=5,n_farout=2): # --DC
    # from purge_outliers
    # REPEATS n TIMES
    nx = len(x)
    xg = x[:]
    for i in range(n):
        dx = abs(x - median(x))
        outliers = greater(dx, n_sigma * std(xg))
        xg = compress(logical_not(outliers), x)
        xglo, xghi = min(xg), max(xg)
        xgmed = (xglo + xghi) / 2.
        xgrange = xghi - xglo
        n_out = abs(x - xgmed) / xgrange - 0.5
        outliers = greater(n_out, n_farout)
    n_out = (x - xgmed) / xgrange
    toolo = less(n_out + 0.5, -n_farout)
    outliers = outliers - 2 * toolo
    return outliers

def pointsrobust(x,y,limits=(None,None,None,None),title='Plot'): # --DC
    #Quickly plot two vectors using biggles
    p=FramedPlot()
    if limits[0]!=None and limits[1]!=None:
        p.xrange=limits[0],limits[1]
    if limits[2]!=None and limits[3]!=None:
        p.yrange=limits[2],limits[3]
    p.add(Slope(0.))
    outliers = mark_faroutliers(y)
    if sum(outliers):
        xg, yg = compress(logical_not(outliers), (x, y))
        p.add(Points(xg,yg))
        yglo, yghi = min(yg), max(yg)
        toohi = greater(y, yghi)
        toolo = less(y, yglo)
        y = clip(y, yglo, yghi)
        dy = (yghi - yglo) / 20.
        if sum(toohi):
            xohi, yohi = compress(toohi, (x, y))
            for i in range(len(yohi)):
                p.add(Curve([xohi[i], xohi[i]], [yohi[i], yohi[i]+dy]))
            p.add(Points(xohi, yohi+dy, type='half filled triangle'))
        if sum(toolo):
            xolo, yolo = compress(toolo, (x, y))
            for i in range(len(yolo)):
                p.add(Curve([xolo[i], xolo[i]], [yolo[i], yolo[i]+dy]))
            p.add(Points(xolo, yolo-dy, type='half filled inverted triangle'))
    else:
        p.add(Points(x,y))  # no outliers
    p.title=title
    p.yrange = [yglo-2*dy, yghi+2*dy]
    p.show()

class NumberCounts:
    #Define number counts and produce some plots
    def __init__(self,m,dm=1.,mmin=10.,mmax=35.,area=1.,xcor=None,ycor=None,type_cor='negative'):
        #xcor and ycor are corrections to the total number counts, e.g. area or incompleteness
        if mmin==10. and mmax==35.:
            xm=arange(10.,35.,dm)
            imin=searchsorted(xm,min(m))
            imax=searchsorted(xm,max(m))
            self.xm=xm[imin-1:imax]
            print('min(m),max(m)',min(m),max(m))
            print('self.xm[0],self.xm[-1]',self.xm[0],self.xm[-1])
        else:
            self.xm=arange(mmin,mmax+dm,dm)
            
        self.dnc=hist(m,self.xm)
        self.xm=self.xm+dm*0.5
        self.dnc=self.dnc/area/dm
        if xcor!=None and ycor!=None:
            if type_cor=='negative':
                self.dnc-=match_resol(xcor,ycor,self.xm)
                self.dnc=clip(self.dnc,0.,1e50)
            elif type_cor=='positive': self.dnc+=match_resol(xcor,ycor,self.xm)
            elif type_cor=='multiplicative': self.dnc*=match_resol(xcor,ycor,self.xm)

        self.cnc=add.accumulate(self.dnc)
        try:
            self.ldnc=log10(self.dnc)
        except:
            print('Differential numbers counts contains bins with zero galaxies')
            print('We set those values to 1e-1')
            dnc=where(equal(self.dnc,0.),1e-2,self.dnc)
            self.ldnc=log10(dnc)
                        
        try:
            self.lcnc=log10(self.cnc)
        except:
            print('Could not calculate log of cumulative numbers counts')
            
class lsq:
    #Defines a least squares minimum estimator given two 
    #vectors x and y
    def __init__(self,x,y,dy=0.):
        try: dy.shape
        except: dy=x*0.+1.
        dy2=dy**2
        s=add.reduce(1./dy2)
        sx=add.reduce(x/dy2)
        sy=add.reduce(y/dy2)
        sxx=add.reduce(x*x/dy2)
        sxy=add.reduce(x*y/dy2)
        delta=s*sxx-sx*sx
        self.a=(sxx*sy-sx*sxy)/delta
        self.b=(s*sxy-sx*sy)/delta
        self.da=sqrt(sxx/delta)
        self.db=sqrt(s/delta)
    def fit(self,x):
        return self.b*x+self.a

def rotation(x,y,angle):
    xp=x*cos(angle)+y*sin(angle)
    yp=-x*sin(angle)+y*cos(angle)
    return xp,yp





#Tests 

def Testing(test):
    print('Testing ',test,'...')

def test():
    """ Tests the functions defined in this module"""

    Testing("I/O FUNCTIONS")
    
    test="put_str and get_str"
    x=arange(100.)
    y=list(map(str,sin(x)))
    x=list(map(str,x))
    put_str('test.txt',(x,y))
    xn,yn=get_str('test.txt',list(range(2)))
    if xn!=x:raise test
    if yn!=y:raise test

    test='put_data and get_data'
    Testing(test)
    x=arange(100.)
    y=sin(x)
    put_data('test.dat',(x,y),'header of test.dat','%.18f %.18f')
    xn,yn=get_data('test.dat',list(range(2)))
    if sometrue(not_equal(xn,x)): raise test
    if sometrue(not_equal(yn,y)): raise test

    test="put_header and get_header"
    Testing(test)
    f=open('test.head','w')
    f.close()

    lines=["This is a test header line",
	   "This is another test header line",
	   "This is yet another test header line"]

    put_header('test.head',lines[2])
    put_header('test.head',lines[1])
    put_header('test.head',lines[0])

    nlines=get_header("test.head")
    lines=string.join(['#'+x+'\n' for x in lines],'')

    if lines!=nlines: 
        print(repr(lines))
        print(repr(nlines))
        raise test

    test='put_2Darray and get_2Darray'
    Testing(test)
    x=arange(200.)
    y=reshape(x,(40,-1))
    put_2Darray('test.dat',y,'header of test.dat')
    yn=get_2Darray('test.dat')
    comp=not_equal(yn,y)
    for i in range(yn.shape[1]):
        if sometrue(comp[:,i]): raise test
    
    #Testing("MISC NUMERICAL FUNCTIONS")

    test='ascend'
    Testing(test)
    y=sin(arange(100))
    z=sort(y)
    if not ascend(z): raise test
    z=z[::-1]
    if ascend(z): raise test

#    test="hist"
#    Testing(test)
#    x=arange(0.,100.,.1)
#    y=arange(100.)
#    h=hist(x,y)
#    points(h,ones(100)*10)
#    if sometrue(not_equal(h,ones(100)*10)): raise test

    test="bin_aver"
    Testing(test)
    x=arange(0.,10.1,.1)
    xb=arange(10.)+.01
    y=x
    yb=bin_aver(x,y,xb)+.45
    yr=arange(10)+1.
    if sometrue(greater_equal(yb-yr,1e-10)): raise test
    
    test='dist'
    Testing(test)
    a=arange(0,2.*pi,pi/6.)
    x=10.*sin(a)+3.
    y=5*cos(a)-3.
    d=sqrt((((x-3.)**2+(y+3.)**2)))
    nd=list(map(dist,x,y,ones(len(x))*3.,ones(len(x))*-3.))
    if sometrue(not_equal(d,nd)):
        print(d)
        print(nd)
        raise test


    test="loc2d"
    Testing(test)
    m=fromfunction(dist,(10,10))
    if loc2d(m)!=(9,9): raise test
    if loc2d(m,'min')!=(0,0): raise test


    test="match_objects"
    Testing(test)
    x=arange(10.)
    y=arange(10,20.)
    t1=(list(map(str,x)),list(map(str,y)))
    x=x+RandomArray.random(x.shape)*.707/2.
    y=y+RandomArray.random(y.shape)*.707/2.
    x0=arange(10000.)
    y0=arange(10.,10010.)
    t2=(list(map(str,x0)),list(map(str,y0)))
    cosas1=match_objects((x,y),(x0,y0),t1,t2,accuracy=.5)
    if not (cosas1[2]==cosas1[4] and cosas1[3]==cosas1[5]):
        raise test

    test="match_min"
    Testing(test)
    x=arange(10.)
    y=arange(10,20.)
    t1=(list(map(str,x)),list(map(str,y)))
    x=x+RandomArray.random(x.shape)*.707/2.
    y=y+RandomArray.random(y.shape)*.707/2.
    x0=arange(10000.)
    y0=arange(10.,10010.)
    t2=(list(map(str,x0)),list(map(str,y0)))
    cosas1=match_min((x,y),(x0,y0),t1,t2)
    #put_data('bobo',cosas1)
    #os.system('more bobo')
    if not (cosas1[2]==cosas1[4] and cosas1[3]==cosas1[5]):
        raise test


    test="match_resol"
    Testing(test)
    xobs=arange(0.,10.,.33)
    yobs=cos(xobs)*exp(-xobs)
    xt=arange(0.,10.,.33/2.)
    yt=match_resol(xobs,yobs,xt)
    ytobs=cos(xt)*exp(-xt)
    if plots:
        plot=FramedPlot()
        plot.add(Points(xobs,yobs,color="blue",type='cross'))
        plot.add(Curve(xt,yt,color="red"))
        plot.add(Points(xt,yt,color="red",type='square'))
        plot.show()
        print("The crosses are the original data")
        print("The continuous line/red squares represents the interpolation")
    else:
        print('   X     Y_Interp  Y_expected')
        for i in range(len(x)):
            print(3*'%7.4f  '% (xt[i],yt[i],ytobs[i]))

    test="gauss_int"
    Testing(test)
    x=arange(0.,3,.5)
    p=gauss_int(x)
    pt=array(
	[0.,.19146,.34134,.43319,.47725,.49379])
    diff=abs(p-pt)
    print(diff)
    if sometrue(greater(diff,2e-5)): raise test

    test="inv_gauss_int"
    Testing(test)
    for i in range(len(pt)):
        z=inv_gauss_int(2.*pt[i])
        if abs(x[i]-z) > 1e-3 : raise test

    print('Everything tested seems to be OK in useful.py')


if __name__ == '__main__':test()
else: pass


