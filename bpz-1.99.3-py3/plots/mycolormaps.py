# ~/lp/mycolormaps.py

# ~/p/
# colormapnew.py
# colormapdata.py
# colormaps.py

# /Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/matplotlib/
# _cm.py
# pyplot.py
# colors.py

# TO SET ONE OF THESE COLORMAPS, JUST CALL, e.g.:
# >>> gray1070()
# OR:
# >>> colormap(x, y, z, cmap=cm.gray1070)

from pylab import *
import matplotlib

LUTSIZE = mpl.rcParams['image.lut']

# cm.datad.keys()


#################################

lo, hi = 0.1, 0.7
x = 1  # DOESN'T MATTER??
_gray1070_data =  {
    'red':   ((0., x, lo), (1, hi, x)),
    'green': ((0., x, lo), (1, hi, x)),
    'blue':  ((0., x, lo), (1, hi, x))}

cm.gray1070 = matplotlib.colors.LinearSegmentedColormap('gray1070', _gray1070_data, LUTSIZE)

cm.datad['gray1070'] = _gray1070_data

def gray1070():
    rc('image', cmap='gray1070')
    im = gci()
    
    if im is not None:
        im.set_cmap(cm.gray1070)
    draw_if_interactive()

#################################

lo, hi = 0.2, 0.7
x = 1  # DOESN'T MATTER??
_gray2070_data =  {
    'red':   ((0., x, lo), (1, hi, x)),
    'green': ((0., x, lo), (1, hi, x)),
    'blue':  ((0., x, lo), (1, hi, x))}

cm.gray2070 = matplotlib.colors.LinearSegmentedColormap('gray2070', _gray2070_data, LUTSIZE)

cm.datad['gray2070'] = _gray2070_data

def gray2070():
    rc('image', cmap='gray2070')
    im = gci()
    
    if im is not None:
        im.set_cmap(cm.gray2070)
    draw_if_interactive()

#################################

lo, hi = 0.0, 1.0
x = 1  # DOESN'T MATTER??
_gray0010_data =  {
    'red':   ((0., x, lo), (1, hi, x)),
    'green': ((0., x, lo), (1, hi, x)),
    'blue':  ((0., x, lo), (1, hi, x))}

cm.gray0010 = matplotlib.colors.LinearSegmentedColormap('gray0010', _gray0010_data, LUTSIZE)

cm.datad['gray0010'] = _gray0010_data

def gray0010():
    rc('image', cmap='gray0010')
    im = gci()
    
    if im is not None:
        im.set_cmap(cm.gray0010)
    draw_if_interactive()

#################################

lo, hi = 1.0, 0.0
x = 1  # DOESN'T MATTER??
_grayinv_data =  {
    'red':   ((0., x, lo), (1, hi, x)),
    'green': ((0., x, lo), (1, hi, x)),
    'blue':  ((0., x, lo), (1, hi, x))}

cm.grayinv = matplotlib.colors.LinearSegmentedColormap('grayinv', _grayinv_data, LUTSIZE)

cm.datad['grayinv'] = _grayinv_data

def grayinv():
    rc('image', cmap='grayinv')
    im = gci()
    
    if im is not None:
        im.set_cmap(cm.grayinv)
    draw_if_interactive()

#################################

lo, hi = 1.0, 0.2
x = 1  # DOESN'T MATTER??
_grayinv20_data =  {
    'red':   ((0., x, lo), (1, hi, x)),
    'green': ((0., x, lo), (1, hi, x)),
    'blue':  ((0., x, lo), (1, hi, x))}

cm.grayinv20 = matplotlib.colors.LinearSegmentedColormap('grayinv20', _grayinv20_data, LUTSIZE)

cm.datad['grayinv20'] = _grayinv20_data

def grayinv20():
    rc('image', cmap='grayinv20')
    im = gci()
    
    if im is not None:
        im.set_cmap(cm.grayinv20)
    draw_if_interactive()

#################################

lo, hi = 0.7, 0.2
x = 1  # DOESN'T MATTER??
_grayinv2070_data =  {
    'red':   ((0., x, lo), (1, hi, x)),
    'green': ((0., x, lo), (1, hi, x)),
    'blue':  ((0., x, lo), (1, hi, x))}

cm.grayinv2070 = matplotlib.colors.LinearSegmentedColormap('grayinv2070', _grayinv2070_data, LUTSIZE)

cm.datad['grayinv2070'] = _grayinv2070_data

def grayinv2070():
    rc('image', cmap='grayinv2070')
    im = gci()
    
    if im is not None:
        im.set_cmap(cm.grayinv2070)
    draw_if_interactive()

#################################

lo, hi = 0.9, 0.2
x = 1  # DOESN'T MATTER??
_grayinv2090_data =  {
    'red':   ((0., x, lo), (1, hi, x)),
    'green': ((0., x, lo), (1, hi, x)),
    'blue':  ((0., x, lo), (1, hi, x))}

cm.grayinv2090 = matplotlib.colors.LinearSegmentedColormap('grayinv2090', _grayinv2090_data, LUTSIZE)

cm.datad['grayinv2090'] = _grayinv2090_data

def grayinv2090():
    rc('image', cmap='grayinv2090')
    im = gci()
    
    if im is not None:
        im.set_cmap(cm.grayinv2090)
    draw_if_interactive()

#################################
# ~/p/colormapnew2.py

#colormap_map[ 'jetw' ] = array(

data = array(
   [ [ 1, 1, 1 ],
     [ 0.0, 0.0, 0.5625 ],
     [ 0.0, 0.0, 0.625 ],
     [ 0.0, 0.0, 0.6875 ],
     [ 0.0, 0.0, 0.75 ],
     [ 0.0, 0.0, 0.8125 ],
     [ 0.0, 0.0, 0.875 ],
     [ 0.0, 0.0, 0.9375 ],
     [ 0.0, 0.0, 1.0 ],
     [ 0.0, 0.0625, 1.0 ],
     [ 0.0, 0.125, 1.0 ],
     [ 0.0, 0.1875, 1.0 ],
     [ 0.0, 0.25, 1.0 ],
     [ 0.0, 0.3125, 1.0 ],
     [ 0.0, 0.375, 1.0 ],
     [ 0.0, 0.4375, 1.0 ],
     [ 0.0, 0.5, 1.0 ],
     [ 0.0, 0.5625, 1.0 ],
     [ 0.0, 0.625, 1.0 ],
     [ 0.0, 0.6875, 1.0 ],
     [ 0.0, 0.75, 1.0 ],
     [ 0.0, 0.8125, 1.0 ],
     [ 0.0, 0.875, 1.0 ],
     [ 0.0, 0.9375, 1.0 ],
     [ 0.0, 1.0, 1.0 ],
     [ 0.0625, 1.0, 1.0 ],
     [ 0.125, 1.0, 0.9375 ],
     [ 0.1875, 1.0, 0.875 ],
     [ 0.25, 1.0, 0.8125 ],
     [ 0.3125, 1.0, 0.75 ],
     [ 0.375, 1.0, 0.6875 ],
     [ 0.4375, 1.0, 0.625 ],
     [ 0.5, 1.0, 0.5625 ],
     [ 0.5625, 1.0, 0.5 ],
     [ 0.625, 1.0, 0.4375 ],
     [ 0.6875, 1.0, 0.375 ],
     [ 0.75, 1.0, 0.3125 ],
     [ 0.8125, 1.0, 0.25 ],
     [ 0.875, 1.0, 0.1875 ],
     [ 0.9375, 1.0, 0.125 ],
     [ 1.0, 1.0, 0.0625 ],
     [ 1.0, 1.0, 0.0 ],
     [ 1.0, 0.9375, 0.0 ],
     [ 1.0, 0.875, 0.0 ],
     [ 1.0, 0.8125, 0.0 ],
     [ 1.0, 0.75, 0.0 ],
     [ 1.0, 0.6875, 0.0 ],
     [ 1.0, 0.625, 0.0 ],
     [ 1.0, 0.5625, 0.0 ],
     [ 1.0, 0.5, 0.0 ],
     [ 1.0, 0.4375, 0.0 ],
     [ 1.0, 0.375, 0.0 ],
     [ 1.0, 0.3125, 0.0 ],
     [ 1.0, 0.25, 0.0 ],
     [ 1.0, 0.1875, 0.0 ],
     [ 1.0, 0.125, 0.0 ],
     [ 1.0, 0.0625, 0.0 ],
     [ 1.0, 0.0, 0.0 ],
     [ 0.9375, 0.0, 0.0 ],
     [ 0.875, 0.0, 0.0 ],
     [ 0.8125, 0.0, 0.0 ],
     [ 0.75, 0.0, 0.0 ],
     [ 0.6875, 0.0, 0.0 ],
     [ 0.625, 0.0, 0.0 ],
     [ 0.5625, 0.0, 0.0 ] ],
     float )

red, green, blue = transpose(data)
n = len(red)
x = mgrid[0:1:n*1j]

_jetw_data = {}
_jetw_data['red']   = transpose(array([x, red, red]))
_jetw_data['green'] = transpose(array([x, green, green]))
_jetw_data['blue']  = transpose(array([x, blue, blue]))

cm.jetw = matplotlib.colors.LinearSegmentedColormap('jetw', _jetw_data, LUTSIZE)

cm.datad['jetw'] = _jetw_data

def jetw():
    rc('image', cmap='jetw')
    im = gci()
    
    if im is not None:
        im.set_cmap(cm.jetw)
    draw_if_interactive()

