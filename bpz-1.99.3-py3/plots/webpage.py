# ~/bpz-1.99.3/plots/webpage.py

# BPZ results for catalog (SED fits and P(z))

# python $BPZPATH/plots/webpage.py root
#   Produces html/index.html by default

# python $BPZPATH/plots/webpage.py root i1-10
# python $BPZPATH/plots/webpage.py root i-10
#   First 10 objects

# python $BPZPATH/plots/webpage.py root 2225,1971,7725
#   Objects with ID numbers 2225, 1971, 7725

# python $BPZPATH/plots/webpage.py root some.i
#   Objects with ID numbers listed in file some.i (one per line)

# python $BPZPATH/plots/webpage.py root -ZMAX 6
#   Max z for P(z)  (Default 7)

# python $BPZPATH/plots/webpage.py root -DIR myhtmldir
#   Makes myhtmldir/index.html

# python $BPZPATH/plots/webpage.py root -REDO
# python $BPZPATH/plots/webpage.py root -REDO sed
# python $BPZPATH/plots/webpage.py root -REDO prob
#   Redo all plots or just the sed or P(z) plots
#   Default: Don't redo plots if they already exist

#################################
# Color image & segment support turned off:
# python $BPZPATH/plots/webpage.py root ids -DIR outdir -SEGM segmids
# python $BPZPATH/plots/webpage.py A1689 1702,1718 -COLOR A1689.tif -OFFSET 329,212 -SEGMOFFSET -21,-20

# ~/bpz-1.99.2/plots/webpage.py
# ~/ACS/CL0024/colorpro/webpage.py
# label.py
# ~/p/stampbpzhtml.py

# id x y segmid?
# outdir
# segm.fits
# color image

from coetools import *
#import Image #, ImageDraw, ImageFont
import sedplotAB, probplot
#from coeplot2a import *  # prange

#dir = join(home, 'ACS/color/production/CL0024')
#file = join(dir, 'CL0024.png')

# colorstamps/stamp###.png
# segm/segm###.gif
# sedplots/UDF_sed_###.png
# probplots/probplot###.png

n = 200  # IMAGE SIZE

def colorstamps(cat, outdir, colorfile, addon='', offset=(0,0)):
    im = Image.open(colorfile)
    nx, ny = im.size
    dx = dy = n/2
    dxo, dyo = offset
    outdir = join(outdir, addon)
    if not exists(outdir):
        os.mkdir(outdir)
        
    for i in range(cat.len()):
        id = roundint(cat.id[i])
        outfile = 'stamp%d.png' % id
        outfile = join(outdir, outfile)
        if os.path.exists(outfile):
            continue
        else:
            print outfile
        
        x = roundint(cat.x[i])
        y = roundint(cat.y[i])
        stamp = im.crop((x-dx-dxo,ny-y-dy+dyo,x+dx-dxo,ny-y+dy+dyo))
        stamp.save(outfile)

def segmstamps(cat, outdir, segmfile, offset=(0,0)):
    segm = loadfits(segmfile)
    dx = dy = n/2
    dxo, dyo = offset
    if not exists(outdir):
        os.mkdir(outdir)
    
    segmids = cat.get('segmid', cat.id).round().astype(int)
    segmgif = Image.new("L", (n,n))
    for i in range(cat.len()):
        id = segmids[i]
        outfile = 'segm%d.gif' % id
        outfile = join(outdir, outfile)
        #delfile(outfile)
        if os.path.exists(outfile):
            continue
        else:
            print outfile
        
        x = roundint(cat.x[i])
        y = roundint(cat.y[i])
        segmstamp = segm[y-dy-dyo:y+dy-dyo,x-dx-dxo:x+dx-dxo]
        idstamp = equal(segmstamp, id).round().astype(int)\
            + greater(segmstamp, 0).round().astype(int)
        idstamp = 127 * (2 - idstamp)
        data = ravel(idstamp)
        segmgif.putdata(data)
        segmgif = segmgif.transpose(Image.FLIP_TOP_BOTTOM)
        segmgif.save(outfile)


def sedplots(cat, root, outdir, redo=False):
    if not exists(outdir):
        os.mkdir(outdir)
    
    ids = cat.get('segmid', cat.id).round().astype(int)
    b=sedplotAB.bpzPlots(root, ids, probs=None)
    redo = redo in [True, 'sed']
    b.flux_comparison_plots(show_plots=0, save_plots='png', colors={}, nomargins=0, outdir=outdir, redo=redo)
    #os.system('\mv %s_sed_*.png %s' % (root, outdir))


def probplots(cat, root, outdir, zmax=7, redo=False):
    if not exists(root+'.probs'):
        return

    if not exists(outdir):
        os.mkdir(outdir)
    
    ids = cat.get('segmid', cat.id).round().astype(int)
    redo = redo in [True, 'prob']
    for i in range(cat.len()):
        id = ids[i]
        probplot.probplot(root, id, zmax=zmax, nomargins=0, outdir=outdir, redo=redo)
    
    #os.system('\mv probplot*png ' + outdir)

def webpage(cat, bpzroot, outfile, ncolor=1, idfac=1.):
    fout = open(outfile, 'w')

    coloraddons = list(string.lowercase)
    coloraddons[0] = ''
    coloraddons = coloraddons[:ncolor]
    
    bpzpath = os.environ.get('BPZPATH')
    inroll = join(bpzpath, 'plots')
    if 0:
        inroll = join(inroll, 'rollover.txt')
        for line in loadfile(inroll, keepnewlines=1):
            fout.write(line)
    
    fout.write('\n')
    if 0:
        fout.write('Roll mouse over color images to view segments.<br><br>\n\n')
    fout.write('<h1>BPZ results for %s.cat</h1>\n\n' % bpzroot)
    ids = cat.id.round().astype(int)
    segmids = cat.get('segmid', ids).round().astype(int)
    for i in range(cat.len()):
        id = ids[i]
        segmid = segmids[i]
        id2 = id * idfac
        fout.write('Object #%s' % num2str(id2))
        #fout.write('Object #%d' % id)
        if 'zb' in cat.labels:
            fout.write(' &nbsp; BPZ = %.2f' % cat.zb[i])
        if ('zbmin' in cat.labels) and ('zbmax' in cat.labels):
            fout.write(' [%.2f--%.2f]' % (cat.zbmin[i], cat.zbmax[i]))
        if 'tb' in cat.labels:
            fout.write(' &nbsp; type = %.2f' % cat.tb[i])
        if 'chisq2' in cat.labels:
            fout.write(' &nbsp; chisq2 = %.2f' % cat.chisq2[i])
        if 'odds' in cat.labels:
            fout.write(' &nbsp; ODDS = %.2f' % cat.odds[i])
        if 'zspec' in cat.labels:
            fout.write(' &nbsp; spec-z = %.2f' % cat.zspec[i])
        fout.write('<br>\n')
        
        for addon in coloraddons:
            fout.write(' <a href="#">')
            fout.write(' <img src="colorstamps/%s/stamp%d.png"' % (addon, id))
            fout.write(' hsrc="segm/segm%d.gif"' % segmid)
            fout.write(' border=0')
            fout.write('></a>\n')
            # fout.write(' border=0 USEMAP="#map%d"' % id)
        
        if 0:
            fout.write(' <img src="segm/segm%d.gif" border=0>\n' % segmid)
        #fout.write(' <img src="sedplots/%s_sed_%d.png" border=0>\n' % (bpzroot, segmid))
        #fout.write(' <img src="probplots/probplot%d.png" border=0>' % segmid)
        fout.write(' <img src="sedplots/%s_sed_%d.png"   border=0 height=300 width=400>\n' % (bpzroot, segmid))
        fout.write(' <img src="probplots/probplot%d.png" border=0 height=300 width=400>' % segmid)
        fout.write('<br>\n')
        fout.write('<br>\n\n')
    
    fout.close()

# WANT INDICES AS OUTPUT
# None: all
# deez.i: ids in text file
# i0: 1st (i = 0) object
# 285: id = 285
# 285,63: ids = 285,63

# python $BPZPATH/plots/webpage.py root ids -DIR outdir -SEGM segmids
def run():
    bpzroot = sys.argv[1]
    #cat = loadcat(bpzroot+'.cat')
    #cat = loadcat(bpzroot+'_photbpz.cat')
    cat = loadcat(bpzroot+'_bpz.cat')

    ids = None
    mycat = cat
    if len(sys.argv) > 2:
        if sys.argv[2][0] <> '-':
            id_str = sys.argv[2]
            
            if id_str[-2:] == '.i':  # External file with IDs (one per line)
                ids = ravel(loaddata(id_str).round().astype(int))
                mycat = cat.takeids(ids)
            elif id_str[0] == 'i':  # Indices
                num = id_str[1:]
                if string.find(num, '-') == -1:  # single number
                    i = int(id_str[1:])
                    mycat = cat.take(array([i+1]))
                else:
                    lo, hi = string.split(num, '-')
                    lo = lo or 1
                    lo = int(lo)
                    hi = int(hi)
                    hi = hi or cat.len()
                    ii = arange(lo-1, hi)
                    mycat = cat.take(ii)
            else:  # IDs separated by commas
                ids = stringsplitatoi(id_str, ',')
                mycat = cat.takeids(ids)

    params = params_cl()
    outdir = params.get('DIR', 'html')
    if not exists(outdir):
        os.mkdir(outdir)

    if 0:
        segmfile = params.get('SEGM', 'segm')

        if 'SEGMID' in params.keys():
            segm_str = params['SEGMID']
            if id_str[-2:] == '.i':
                segmids = ravel(loaddata(id_str).round().astype(int))
            else:
                segmids = stringsplitatoi(id_str, ',')
            mycat.assign('segmid', segmids)

        #colorfile = params['COLOR']
        colorfiles = params.get('COLOR', bpzroot+'.png')
        #colorfiles = list(colorfiles)  # MAKE LIST IF ONLY ONE
        if type(colorfiles) == str:
            colorfiles = [colorfiles]

        offset = params.get('OFFSET', '0,0')
        offset = stringsplitatoi(offset, ',')

        segmoffset = params.get('SEGMOFFSET', '0,0')
        segmoffset = stringsplitatoi(segmoffset, ',')
    
    idfac = params.get('IDFAC', 1.)

    zmax = params.get('ZMAX', 7.)

    redo = params.get('REDO', False)
    if redo == None:
        redo = True

    ltrs = list(string.lowercase)
    ltrs[0] = ''

    colorfiles = []
    if 0:
        for colorfile, addon in zip(colorfiles, ltrs):
            colorstamps(mycat, join(outdir, 'colorstamps'), colorfile, addon, offset)  # id x y
    
        segmstamps(mycat, join(outdir, 'segm'), segmfile, segmoffset)  # id/segmid x y
    
    sedplots(mycat, bpzroot, join(outdir, 'sedplots'), redo=redo)  # id
    probplots(mycat, bpzroot, join(outdir, 'probplots'), zmax=zmax, redo=redo)  # id
    webpage(mycat, bpzroot, join(outdir, 'index.html'), len(colorfiles), idfac=idfac)  # id segmid

if __name__ == '__main__':run()
else: pass
