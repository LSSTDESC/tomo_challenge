# $p/probplot.py bvizjh 1234
# $p/probplot.py bvizjh_cut_sexseg2_allobjs_newres_offset3_djh_jhbad-99_ssp25Myrz008 6319

# plots P(z) for a given galaxy ID #
# blue line is zb
# green lines are zbmin, zbmax
# red lines bracket 95% odds interval (hopefully that's what you want!)
# -- i could make this an input...

# from ~/UDF/probstest.py

# SINGLE GALAXY P(z)
# BPZ RUN WITH -ODDS 0.95
# 95% = 2-sigma
# OUTPUT ODDS = P(dz) = P(zb-dz:zb+dz), where dz = 2 * 0.06 * (1 + z)  [2 for 2-sigma]
# zbmin & zbmax CONTAIN 95% OF P(z) (EACH WING CONTAINS 2.5%)
# -- UNLESS P(z) IS TOO SHARP: MIN_RMS=0.06 TAKES OVER: dz >= 2 * 0.06 * (1+z)

#from ksbtools import *
#htmldir = loadfile('/home/coe/UDF/htmldir.txt')[0]

#import matplotlib
#matplotlib.use('TkAgg')
#from pylab import *
#from coeplot2 import fillbetween, clear
#from coeplot import *
from coetools import *
#from numpy import *
from coeplott import *
from os.path import exists, join

#htmldir = './'
#htmldir = loadfile('/Users/coe/p/htmldir.txt')[0]
htmldir = ''


def probplot(root, id1, zmax=None, zspec=-1, save=1, pshow=0, nomargins=0, outdir='', redo=False):
    outoforder = 1
    # LOAD PROBABILITIES, ONE LINE AT A TIME (MUCH MORE EFFICIENT!!)
    #outdir = 'bpzplots/'
    #outdir = htmldir + 'bpzplots/'
    #outdir = htmldir + 'probplots/'
    #if not os.path.exists(outdir):
    #    os.mkdir(outdir)

    # FIRST I CHECK IF THE PLOT EXISTS ALREADY
    # THIS MAKES IT TOUGH TO CALL WITH i3
    #  BECAUSE I HAVE TO READ THE PROBS FILE FIRST TO GET THE ID NUMBER...
    outimg = 'probplot%d.png' % id1
    if exists(join(outdir, outimg)) and not pshow and not redo:
        print join(outdir, outimg), 'ALREADY EXISTS'
    else:
        if save:
            print 'CREATING ' + join(outdir, outimg) + '...'
        if nomargins:
            figure(1, figsize=(4,2))
            clf()
            axes([0.03,0.2,0.94,0.8])
        else:
            figure()
            clf()
        ioff()
        fprob = open(root + '.probs', 'r')
        line = fprob.readline() # header
        zrange = strbtw(line, 'arange(', ')')
        zrange = stringsplitatof(zrange, ',')
        #print zrange
        z = arange(zrange[0], zrange[1], zrange[2])
        
        if type(id1) == str:
            i = string.atoi(id1[1:]) - 1
            for ii in range(i+1):
                line = fprob.readline()
            i = string.find(line, ' ')
            id = line[:i]
            id1 = id[:]
            print id1, 'id1'
            id = string.atoi(id)
        else:
            id = 0
            while (id <> id1) and line and ((id < id1) or outoforder):
                line = fprob.readline()
                i = string.find(line, ' ')
                id = line[:i]
                id = string.atoi(id)
            if (id <> id1):
                print '%d NOT FOUND' % id1
                print 'QUITTING.'
                sys.exit()
            #print 'FOUND IT...'
        
        fprob.close()

        prob = stringsplitatof(line[i:-1])
        n = len(prob)
        #z = (arange(n) + 1) * 0.01
	if zmax == None:
            zmax = max(z)
	else:
	   nmax = roundint(zmax / zrange[2])
	   z = z[:nmax]  # nmax+1
	   prob = prob[:nmax]  # nmax+1
        #z = (arange(1200) + 1) * 0.01
        #zc, pc = compress(prob, (z, prob))
        zc, pc = z, prob
        pmax = max(prob)
        #print 'total = ', total(pc)
        #points(zc, pc)


        # LOAD BPZ RESULTS, ONE LINE AT A TIME (MUCH MORE EFFICIENT!!)
        fin = open(root + '_bpz.cat', 'r')
        #fin = open(root + '_b.bpz', 'r')
        #fin = open(root + '.bpz', 'r')
        line = '#'
        while line[0] == '#':
            lastline = line[:]
            line = fin.readline() # header
        labels = string.split(lastline[1:-1])

        line = string.strip(line)
        i = string.find(line, ' ')
        id = line[:i]
        id = string.atoi(id)
        while id <> id1 and (id < id1 or outoforder):
            line = fin.readline()
            line = string.strip(line)
            i = string.find(line, ' ')
            id = line[:i]
            id = string.atoi(id)

        fin.close()

        data = stringsplitatof(line[i:])
        labels = labels[1:] # get rid of "id"
        for i in range(len(labels)-1):
            exec(labels[i] + ' = data[i]')
        # ['id', 'zb', 'zbmin', 'zbmax', 'tb', 'odds', 'zml', 'tml', 'chisq', 'M0', 'nf', 'jhgood', 'stel', 'x', 'y']

        dz = 2 * 0.06 * (1 + zb)  # 2-sigma = 95% (ODDS)
        zlo = zb - dz
        zhi = zb + dz

        #p = FramedPlot()
        if zspec >= 0:
            #p.add(Curve([zspec, zspec], [0, pmax*1.05], color='magenta', linewidth=3, linetype='dashed'))
            #p.add(Curve([zspec, zspec], [0, pmax*1.05], color='orange', linewidth=5))
            lw = [5,3][nomargins]
            #plot([zspec, zspec], [0, pmax*1.05], color=(1,0.5,0), linewidth=lw)
            #vline([zspec], (1,0.5,0), linewidth=lw)
            vline([zspec], (1,0,0), linewidth=lw, alpha=0.5)
            #p.add(Curve([zspec, zspec], [pmax*1.01, pmax*1.05], color='red', linewidth=3))
            #p.add(Curve([zspec, zspec], [-0.01*pmax, -0.0*pmax], color='red', linewidth=3))
        #p.add(FillBelow(zc, pc, color='blue'))
        #plot(zc.tolist(), pc)
	#zc = zc[:200]
	#pc = pc[:200]
        fillbetween(zc, zeros(len(pc)), zc, pc, facecolor='blue')
        #fillbetween(zc, pc, zc, zeros(len(pc)), facecolor='blue')
	#print pc
        #p.add(FillBetween(zc, pc, zc, zeros(len(pc)), color='blue'))
	plot([zc[0], zc[-1]], [0, 0], color='white')
        #p.add(Slope(0, color='white'))
        if 0:
            p.add(FillBelow(zc, pc, color='grey50'))
            p.add(Curve([zb, zb], [0, pmax*.1], color='cyan', linewidth=5))
            p.add(Curve([zbmin, zbmin], [0, pmax*.07], color='green', linewidth=5))
            p.add(Curve([zbmax, zbmax], [0, pmax*.07], color='green', linewidth=5))
            p.add(Curve([zlo, zlo], [0, pmax*.07], color='red'))
            p.add(Curve([zhi, zhi], [0, pmax*.07], color='red'))
        #p.add(Curve(zc, pc, linewidth=3))
        #title('ODDS = %.3f' % odds)
        #p.title = 'ODDS = %.3f' % odds
        #p.show()
        #p.xrange = [0, 12]
        #p.xrange = [0, 6]
        #p.x.ticks = 13
        #p.xrange = [0, zmax]
        #p.x.ticks = int(zmax)+1
        #p.x.subticks = 4
        #p.y.tickdir = 1
        #p.x1.tickdir = 1
        #p.y.draw_ticklabels = 0
        #p.y.draw_ticks = 0
        #p.yrange = [0, pmax*1.05]
        #p.yrange = [-0.01*pmax, pmax*1.05]
        #p.yrange = [0, 0.03]
        #p.title = None
        #p.page_margin = 0
        xlabel('z')
        if nomargins:
            ax = gca()
            #ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ylabel('P(z)')
        ylim(0,1.05*pmax)
        if save:
            #print 'SAVING', outdir+outimg
            #p.write_img(400, 200, outdir+outimg)
            savefig(join(outdir, outimg))
            os.system('chmod 644 '+join(outdir, outimg))
        if pshow:
            show()
            print 'KILL PLOT WINOW TO TERMINATE OR CONTINUE.'
	    #pause()
            #p.show()
        #os.system('eog %s &' % (outdir+outimg))
        #print 'DONE'
	#clear()  # OTHERWISE FIGS WILL DRAW OVER THEMSELVES!

if __name__ == '__main__':
    #root = sys.argv[1]
    id1 = sys.argv[2]
    if id1[0] <> 'i':
        id1 = string.atoi(id1)
    params = params_cl()
    save_plots = 'SAVE' in params
    show_plots = 1 - save_plots
    zspec = params.get('ZSPEC', -1)
    zmax = params.get('ZMAX', None)
    nomargins = 'NOMARGINS' in params
    #probplot(sys.argv[1], string.atoi(sys.argv[2]), -1, 0, 1)
    probplot(sys.argv[1], id1, zmax=zmax, zspec=zspec, save=save_plots, pshow=show_plots, nomargins=nomargins)
else:
    pass
