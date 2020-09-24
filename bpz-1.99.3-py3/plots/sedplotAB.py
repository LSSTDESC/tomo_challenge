# sedplotAB.py

# python $BPZPATH/plots/sedplotAB.py XXX
#   Plot SED fits for all objects in order:
#   (Input files are XXX.bpz, XXX.columns)

# python $BPZPATH/plots/sedplotAB.py XXX NNN
#   Plot SED fits for object with id = 2225

# python $BPZPATH/plots/sedplotAB.py XXX i4
#   Plot SED fits for 4th object in catalog (numbering beginning at 1, not 0)

# -SAVE to save (.png, .eps, .pdf) instead of just display

# -ZSPEC uses spec-z instead of photo-z

# -VERBOSE give more info


from useful import *
from bpz_tools import *
from coeplott import *  # pylab, prange, rectangle
import shelve
from MLab_coe import ndec
from coeio import loaddata, loadfile, params_cl, str2num, loaddict, findmatch1, pause  #, prange, plotconfig
from numpy import *  # (REIMPORT!, OTHERWISE numarray TAKES OVER)
import string  # AFTER numpy, WHICH HAS ITS OWN string
from os.path import exists, join

def obs_spectrum_AB(sed, z):
    lam, flam = obs_spectrum(sed, z)
    fnu = flam * lam**2
    m = flux2mag(fnu)  # -2.5 * log10(fnu)
    return lam, m

iyl = 0
def legend1(lab, color, ms=12):
    global iyl
    xl = 15000
    yl = 20
    dyl = 0.3
    dxl = 600
    plot([xl], [yl + dyl * iyl], 'o', mec=color, mfc=color, ms=ms)
    text(xl + dxl, yl + dyl * iyl, lab, va='center')
    iyl += 1
    

class bpzPlots:
    """
    Usage:
    python bpzPlots.py 
    Define a series of plots given the name of a bpz run and a string which
    identifies an object, and which can be either its ID of a pair of x,y
    coordinates e.g. 1001,2000.In the later case, a tuple containing the indexes of
    the columns containing X,Y information in the main catalog have to be supplied. 
    The plots will be produced for the closest
    object in the catalog. Optionally, different values for the bpz catalog
    and flux comparison file can be introduced
    """
    
    def __init__(self, run_name, id_str=None,
                 cat=None,probs=None,flux_comparison=None,columns=None,
                 xy_cols=None,spectra='CWWSB_fuv_f_f.list',
		 show_plots=1,save_plots=0,verbose=False
                 ):
        self.run_name=run_name
        
        #Define names of data catalogs
        if cat == None: self.cat=run_name+'.bpz'
        else: self.cat =bpz
        
	self.bpzstr = loadfile(self.cat)
	self.bpzparams = {}
	i = 0
	while self.bpzstr[i][:2] == '##':
	    line = self.bpzstr[i][2:]
	    if '=' in line:
		[key, value] = string.split(line, '=')
		self.bpzparams[key] = value
	    i = i + 1
	self.bpzcols = []
	while self.bpzstr[i][:2] == '# ':
	    line = self.bpzstr[i][2:]
	    [col, key] = string.split(line)
	    self.bpzcols.append(key)	   
	    i = i + 1
	
	self.spectra = self.bpzparams.get('SPECTRA', 'CWWSB_fuv_f_f.list')
	self.columns = self.bpzparams.get('COLUMNS', run_name+'.columns')
	self.flux_comparison = self.bpzparams.get('FLUX_COMPARISON', run_name+'.flux_comparison')
	self.interp = str2num(self.bpzparams.get('INTERP', '2'))

        self.verbose = verbose
	params = params_cl()
        if verbose:
            print 'params', params

	self.bw = 'BW' in params
        self.thick = 'THICK' in params

        #Look for the corresponding ID number

        if ',' in str(id_str) and xy_cols<>None: #If the input is a pair of X,Y coordinates
            x,y=map(float,tuple(split(id_str),','))
            x_cat,y_cat=get_data(self.cat,xy_cols)
            self.id=argmin(dist(x_cat,y_cat,x,y))+1
	elif id_str == None:
	    self.id = None
	else:
	    if id_str[-2:] == '.i':
	        self.id = ravel(loaddata(id_str).astype(int))
	    elif id_str[0] == 'i':
		#id_str[1:]
                #self.id = self.id[int(id_str[1:])-1]
                self.id = id_str
	    elif singlevalue(id_str):
		self.id=int(id_str)
            else:
                self.id = array(id_str).astype(int)
            
        self.templates=get_str(sed_dir+self.spectra,0)
        
    def flux_comparison_plots(self,show_plots=1,save_plots=0,colors={},nomargins=0,outdir='',redo=False):
      verbose = self.verbose  
      if verbose:
          print 'Reading flux comparison data from %s' % self.flux_comparison
      #Get the flux comparison data
      #z = zeros(4, Float)
      #print less(z, 1)
      all=get_2Darray(self.flux_comparison) #Read the whole file
      #print 'less'
      #print less(all, 1)
      id=all[:,0] #ID column
        
      #Get the row which contains the ID number we are interested in
      if self.id == None:
	  self.id = id  # DEFAULT: DO 'EM ALL
      if type(self.id) == str:
        i_ids = [str2num(self.id[1:])-1]
        #i_ids = [int(self.id)]
      else:
          try:
              n = len(self.id)  # IF THIS FAILS, THEN IT'S NOT AN ARRAY / LIST
              i_ids = []
              for selfid in self.id:
                  i_id = findmatch1(id, selfid)
                  if i_id == -1:
                      print 'OBJECT #%d NOT FOUND.' % selfid
                      sys.exit()
                  i_ids.append(i_id)
          except:
                i_id = findmatch1(id, self.id)
                if i_id == -1:
                    print 'OBJECT NOT FOUND.'
                    sys.exit()
                else:
                    i_ids = [i_id]
      ncols=len(all[0,:])
      nf=(ncols-5)/3
      for i_id in i_ids:
        if nomargins:
            figure(1, figsize=(2,2))
            clf()
            axes([0,0,1,1])
        else:
            #thick(top=0.9)
            thick()
            figure(1)
            clf()
        ioff()
	print 'sed plot %d / %d : #%d' % (i_id+1, len(i_ids), id[i_id])
	if save_plots:
	    outimg = self.run_name+'_sed_%d.png' % id[i_id]
	    if exists(join(outdir, outimg)) and not redo:
		print join(outdir, outimg), 'ALREADY EXISTS'
		continue
        ft=all[i_id,5:5+nf]  # FLUX (from spectrum for that TYPE)
        fo=all[i_id,5+nf:5+2*nf]  # FLUX (OBSERVED)
        efo=all[i_id,5+2*nf:5+3*nf]  # FLUX_ERROR (OBSERVED)
	#efo = array(efo)
	#print type(efo)
	#observed = less(efo, 1)
	observed = efo < 1
	#print 'ft', ft
	#print 'fo', fo
	#print 'efo', efo
        prar = array([ft, fo, efo])
        if 0:
            print '         ft                fo              efo'
            print transpose(prar)

        #Get the redshift, type and magnitude of the galaxy
        m,z,t=all[i_id,1],all[i_id,2],all[i_id,3]
	params = params_cl()
	
        bpzdata = loaddata(self.bpzstr)
	if len(bpzdata.shape) == 1:
	    bpzdata.shape = (1, len(bpzdata))
        bpzdata = transpose(bpzdata)
        #odds = bpzdata[self.bpzcols.index('ODDS')]
        if 'ODDS' in self.bpzcols:
            i = self.bpzcols.index('ODDS')
        else:
            i = self.bpzcols.index('ODDS_1')
        odds = bpzdata[i]

	# Z-SPEC
	if 'Z_S' in self.bpzcols:
	    #bpzdata = transpose(loaddata(self.bpzstr))
	    zspeccol = self.bpzcols.index('Z_S')
	    zspec = bpzdata[zspeccol]
	    #zspec = ravel(loaddata(params['ZSPEC']))
	if 'ZSPEC' in params:
	    z = zspec[i_id]
	    print "z SET TO SPECTROSCOPIC VALUE OF %.3f" % z
	    ## 	z = 0.1972
## 	print "z ARTIFICIALLY SET TO ", z

 	#print "USING INTERP=2 (YOUR RESULTS MAY VARY)"
	if verbose:
            print "type=", t
	interp = self.interp
        if verbose:
            print "USING INTERP=%d" % interp
	t = (t + interp) / (1.*interp + 1)
        #print 'ARTIFICIALLY SETTING z & TYPE'
        # NOTE: THIS DOESN'T MOVE THE BLUE BOXES!
        #z, t = 3.14, 1
        #t, z = 5.33, 2.77
	#t = (t + 2) / 3.
	print "%.3f" % t,
	betweentypes = ndec(t)
## 	t = int(round(t))
## 	print t,
        sed=self.templates[int(t-1)]
	print sed,
	print "z=", z,
        #odds = [odds]
        print 'odds=%.2f' % odds[i_id]
        #print 'odds=%.2f' % odds
        if 'Z_S' in self.bpzcols:
	    print 'zspec=%.3f' % zspec[i_id]
	    
        #Get the filter wavelengths
        if verbose:
            print 'Reading filter information from %s' % self.columns
        filters=get_str(self.columns,0,nrows=nf,)
        lambda_m=ft*0.

        for i in range(len(filters)): lambda_m[i]=filter_center(filters[i])
	#print filters
	#print lambda_m
	
        # chisq 2
        eft = ft / 15.
        #eft = zeros(nf) + max(eft)
	#print eft
	#print max(eft)
	#print 'eft', eft
	eft = array([max(eft)] * nf)
	#print 'eft', eft
	#print 'efo', efo
        #eft = ones(nf) * max(eft)
        #eft = zeros(nf) + max(eft)
        #ef = sqrt(efo**2 + eft**2)
        #ef = hypot(eft, eft)
        #ef = hypot(efo, efo)
	#print type(efo), type(eft)
	#efo = array(efo.tolist())
	#eft = array(eft.tolist())
        ef = hypot(efo, eft)
        #
        dfosq = ((ft - fo) / ef) ** 2
        dfosqsum = sum(dfosq)
        #
        #observed = less(efo, 1)
        observed = efo < 1
	#print observed
	nfobs = 0
	for obs1 in observed:
	    if obs1:
		nfobs += 1
	#print nfobs
        #nfobs = sum(observed)
        #
        if nfobs > 1:
            dof = max([nfobs - 3, 1])  # 3 params (z, t, a)
            chisq2 = dfosqsum / dof
        elif nfobs: # == 1
            chisq2 = 999.
        else:
            chisq2 = 9999.

        print 'chisq2 = ', chisq2

        #Convert back to AB magnitudes
        efo1 = e_frac2mag(efo/fo)
        efo2 = flux2mag(efo)
        fo  = flux2mag(fo)
        ft  = flux2mag(ft)
        eft = p2p(ft) / 15. * ones(nf)

        seen = less(fo, 90)
        efo = where(seen, efo1, efo2)
        fo = where(seen, fo, 99)

	ptitle = "#%d, type=%.2f, bpz=%.2f, odds=%.2f" % (id[i_id], t, z, odds[i_id])
	if 'Z_S' in self.bpzcols:
	    ptitle += ", zspec=%.2f" % zspec[i_id]
		
	# -DC CORRECTING FLUX ERRORS:
	# mag = (-99, 0) NOT HANDLED CORRECLY BY BPZ WHEN OUTPUTTING TO .flux_comparison
	# ASSIGNS LARGE ERROR TO FLUX (DIVIDING BY ZERO MAG_ERROR)
	# MAYBE bpz SHOULD ASSIGN mag = (-99, maglimit) INSTEAD, LIKE IT DOES FOR mag = (99,
	#efo = where(less(efo, 1), efo, 0)
	#fomin = where(fo, fo-efo, 0)
	#fomax = where(fo, fo+efo, 2*efo)

        #print sed, z
        x,y=obs_spectrum_AB(sed,z)
        
        if nomargins:
            ax = gca()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            xlabel(r"Wavelength  $\lambda$ ($\AA$)")
            ylabel("Magnitude (AB)")
            #title(ptitle)

	if betweentypes:
	    # INTERPOLATE! --DC
	    t1 = int(t)
	    t2 = t1 + 1
	    sed2=self.templates[int(t2-1)]
	    x2,y2 = obs_spectrum_AB(sed2,z)
	    y2 = match_resol(x2,y2,x)
	    y = (t2 - t) * y + (t - t1) * y2

        #Normalize spectrum to model fluxes
        y_norm=match_resol(x,y,lambda_m)
        if sum(ft):
            #dm = mean(y_norm - ft) # Normalize magnitudes
            #print dm
            #pause()
            #y = y - mean(y_norm - ft) # Normalize magnitudes
            y_norm_seen = compress(seen, y_norm)
            ft_seen = compress(seen, ft)
            #
            # Check for model flux non-detections (may throw everything off!)
            dm1 = min(y_norm_seen)
            y_norm_seen = y_norm_seen - dm1
            seen2 = less(y_norm_seen, 90)
            y_norm_seen = compress(seen2, y_norm_seen)
            ft_seen = compress(seen2, ft_seen)
            #
            #print y_norm_seen
            #print ft_seen
            dm2 = mean(y_norm_seen - ft_seen) # Normalize magnitudes
            #print y_norm_seen - ft_seen
            #print dm2
            y = y - dm1 - dm2
            y_norm_seen = y_norm_seen - dm2
            #print y_norm_seen
            #
            # Re-Normalize to brightest:
            i = argmin(ft_seen)
            #print ft_seen[i]
            dm3 = y_norm_seen[i] - ft_seen[i]
            y = y - dm3
            y_norm_seen = y_norm_seen - dm3
            #print y_norm_seen
            #print ft_seen
        else:
            print 'OBSERVED FIT MINIZED & COMPRIMISED!!'
        
        black = 'grey60'

        xrange = prange(lambda_m, None, 0.075)
        xrange = prangelog(lambda_m, None, 0.075)
        inxrange = between(xrange[0], x, xrange[1])
        xinxrange = compress(inxrange, x)
        yinxrange = compress(inxrange, y)

        yyy2 = concatenate([fo, ft])
        #yyy  = concatenate([fo, ft, yinxrange, fo+efo, fo-efo])
        #yyy  = concatenate([fo, ft, fo+efo, fo-efo, fo-eft, fo+eft])
        #yyy  = concatenate([fo, ft, fo-eft, fo+eft])

        #fog = compress(observed, fo)
        #efog = compress(observed, efo)
        fog  = compress(observed*seen, fo)
        efog = compress(observed*seen, efo)
        ftg  = compress(observed*seen, ft)
        eftg = compress(observed*seen, eft)
        yyy  = concatenate([ftg-eftg, ftg+eftg, fog+efog, fog-efog])
        #yyy  = concatenate([fog, ft, fog+efog, fog-efog])
        #yyy  = concatenate([fog, ft, fog+efog+0.2, fog-efog-0.2])
        #print yyy, 'yyy'
        #print minmax(yyy)
        yrange = prange(yyy)
        #print yrange
        
        yyymax = min([max(yyy), 1.5*max(yyy2), 1.5*max(fo)]) # cap max, so you don't show big peaks and so you don't squash all efo

        # THIS SHOULD FORCE THE PLOT WINDOW TO WHAT I WANT
        # BUT IT DOESN'T QUITE WORK
	#plot([xrange[0]], [yrange[0]])
	#plot([xrange[1]], [yrange[1]])

        zorder = -1
        if nobox:
            zorder = 10
	plot(xinxrange.tolist(), yinxrange.tolist(), color='gray', zorder=zorder)
        linewidth = 1.5 + 2 * self.thick
        fontsize = 3 + 0.5 * self.thick
        
	maxy = max(y)

        indict = os.path.join(os.environ['BPZPATH'], 'plots/filtnicktex.dict')
        filtdict = loaddict(indict, silent=True)

        def customfiltcolor1(filt, lam):
            if lam < 4500:
                color = 'magenta'
            elif lam > 10000:
                color = 'red'
            elif filt[:3] == 'HST':
                color = 0, 0.8, 0
            elif filt[-4:] == 'LRIS':
                color = 'orange'
            else:
                color = 'gray50'
            #print color, lam, filt
            return color

        def customfiltcolor(filt, lam):
            color = 'red'
            for key in colors.keys():
                if string.find(filt, key) > -1:
                    color = colors[key]
                    color = str(color)  # for 0.50
                    if string.find(color, ',') > -1:
                        color = tuple(stringsplitatof(color, ','))
                    break
            return color

        #print '  filt     lambda     fo        efo         ft        chi'
        #print 'filt     lambda   fo      efo     ft      chi'
        if verbose:
            print ' filt      lambda    m      dm      mt      chi'
        
	for i in range(len(filters)):
            color = 'red'
            blue = 'blue'
            colorful = 'COLORFUL' in params
            if 1:
                color = customfiltcolor(filters[i], lambda_m[i])
                
            if filters[i] in filtdict.keys():
                filtnick = filtdict[filters[i]][1:-1]
            else:
                filtnick = filters[i]
                if len(filtnick) > 8:
                    filtnick = filtnick[:8]

            if verbose:
                print '%8s  %7.1f  %5.2f  %6.3f  %5.3f   %.3f' \
                    % (filtnick, lambda_m[i], fo[i], efo[i], ft[i], sqrt(dfosq[i]))
            ms = [7,4][nomargins]
	    if observed[i]:  # OBSERVED
                if max(eft) < yyymax*10:
		    #print 'max(eft) < yyymax*10'
                    rectwidth = .015 * (xrange[1] - xrange[0])
                    if 1:  # logx
                        rectwidthlog = .015 * (log10(xrange[1]) - log10(xrange[0]))
                        rectwidth = rectwidthlog * lambda_m[i] / log10(e)
                    if not nobox:  # blue rectangles (model fluxes)
                        rectangle([lambda_m[i]-rectwidth, ft[i]-eft[i]], [lambda_m[i]+rectwidth, ft[i]+eft[i]], color=blue, linewidth=linewidth)
                else:
		    print 'NOT max(eft) < yyymax*10'
                    plot([lambda_m[i]], [ft[i]], 'v', markersize=6, markerfacecolor='blue')
		if fo[i]:  # DETECTED
                    zorder = 1
                    if filters[i][:3] == 'HST':
                        zorder = 5
                    plot([lambda_m[i]], [fo[i]], 'o', mfc=color, mec=color, ms=12, zorder=zorder)
                    # Set plot limits (at least x) for ploterrorbars
                    xlim(xrange)
                    ylim(yrange[::-1])
                    if seen[i]:
                        ploterrorbars(array([lambda_m[i]]), array([fo[i]]), array([efo[i]]), color='k', xlog=plotlogx, lw=linewidth, zorder=zorder+1)
                    else:
                        ploterrorbars(array([lambda_m[i]]), array([99]), array([99-efo[i]]), color='k', xlog=plotlogx, lw=linewidth, zorder=zorder+1)
                    yl = min([fo[i], ft[i]-eft[i]*0.7])
		else:  # NOT DETECTED
                    if 0:
                        print 'NOT DETECTED'
                        print [lambda_m[i]], [fo[i]]
                        print [lambda_m[i], lambda_m[i]], [0, efo[i]]
                    plot([lambda_m[i]], [fo[i]], '^', markerfacecolor=color, markeredgecolor=color, markersize=ms)
		    plot([lambda_m[i], lambda_m[i]], [0., efo[i]], linewidth=linewidth, color=color)
                    yl = yyymax*0.04
	    else:  # NOT OBSERVED
		plot([lambda_m[i]], [0], 'o', markerfacecolor='w', markeredgecolor=color, markersize=ms)

        if self.thick:
            plotconfig(696)
            configure('fontsize_min', 2)
            configure('fontface', 'HersheySans-Bold')
            p.frame.spine_style['linewidth'] = 3
            p.frame.ticks_style['linewidth'] = 3
            p.frame.subticks_style['linewidth'] = 3

        if colorful:
            legend1('DuPont/NOT', 'magenta')
            legend1('ACS', (0,0.8,0))
            legend1('Keck', 'orange')
            legend1('NTT', 'red')

        if plotlogx:
            semilogx()
            retickx()

        xlim(xrange)
        ylim(yrange[::-1])

        if show_plots:
            if show_plots == True:
                print 'KILL PLOT WINOW TO TERMINATE OR CONTINUE.'
                show()
                print
            elif show_plots > 1:
                print 'Hit <Enter> to CONTINUE.'
                show()
                pause()
            show_plots += 1

	# ZSPEC thick bw show? save?:eps/png
	if save_plots:
	    print 'SAVING', join(outdir, outimg)
            if save_plots == 'png':
                savepng(join(outdir, decapfile(outimg)))
            elif save_plots == 'pdf':
                savepdf(join(outdir, decapfile(outimg)))
            else:
                savepngpdf(join(outdir, decapfile(outimg)))
	    clear()  # OR ELSE SUBSEQUENT PLOTS WILL PILE ON

            #print self.thick
            print

nobox = False
plotlogx = False

def run():
    global nobox, plotlogx
    id_str = None
    if len(sys.argv) > 2:
        if sys.argv[2][0] <> '-':
            id_str = sys.argv[2]
    params = params_cl()
    save_plots = 'SAVE' in params
    if save_plots:
        if params['SAVE']:
            save_plots = params['SAVE']

    show_plots = 1 - save_plots
    
    colors = {}
    if 'COLORS' in params:
        colors = loaddict(params['COLORS'])
    
    nomargins = 'NOMARGINS' in params
    nobox = 'NOBOX' in params
    plotlogx = 'LOGX' in params
    verbose = 'VERBOSE' in params
    
    b=bpzPlots(run_name=sys.argv[1], id_str=id_str, verbose=verbose)
    b.flux_comparison_plots(show_plots=show_plots, save_plots=save_plots, colors=colors, nomargins=nomargins)
    
if __name__ == '__main__':run()
else: pass
