"""
This is an example tomographic bin generator using a random forest.

Every classifier module needs to:
 - have construction of the type 
       __init__ (self, bands, options) (see examples below)
 -  implement two functions: 
        train (self, training_data,training_z)
        apply (self, data).
 - define valid_options class varible.

See Classifier Documentation below.
"""

import os 
from .base import Tomographer
import numpy as np
import rpy2
import rpy2.robjects as ro
import rpy2.robjects.packages as rpack
from rpy2.robjects.vectors import StrVector, IntVector, DataFrame, FloatVector, FactorVector
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from ..snc import SnCalc 
from scipy.optimize import minimize
from scipy.integrate import simps 
import pickle

#Check that all the needed packages are installed
# R package nameo
packnames = ('data.table','itertools','foreach','doParallel','RColorBrewer','devtools','matrixStats')
base=ro.packages.importr("base")
utils=ro.packages.importr("utils")
stats=ro.packages.importr("stats")
gr=ro.packages.importr("graphics")
dev=ro.packages.importr("grDevices")
utils.chooseCRANmirror(ind=1)
# Selectively install what needs to be installed.
names_to_install = [x for x in packnames if not rpack.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
base.Sys_setenv(TAR=base.system("which tar",intern=True))
devtools=ro.packages.importr("devtools")
#devtools.install_github("AngusWright/helpRfuncs")
#devtools.install_github("AngusWright/kohonen/kohonen")
kohonen=ro.packages.importr("kohonen")

class ComplexSOM(Tomographer):
    """ Complex SOM Classifier with SNR Optimisation """
    
    # valid parameter -- see below
    valid_options = ['bins','som_dim','num_groups','num_threads',
            'group_type','data_threshold','sparse_frac','plots',
            'plot_dir','metric','use_inbin','use_outbin']
    # this settings means arrays will be sent to train and apply instead
    # of dictionaries
    wants_arrays = False
    
    def __init__ (self, bands, options):
        """Constructor
        
        Parameters:
        -----------
        bands: str
          string containg valid bands, like 'riz' or 'griz'
        options: dict
          options come through here. Valid keys are listed as valid_options
          class variable. 

        Note:
        -----
        Valiad options are:
            'bins' - number of tomographic bins
            'som_dim' - dimensions of the SOM
            'num_groups' - number of SOM groups/associations
            'num_threads' - number of threads to use
            'sparse_frac' - the sparse-sampling fraction used for training
            'group_type' - do we want grouping by colour or redshift
            'data_threshold' - number of threads to use

        """
        self.bands = bands
        self.opt = options

    def train (self, training_data, training_z):
        """Trains the SOM and outputs the resulting bins
        
        Parameters:
        -----------
        training_data: numpy array, size Ngalaxes x Nbands
          training data, each row is a galaxy, each column is a band as per
          band defined above
        training_z: numpy array, size Ngalaxies
          true redshift for the training sample

        """

        print("Initialising")
        #Number of tomographic bins 
        n_bin = self.opt['bins']
        #Dimensions of the SOM
        som_dim = self.opt['som_dim'] 
        #Number of discrete SOM groups
        num_groups = self.opt['num_groups'] 
        #Number of threads 
        num_threads = self.opt['num_threads'] 
        #Sparse Frac 
        sparse_frac = self.opt['sparse_frac'] 
        #Data Threshold
        data_threshold = self.opt['data_threshold']
        #Metric
        metric = self.opt['metric']
        #Flag bad bins with inbin probability
        use_inbin = self.opt['use_inbin']
        #Flag bad bins with outbin probability
        use_outbin = self.opt['use_outbin']
        #Plots
        plots = self.opt['plots']
        #Plot Output Directory
        plot_dir = self.opt['plot_dir']
        #Group Type
        group_type = self.opt['group_type']
        #Check that the group type is a valid choice
        if group_type != 'colour' and group_type != 'redshift':
            group_type = 'redshift'
        #Define the assign_params variable, used in bin optimisation 
        assign_params = {'p_inbin_thr': 0.5,
                'p_outbin_thr': 0.2,
                'use_p_inbin': use_inbin,
                'use_p_outbin': use_outbin}

        #Define the redshift summary statistics (used for making groups in the 'redshift' case
        property_labels = ("mean_z_true","med_z_true","sd_z_true","mad_z_true","iqr_z_true")
        property_expressions = ("mean(data$redshift_true)","median(data$redshift_true)","sd(data$redshift_true)",
                                "mad(data$redshift_true)",
                                "diff(quantile(data$redshift_true,probs=pnorm(c(-2,2))))")
        #Define the SOM variables
        if self.bands == 'riz':
            #riz bands
            #expressions = ("r_mag-i_mag","r_mag-z_mag","i_mag-z_mag",
            #               "z_mag","r_mag-i_mag-(i_mag-z_mag)")
            expressions = ("r-i","r-z","i-z",
                           "z","r-i-(i-z)")
        elif self.bands == 'griz':
            #griz bands
            #expressions = ("g_mag-r_mag","g_mag-i_mag",
            #               "g_mag-z_mag","r_mag-i_mag","r_mag-z_mag","i_mag-z_mag",
            #               "z_mag","g_mag-r_mag-(r_mag-i_mag)",
            #               "r_mag-i_mag-(i_mag-z_mag)")
            expressions = ("g-r","g-i",
                           "g-z","r-i","r-z","i-z",
                           "z","g-r-(r-i)",
                           "r-i-(i-z)")
        elif self.bands == 'grizy':
            #grizy bands
            #expressions = ("g_mag-r_mag","g_mag-i_mag",
            #               "g_mag-z_mag","g_mag-y_mag","r_mag-i_mag","r_mag-z_mag","r_mag-y_mag","i_mag-z_mag","i_mag-y_mag",
            #               "z_mag-y_mag","z_mag","g_mag-r_mag-(r_mag-i_mag)",
            #               "r_mag-i_mag-(i_mag-z_mag)","i_mag-z_mag-(z_mag-y_mag)")
            expressions = ("g-r","g-i",
                           "g-z","g-y","r-i","r-z","r-y","i-z","i-y",
                           "z-y","z","g-r-(r-i)",
                           "r-i-(i-z)","i-z-(z-y)")
        elif self.bands == 'ugriz':
            #ugrizy bands
            #expressions = ("u_mag-g_mag","u_mag-r_mag","u_mag-i_mag","u_mag-z_mag","g_mag-r_mag","g_mag-i_mag",
            #               "g_mag-z_mag","r_mag-i_mag","r_mag-z_mag","i_mag-z_mag",
            #               "z_mag","u_mag-g_mag-(g_mag-r_mag)","g_mag-r_mag-(r_mag-i_mag)",
            #               "r_mag-i_mag-(i_mag-z_mag)")
            expressions = ("u-g","u-r","u-i","u-z","g-r","g-i",
                           "g-z","r-i","r-z","i-z",
                           "z","u-g-(g-r)","g-r-(r-i)",
                           "r-i-(i-z)")
        elif self.bands == 'ugrizy':
            #ugrizy bands
            #expressions = ("u_mag-g_mag","u_mag-r_mag","u_mag-i_mag","u_mag-z_mag","u_mag-y_mag","g_mag-r_mag","g_mag-i_mag",
            #               "g_mag-z_mag","g_mag-y_mag","r_mag-i_mag","r_mag-z_mag","r_mag-y_mag","i_mag-z_mag","i_mag-y_mag",
            #               "z_mag-y_mag","z_mag","u_mag-g_mag-(g_mag-r_mag)","g_mag-r_mag-(r_mag-i_mag)",
            #               "r_mag-i_mag-(i_mag-z_mag)","i_mag-z_mag-(z_mag-y_mag)")
            expressions = ("u-g","u-r","u-i","u-z","u-y","g-r","g-i",
                           "g-z","g-y","r-i","r-z","r-y","i-z","i-y",
                           "z-y","z","u-g-(g-r)","g-r-(r-i)",
                           "r-i-(i-z)","i-z-(z-y)")

        print("Preparing the data")
        training_data = pd.DataFrame.from_dict(training_data)
        #Add the redshift variable to the train data
        print("Adding redshift info to training data")
        training_data['redshift_true'] = training_z

        if sparse_frac < 1:
            print("Sparse Sampling the training data")
            cut = np.random.uniform(0, 1, training_z.size) < sparse_frac
            training_data = training_data[cut]
            training_z = training_z[cut]

        #Construct the training data frame (just a python-to-R data conversion)
        print("Converting the data to R format")
        with localconverter(ro.default_converter + pandas2ri.converter):
              #train_df = ro.conversion.py2rpy(train[['u_mag','g_mag','r_mag','i_mag','z_mag','y_mag']])
              train_df = ro.conversion.py2rpy(training_data)

        #Construct or Load the SOM 
        som_outname = f"SOM_{som_dim}_{self.bands}.pkl"
        if not os.path.exists(som_outname):
            print("Training the SOM using R kohtrain")
            #Train the SOM using R kohtrain
            som=kohonen.kohtrain(data=train_df,som_dim=IntVector(som_dim),max_na_frac=0,data_threshold=FloatVector(data_threshold),
                        n_cores=num_threads,train_expr=StrVector(expressions),train_sparse=False,sparse_frac=sparse_frac)
            #Output the SOM 
            #base.save(som,file=som_outname)
            with open(som_outname, 'wb') as f:
                pickle.dump(som, f)
        else:
            print("Loading the pretrained SOM")
            with open(som_outname, 'rb') as f:
                som = pickle.load(f)
            som.rx2['unit.classif']=FloatVector([])

        #If grouping by redshift, construct the cell redshift statistics
        if group_type == 'redshift' or plots == True:
            print("Constructing cell-based redshift properties")
            #Construct the Nz properties per SOM cell
            cell_prop=kohonen.generate_kohgroup_property(som=som,data=train_df,
                        expression=StrVector(property_expressions),expr_label=StrVector(property_labels))
            print("Constructing redshift-based hierarchical cluster tree")
            #Cluster the SOM cells into num_groups groups
            props = kohonen.kohwhiten(cell_prop.rx2['property'],train_expr=base.colnames(cell_prop.rx2['property']),
                    data_missing='NA',data_threshold=FloatVector([0,12]))
            props = props.rx2("data.white")
            props.rx[base.which(base.is_na(props))] = -1
            print(base.summary(props))
            hclust=stats.hclust(stats.dist(props))
            cell_group=stats.cutree(hclust,k=num_groups)

            #Assign the cell groups to the SOM structure
            som.rx2['hclust']=hclust
            som.rx2['cell_clust']=cell_group

        #Construct the Nz properties per SOM group
        print("Constructing group-based redshift properties")
        group_prop=kohonen.generate_kohgroup_property(som=som,data=train_df,
            expression=StrVector(property_expressions),expr_label=StrVector(property_labels),
            n_cluster_bins=num_groups)

        #extract the training som (just for convenience)
        train_som = group_prop.rx2('som')

        if plots == True: 
            #Make the diagnostic plots
            props = group_prop.rx2('property')
            cprops = cell_prop.rx2('property')
            print(base.colnames(props))
            print("Constructing SOM diagnostic figures")
            #Initialise the plot device
            dev.png(file=f'{plot_dir}/SOMfig_%02d.png',height=5,width=5,res=220,unit='in')
            #Paint the SOM by the training properties
            for i in range(len(expressions)): 
                gr.plot(train_som,property=i+1,shape='straight',heatkeywidth=som_dim[0]/20,ncolors=1e3,zlim=FloatVector([-3,3]))
            #Plot the standard diagnostics:
            #Count
            gr.plot(train_som,type='count',shape='straight',heatkeywidth=som_dim[0]/20,zlog=True,ncolors=1e3)
            #Quality
            gr.plot(train_som,type='quality',shape='straight',heatkeywidth=som_dim[0]/20,zlim=FloatVector([0,0.5]),ncolors=1e3)
            #UMatrix
            gr.plot(train_som,type='dist',shape='straight',heatkeywidth=som_dim[0]/20,zlog=True,zlim=FloatVector([0,0.5]),ncolors=1e3)
            #Sometimes these figures need customised limits; the "try" stops bad plots from stopping the code
             
            #Paint by the redshift diagnostics:
            #mean redshift
            gr.plot(train_som,property=cprops.rx(True,base.which(cprops.colnames.ro=='mean_z_true')),ncolors=1e3,zlog=False,
                    type='property',shape='straight',heatkeywidth=som_dim[0]/20,main='Cell mean redshift',zlim=FloatVector([0,1.8]))
            #redshift std
            gr.plot(train_som,property=cprops.rx(True,base.which(cprops.colnames.ro=='sd_z_true')),ncolors=1e3,zlog=False,
                    type='property',shape='straight',heatkeywidth=som_dim[0]/20,main='Cell redshift stdev',zlim=FloatVector([0,0.2]))
            #2sigma redshift IQR
            gr.plot(train_som,property=cprops.rx(True,base.which(cprops.colnames.ro=='iqr_z_true')),ncolors=1e3,zlog=False,
                    type='property',shape='straight',heatkeywidth=som_dim[0]/20,main='Cell 2sigma IQR',zlim=FloatVector([0,0.4]))
            #mean redshift
            gr.plot(train_som,property=props.rx(True,base.which(props.colnames.ro=='mean_z_true')),ncolors=1e3,zlog=False,
                    type='property',shape='straight',heatkeywidth=som_dim[0]/20,main='Group mean redshift',zlim=FloatVector([0,1.8]))
            #redshift std
            gr.plot(train_som,property=props.rx(True,base.which(props.colnames.ro=='sd_z_true')),ncolors=1e3,zlog=False,
                    type='property',shape='straight',heatkeywidth=som_dim[0]/20,main='Group redshift stdev',zlim=FloatVector([0,0.2]))
            #2sigma redshift IQR
            gr.plot(train_som,property=props.rx(True,base.which(props.colnames.ro=='iqr_z_true')),ncolors=1e3,zlog=False,
                    type='property',shape='straight',heatkeywidth=som_dim[0]/20,main='Group 2sigma IQR',zlim=FloatVector([0,0.4]))
            ##N group
            #gr.plot(train_som,property=props.rx(True,base.which(props.colnames.ro=='N')),ncolors=1e3,zlog=False,
            #        type='property',shape='straight',heatkeywidth=som_dim[0]/20,main='N group')

            #Close the plot device 
            dev.dev_off()

        print("Constructing the per-group Nz")
        #Get the group assignments for the training data 
        train_group = np.array(train_som.rx2['clust.classif'])
        tab=base.table(FactorVector(train_som.rx2['clust.classif'],levels=base.seq(num_groups)))
        print(tab)
        print(np.arange(num_groups)[np.array(tab)==0])
        #Construct the Nz z-axis
        z_arr = np.linspace(0, 2, 124)
        z_cen = z_arr[0:-1]+(z_arr[1]-z_arr[0])/2. 
        #Construct the per-group Nz
        nzs = [(np.histogram(training_z[train_group == group+1],z_arr)[0]) for group in np.arange(num_groups)[np.array(tab)!=0]]
       
        #np.save('plots/nzs.npy', nzs)
       
        #Update the fsky
        # Renormalize to use the same sample definitions as the
        # official metric calculators.
        fsky = 0.25
        ndens_arcmin = 20.
        area_arcmin = (180*60/np.pi)**2*4*np.pi*fsky
        ng_tot_goal = ndens_arcmin * area_arcmin
        ng_tot_curr = np.sum(np.array(tab))
        nzs = [nz * ng_tot_goal / ng_tot_curr for nz in nzs]

        #np.save('plots/nzs_norm.npy', nzs)
       
        #print(nzs)
    
        # Now initialize a S/N calculator for these initial groups.
        os.system('rm wl_nb%d.npz' % n_bin)
        if metric == 'SNR_ww': 
            c_wl = SnCalc(z_cen, nzs, use_clustering=False,fsky=fsky)
        else:
            c_wl = SnCalc(z_cen, nzs, use_clustering=True,fsky=fsky)
        print("Initializing WL")
        c_wl.get_cl_matrix(fname_save='wl_nb%d.npz' % n_bin)
        # Finally, let's write the function to minimize and optimize for a 4-bin case.
        def minus_sn(edges, calc):
            return -calc.get_sn_from_edges(edges,assign_params=assign_params)

        print("Optimizing WL")
        edges_0 = np.linspace(0, 2, n_bin-1)
        res = minimize(minus_sn, edges_0, method='Powell', args=(c_wl,))
        print("WL final edges: ", res.x)
        print("Maximum S/N: ", c_wl.get_sn_from_edges(res.x)*np.sqrt(0.25/fsky),assign_params=assign_params)
        print(" ")
        print(res)

        #Construct the per-group Nz
        print("Outputting trained SOM and redshift ordering of SOM groups")
        groups_in_tomo = c_wl.assign_from_edges(res.x, get_ids=True)
        group_bins = np.zeros(num_groups)
        for bin_no, groups in groups_in_tomo:
            print(bin_no, groups)
            group_bins[groups] = bin_no

        print("Finished")
        self.train_som = train_som
        self.group_bins = group_bins

    def apply (self, data):
        """Applies training to the data.
        
        Parameters:
        -----------
        Data: numpy array, size Ngalaxes x Nbands
          testing data, each row is a galaxy, each column is a band as per
          band defined above

        Returns: 
        tomographic_selections: numpy array, int, size Ngalaxies
          tomographic selection for galaxies return as bin number for 
          each galaxy.
        """
        
        #Number of tomographic bins 
        n_bin = self.opt['bins']
        #Number of discrete SOM groups
        num_groups = self.opt['num_groups'] 

        print("Preparing the data")
        data = pd.DataFrame.from_dict(data)

        #Construct the validation data frame (just a python-to-R data conversion)
        print("Converting the data to R format")
        with localconverter(ro.default_converter + pandas2ri.converter):
              data_df = ro.conversion.py2rpy(data)

        print("Parsing the validation data into the SOM groupings")
        #Generate the validation associations/groups
        group_prop=kohonen.generate_kohgroup_property(som=self.train_som,data=data_df,
            expression="nrow(data)",expr_label="N",
            n_cluster_bins=num_groups,n_cores=self.opt['num_threads'])

        #extract the validation som (just for convenience)
        valid_som = group_prop.rx2('som')

        #Assign the sources, by group, to tomographic bins
        print("Output source tomographic bin assignments")
        valid_bin = base.unsplit(IntVector(self.group_bins),FactorVector(valid_som.rx2['clust.classif'],
            levels=base.seq(num_groups)),drop=False)
        valid_bin = np.array(valid_bin)

        return valid_bin

