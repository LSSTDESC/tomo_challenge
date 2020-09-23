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

from .base import Tomographer
import numpy as np
import rpy2
import rpy2.robjects as ro
import rpy2.robjects.packages as rpack
from rpy2.robjects.vectors import StrVector, IntVector, DataFrame, FloatVector, FactorVector
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


#Check that all the needed packages are installed
# R package nameo
base = None
stats = None
gr = None
dev = None
kohonen = None

def init_r_packages():
    global base, stats, gr, dev, kohonen
    base=ro.packages.importr("base")
    stats=ro.packages.importr("stats")
    gr=ro.packages.importr("graphics")
    dev=ro.packages.importr("grDevices")
    base.Sys_setenv(TAR=base.system("which tar",intern=True))
    kohonen=ro.packages.importr("kohonen")

# #Check that all the needed packages are installed
# # R package nameo
# packnames = ('data.table','itertools','foreach','doParallel','RColorBrewer','devtools','matrixStats')
# base=ro.packages.importr("base")
# utils=ro.packages.importr("utils")
# stats=ro.packages.importr("stats")
# gr=ro.packages.importr("graphics")
# dev=ro.packages.importr("grDevices")
# utils.chooseCRANmirror(ind=1)
# # Selectively install what needs to be installed.
# names_to_install = [x for x in packnames if not rpack.isinstalled(x)]
# if len(names_to_install) > 0:
#     utils.install_packages(StrVector(names_to_install))
# base.Sys_setenv(TAR=base.system("which tar",intern=True))
# devtools=ro.packages.importr("devtools")
# #devtools.install_github("AngusWright/helpRfuncs")
# #devtools.install_github("AngusWright/kohonen/kohonen")
# kohonen=ro.packages.importr("kohonen")

class SimpleSOM1(Tomographer):
    """ Simplistic SOM Classifier """
    
    # valid parameter -- see below
    valid_options = ['bins','som_dim','num_groups','num_threads',
            'group_type','data_threshold','sparse_frac','plots','plot_dir']
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
        init_r_packages()
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
        #Plots
        plots = self.opt['plots']
        #Plot Output Directory
        plot_dir = self.opt['plot_dir']
        #Group Type
        group_type = self.opt['group_type']
        #Check that the group type is a valid choice
        if group_type != 'colour' and group_type != 'redshift':
            group_type = 'redshift'

        #Define the redshift summary statistics (used for making groups in the 'redshift' case
        property_labels = ("mean_z_true","med_z_true","sd_z_true","mad_z_true","N","iqr_z_true")
        property_expressions = ("mean(data$redshift_true)","median(data$redshift_true)","sd(data$redshift_true)",
                                "mad(data$redshift_true)","nrow(data)",
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

        #Construct the training data frame (just a python-to-R data conversion)
        print("Converting the data to R format")
        with localconverter(ro.default_converter + pandas2ri.converter):
              #train_df = ro.conversion.py2rpy(train[['u_mag','g_mag','r_mag','i_mag','z_mag','y_mag']])
              train_df = ro.conversion.py2rpy(training_data)

        print("Training the SOM using R kohtrain")
        #Train the SOM using R kohtrain
        som=kohonen.kohtrain(data=train_df,som_dim=IntVector(som_dim),max_na_frac=0,data_threshold=FloatVector(data_threshold),
                    n_cores=num_threads,train_expr=StrVector(expressions),train_sparse=False,sparse_frac=sparse_frac)

        #If grouping by redshift, construct the cell redshift statistics
        if group_type == 'redshift' or plots == True:
            print("Constructing cell-based redshift properties")
            #Construct the Nz properties per SOM cell
            cell_prop=kohonen.generate_kohgroup_property(som=som,data=train_df,
                        expression=StrVector(property_expressions),expr_label=StrVector(property_labels))
            print("Constructing redshift-based hierarchical cluster tree")
            #Cluster the SOM cells into num_groups groups
            hclust=stats.hclust(stats.dist(cell_prop.rx2['property']))
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
            gr.plot(train_som,property=props.rx(True,base.which(props.colnames.ro=='mean_z_true')),ncolors=1e3,zlog=False,
                    type='property',shape='straight',heatkeywidth=som_dim[0]/20,main='Group mean redshift',zlim=FloatVector([0,1.8]))
            #redshift std
            gr.plot(train_som,property=props.rx(True,base.which(props.colnames.ro=='sd_z_true')),ncolors=1e3,zlog=False,
                    type='property',shape='straight',heatkeywidth=som_dim[0]/20,main='Group redshift stdev',zlim=FloatVector([0,0.2]))
            #2sigma redshift IQR
            gr.plot(train_som,property=props.rx(True,base.which(props.colnames.ro=='iqr_z_true')),ncolors=1e3,zlog=False,
                    type='property',shape='straight',heatkeywidth=som_dim[0]/20,main='group 2sigma IQR',zlim=FloatVector([0,0.4]))
            #N group
            gr.plot(train_som,property=props.rx(True,base.which(props.colnames.ro=='N')),ncolors=1e3,zlog=False,
                    type='property',shape='straight',heatkeywidth=som_dim[0]/20,main='N group')

            #Close the plot device 
            dev.dev_off()



        print("Outputting trained SOM and redshift ordering of SOM groups")
        #Extract the mean-z per group
        group_z = group_prop.rx2('property').rx(True,
                base.which(group_prop.rx2('property').colnames.ro=='mean_z_true'))
        #Order the groups by mean z
        z_order = base.order(group_z)

        print("Finished")
        self.train_som = train_som
        self.group_z = group_z
        self.z_order = z_order

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

        print("Preparing the data")
        data = pd.DataFrame.from_dict(data)

        #Construct the validation data frame (just a python-to-R data conversion)
        print("Converting the data to R format")
        with localconverter(ro.default_converter + pandas2ri.converter):
              #train_df = ro.conversion.py2rpy(train[['u_mag','g_mag','r_mag','i_mag','z_mag','y_mag']])
              data_df = ro.conversion.py2rpy(data)

        print("Parsing the validation data into the SOM groupings")
        #Generate the validation associations/groups
        group_prop=kohonen.generate_kohgroup_property(som=self.train_som,data=data_df,
            expression="nrow(data)",expr_label="N",
            n_cluster_bins=self.opt['num_groups'],n_cores=self.opt['num_threads'])

        #Calculate the cumulative count 
        print("Generate cumulative source counts a.f.o. group mean z")
        zcumsum=base.cumsum(group_prop.rx2('property').rx(self.z_order))

        # Find the edges that split the redshifts into n_z bins of
        # equal number counts in each
        print("Assign the groups to tomographic bins")
        p = np.linspace(0, 100, n_bin + 1)
        n_edges = np.percentile(zcumsum, p)

        # Now put the groups into redshift bins.
        group_bins = FloatVector(base.cut(FloatVector(zcumsum),FloatVector(n_edges),
            include=True)).rx(base.order(self.z_order))

        #extract the validation som (just for convenience)
        valid_som = group_prop.rx2('som')

        #Assign the sources, by group, to tomographic bins
        print("Output source tomographic bin assignments")
        valid_bin = base.unsplit(group_bins,FactorVector(valid_som.rx2['clust.classif'],
            levels=base.seq(self.opt['num_groups'])),drop=False)
        #valid_bin = valid_som.rx2['clust.classif']
        valid_bin = np.array(valid_bin)-1

        return valid_bin

