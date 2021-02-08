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
import sys
import os
import numpy as np
import rpy2
import rpy2.robjects as ro
import rpy2.robjects.packages as rpack
from rpy2.robjects.vectors import StrVector, IntVector, DataFrame, FloatVector, FactorVector, BoolVector
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pickle

#Check that all the needed packages are installed
# R package nameo
# packnames = ('data.table','itertools','foreach','doParallel','RColorBrewer','devtools','matrixStats')
# base=ro.packages.importr("base")
# utils=ro.packages.importr("utils")
# stats=ro.packages.importr("stats")
# gr=ro.packages.importr("graphics")
# dev=ro.packages.importr("grDevices")
# utils.chooseCRANmirror(ind=1)
# Selectively install what needs to be installed.
# names_to_install = [x for x in packnames if not rpack.isinstalled(x)]
# if len(names_to_install) > 0:
    # utils.install_packages(StrVector(names_to_install))
# base.Sys_setenv(TAR=base.system("which tar",intern=True))
# devtools=ro.packages.importr("devtools")
#devtools.install_github("AngusWright/helpRfuncs")
#devtools.install_github("AngusWright/kohonen/kohonen")
# kohonen=ro.packages.importr("kohonen")

#Check that all the needed packages are installed
# R package nameo
base = None
stats = None
gr = None
dev = None
kohonen = None

task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "666")

run_file = f"bpz_run_{task_id}.py"
training_cat_file = f"training_bpz_{task_id}.cat"
training_col_file = f"training_bpz_{task_id}.columns"
bpz_file = f"training_bpz_{task_id}.bpz"
ascii_file = f"training_bpz_{task_id}.asc"
#validation_cat_file = "validato" # not used
validation_col_file = f"validation_bpz_{task_id}.columns"

def init_r_packages():
    global base, stats, gr, dev, kohonen
    base=ro.packages.importr("base")
    stats=ro.packages.importr("stats")
    gr=ro.packages.importr("graphics")
    dev=ro.packages.importr("grDevices")
    base.Sys_setenv(TAR=base.system("which tar",intern=True))
    kohonen=ro.packages.importr("kohonen")

def remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

class PQNLD(Tomographer):
    """ Combined Template and SOM Classifier """
    
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
        remove(ascii_file)
        remove(bpz_file)
        remove(training_col_file)
        remove(training_cat_file)
        remove(validation_col_file)
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
        ZB_property_labels = ("mean_ZB","mad_ZB")
        ZB_property_expressions = ("mean(data$Z_B)",
                                "mad(data$Z_B)")
        property_labels = ("mean_z_true","med_z_true","mad_z_true")
        property_expressions = ("mean(data$redshift_true)",
                                "median(data$redshift_true)",
                                "mad(data$redshift_true)")
        #Define the SOM variables
        if self.bands == 'riz':
            #riz bands
            expressions = ("r-i","r-z","i-z",
                           "z","r-i-(i-z)")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(np.transpose(np.array([
                                   ["r_SDSS","i_SDSS","z_SDSS"],
                                   ["1,2","3,4","5,6"],
                                   ["AB","AB","AB"],
                                   ["0.01","0.01","0.01"],
                                   ["0.00","0.00","0.00"]])))
            #Output the columns file 
            columns.to_csv(training_col_file,sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(np.transpose(
                                  np.array([["M_0","Z_S"],
                                            ["5","7"]])))
            #Output the columns file 
            columns.to_csv(training_col_file,sep=' ',index=None,header=None,mode='a')
        elif self.bands == 'griz':
            #griz bands
            expressions = ("g-r","g-i",
                           "g-z","r-i","r-z","i-z",
                           "z","g-r-(r-i)",
                           "r-i-(i-z)")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(np.transpose(np.array([["g_SDSS","r_SDSS","i_SDSS","z_SDSS"],
                                   ["1,2","3,4","5,6","7,8"],
                                   ["AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01"],
                                   ["0.000001","0.000001","0.000001","0.000001"]])))
            #Output the columns file 
            columns.to_csv(training_col_file,sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(["M_0","Z_S"],
                                   ["7","9"])
            #Output the columns file 
            columns.to_csv(training_col_file,sep=' ',index=None,header=None,mode='a')
        elif self.bands == 'grizy':
            #grizy bands
            expressions = ("g-r","g-i",
                           "g-z","g-y","r-i","r-z","r-y","i-z","i-y",
                           "z-y","z","g-r-(r-i)",
                           "r-i-(i-z)","i-z-(z-y)")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(np.transpose(np.array([["g_SDSS","r_SDSS","i_SDSS","z_SDSS","y_SDSS"],
                                   ["1,2","3,4","5,6","7,8","9,10"],
                                   ["AB","AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01","0.01"],
                                   ["0.000001","0.000001","0.000001","0.000001","0.000001"]])))
            #Output the columns file 
            columns.to_csv(training_col_file,sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(["M_0","Z_S"],
                                   ["7","11"])
            #Output the columns file 
            columns.to_csv(training_col_file,sep=' ',index=None,header=None,mode='a')
        elif self.bands == 'ugriz':
            #ugrizy bands
            expressions = ("u-g","u-r","u-i","u-z","g-r","g-i",
                           "g-z","r-i","r-z","i-z",
                           "z","u-g-(g-r)","g-r-(r-i)",
                           "r-i-(i-z)")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(np.transpose(np.array([["u_SDSS","g_SDSS","r_SDSS","i_SDSS","z_SDSS"],
                                   ["1,2","3,4","5,6","7,8","9,10"],
                                   ["AB","AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01","0.01"],
                                   ["0.000001","0.000001","0.000001","0.000001","0.000001"]])))
            #Output the columns file 
            columns.to_csv(training_col_file,sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(["M_0","Z_S"],
                                   ["9","11"])
            #Output the columns file 
            columns.to_csv(training_col_file,sep=' ',index=None,header=None,mode='a')
        elif self.bands == 'ugrizy':
            #ugrizy bands
            expressions = ("u-g","u-r","u-i","u-z","u-y","g-r","g-i",
                           "g-z","g-y","r-i","r-z","r-y","i-z","i-y",
                           "z-y","z","u-g-(g-r)","g-r-(r-i)",
                           "r-i-(i-z)","i-z-(z-y)")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(np.transpose(np.array([["u_SDSS","g_SDSS","r_SDSS","i_SDSS","z_SDSS","y_SDSS"],
                                   ["1,2","3,4","5,6","7,8","9,10","11,12"],
                                   ["AB","AB","AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01","0.01","0.01"],
                                   ["0.000001","0.000001","0.000001","0.000001","0.000001","0.000001"]])))
            #Output the columns file 
            columns.to_csv(training_col_file,sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(["M_0","Z_S"],
                                   ["9","13"])
            #Output the columns file 
            columns.to_csv(training_col_file,sep=' ',index=None,header=None,mode='a')

        print("Preparing the data for BPZ")
        #Output the BPZ input catalogue 
        training_data = pd.DataFrame.from_dict(training_data)
        print("Adding redshift info to training data")
        training_data['redshift_true'] = training_z
        if os.path.exists(training_cat_file):
            print("Read the BPZ inputs")
            cols = training_data.columns
            training_data = pd.read_csv(training_cat_file,sep=' ',header=None)
            training_data.columns = cols
            training_z = training_data[['redshift_true']]
        else:
            if sparse_frac < 1:
                print("Sparse Sampling the training data")
                cut = np.random.uniform(0, 1, training_z.size) < sparse_frac
                training_data = training_data[cut]

            print("Outputting the BPZ input cat")
            np.savetxt(training_cat_file,training_data,fmt='%3.5f')

        if not os.path.exists("bpz-1.99.3-py3/"): 
            os.system("bash INSTALL.sh")
        
        if not os.path.exists(bpz_file): 
            print("Running BPZ on the training data")
            curdir=os.getcwd()
            os.chdir('bpz-1.99.3-py3')
            os.system("echo '#NEW' > " + run_file)
            os.system("echo import sys >> " + run_file)
            os.system("echo import os >> " + run_file)
            os.system("echo 'os.environ[\"HOME\"]=\""+curdir + "\"' >> " + run_file)
            os.system("echo 'os.environ[\"BPZPATH\"]=\""+curdir+"/bpz-1.99.3-py3/\"' >> " + run_file)
            os.system("echo 'os.environ[\"NUMERIX\"]=\"numpy\"' >> " + run_file)
            argv = ["\'bpz.py\'", f"\'../{training_cat_file}\'",
                "-PRIOR", "NGVS", "-SPECTRA", "CWWSB_capak.list",
                "-ZMIN", "0.001", "-ZMAX", "7.000",
                "-INTERP", "10", "-NEW_AB", "yes",
                "-ODDS", "0.68", "-MIN_RMS", "0.0", "-INTERACTIVE", "no",
                "-PROBS_LITE", "no",
                "-VERBOSE", "yes",
                "-CHECK", "no"]
            os.system("echo 'sys.argv=[\""+"\",\"".join(argv)+"\"]' >> " + run_file)
            # os.system("echo 'exec(open(\"bpz.py\").read())' >> bpz_run.py")
            os.system("echo 'import bpz' >> " + run_file)
            # os.system("cp ../BPZ_SEDs/prior_NGVS.py .")
            # os.system("cp ../BPZ_SEDs/*.sed SED/")
            # os.system("cp ../BPZ_SEDs/CWWSB_capak.list SED/")
            os.system('cat ' + run_file)
            res = os.system("python " + run_file)
            os.chdir('../')

        print("Read the BPZ results")
        #bpz_res = pd.read_fwf("training_bpz.bpz",comment="#",skiprows=62,header=None)
        os.system("grep -v '^#' " + bpz_file + " | sed 's/  / /g' | sed 's/  / /g' | \
                sed 's/  / /g' | sed 's/^ //g' | sed 's/ $//g' > " + ascii_file)
        bpz_res = pd.read_csv(ascii_file,sep=' ',header=None)

        print("Adding BPZ Photoz info to training data")
        training_data['Z_B'] = bpz_res[[1]]

        #Construct the training data frame (just a python-to-R data conversion)
        print("Converting the training data to R format")
        with localconverter(ro.default_converter + pandas2ri.converter):
              train_df = ro.conversion.py2rpy(training_data)

        #Construct or Load the SOM 
        # som_outname = f"SOM_{som_dim}_{self.bands}.pkl"
        # if True: #not os.path.exists(som_outname):
        print("Training the SOM using R kohtrain")
        #Train the SOM using R kohtrain
        som=kohonen.kohtrain(data=train_df,som_dim=IntVector(som_dim),max_na_frac=0,data_threshold=FloatVector(data_threshold),
                    n_cores=num_threads,train_expr=StrVector(expressions),train_sparse=False,sparse_frac=sparse_frac)
        #Output the SOM 
        #base.save(som,file=som_outname)
        # with open(som_outname, 'wb') as f:
        #     pickle.dump(som, f)
        # else:
        #     print("Loading the pretrained SOM")
        #     with open(som_outname, 'rb') as f:
        #         som = pickle.load(f)
        #     som.rx2['unit.classif']=FloatVector([])

        #If grouping by redshift, construct the cell redshift statistics
        print("Constructing cell-based redshift properties")
        #Construct the Nz properties per SOM cell
        cell_prop=kohonen.generate_kohgroup_property(som=som,data=train_df,
                    expression=StrVector(property_expressions),expr_label=StrVector(property_labels),returnMatrix=True)
        som = cell_prop.rx2('som')
        cell_prop_ZB=kohonen.generate_kohgroup_property(som=som,data=train_df,
                    expression=StrVector(ZB_property_expressions),expr_label=StrVector(ZB_property_labels),returnMatrix=True)
        cell_prop_N=kohonen.generate_kohgroup_property(som=som,data=train_df,
                    expression="nrow(data)",expr_label="N",returnMatrix=True)
        print("Generate cumulative source counts a.f.o. cell mean z")
        #Extract the mean-z per group
        cell_z = cell_prop.rx2('property').rx(True,
                base.which(cell_prop.rx2('property').colnames.ro=='mean_z_true'))
        #Order the groups by mean z
        z_order = base.order(cell_z)
        zcumsum=base.cumsum(cell_prop_N.rx2('property').rx(z_order))
        print(base.summary(cell_prop_N.rx2('property')))
        print(base.summary(z_order))
        print("Flagging problematic cells in at most 20 equal N bins of redshift")
        p = np.linspace(0, 100, 21)
        n_edges = np.percentile(zcumsum, p)

        print(n_edges)
        # Now put the groups into redshift bins.
        cell_bins = FloatVector(base.cut(FloatVector(zcumsum),base.unique(FloatVector(n_edges)),
            include=True)).rx(base.order(z_order))
        source_bin = base.unsplit(cell_bins,FactorVector(som.rx2['unit.classif'],
            levels=base.seq(som_dim[0]*som_dim[1])),drop=False)
        badcell = BoolVector(base.rep(False,som_dim[0]*som_dim[1]))
        for zbin in range(n_bin+1):
            zs = train_df.rx2['redshift_true']
            zs = zs.rx(base.which(source_bin.ro==zbin))
            zb = train_df.rx2['Z_B']
            zb = zb.rx(base.which(source_bin.ro==zbin))
            bias = zs.ro - zb
            meanbias = base.mean(bias)   
            madbias = stats.mad(bias)
            zs = cell_prop.rx2('property').rx(True,
                base.which(cell_prop.rx2('property').colnames.ro=='mean_z_true'))
            zb = cell_prop.rx2('property').rx(True,
                base.which(cell_prop_ZB.rx2('property').colnames.ro=='mean_ZB'))
            cellbias = zs.ro - zb
            cellstat = base.abs(cellbias.ro - meanbias)
            cellstat = cellstat.ro/madbias
            celllog = BoolVector(cellstat.ro>=5.)
            badcell.rx[celllog] = True

        print("Constructing redshift-based hierarchical cluster tree")
        #Cluster the SOM cells into num_groups groups
        props = cell_prop.rx2('property')
        props.rx[base.which(base.is_na(props))] = -1
        props.rx[badcell,True] = -1 
        zs = cell_prop.rx2('property').rx(True,
            base.which(cell_prop.rx2('property').colnames.ro=='mean_z_true'))
        if group_type == 'redshift':
            hclust=stats.hclust(stats.dist(props))
            cell_group=stats.cutree(hclust,k=num_groups)
            #Assign the cell groups to the SOM structure
            som.rx2['hclust']=hclust
            som.rx2['cell_clust']=cell_group

        #Construct the Nz properties per SOM group
        print("Constructing group-based redshift properties")
        group_prop=kohonen.generate_kohgroup_property(som=som,data=train_df,
            expression=StrVector(property_expressions),expr_label=StrVector(property_labels),
            n_cluster_bins=num_groups,returnMatrix=True)

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
            gr.plot(train_som,property=props.rx(True,base.which(props.colnames.ro=='mad_z_true')),ncolors=1e3,zlog=False,
                    type='property',shape='straight',heatkeywidth=som_dim[0]/20,main='Group redshift stdev',zlim=FloatVector([0,0.2]))
            ##2sigma redshift IQR
            #gr.plot(train_som,property=props.rx(True,base.which(props.colnames.ro=='iqr_z_true')),ncolors=1e3,zlog=False,
            #        type='property',shape='straight',heatkeywidth=som_dim[0]/20,main='group 2sigma IQR',zlim=FloatVector([0,0.4]))
            ##N group
            #gr.plot(train_som,property=props.rx(True,base.which(props.colnames.ro=='N')),ncolors=1e3,zlog=False,
            #        type='property',shape='straight',heatkeywidth=som_dim[0]/20,main='N group')

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
        
        #Define the SOM variables
        if self.bands == 'riz':
            #riz bands
            expressions = ("r-i","r-z","i-z",
                           "z","r-i-(i-z)")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(np.transpose(np.array([
                                   ["r_SDSS","i_SDSS","z_SDSS"],
                                   ["1,2","3,4","5,6"],
                                   ["AB","AB","AB"],
                                   ["0.01","0.01","0.01"],
                                   ["0.00","0.00","0.00"]])))
            #Output the columns file 
            columns.to_csv(validation_col_file,sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(np.transpose(
                                  np.array([["M_0"],
                                            ["5"]])))
            #Output the columns file 
            columns.to_csv(validation_col_file,sep=' ',index=None,header=None,mode='a')
        elif self.bands == 'griz':
            #griz bands
            expressions = ("g-r","g-i",
                           "g-z","r-i","r-z","i-z",
                           "z","g-r-(r-i)",
                           "r-i-(i-z)")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(["g_SDSS","r_SDSS","i_SDSS","z_SDSS"],
                                   ["1,2","3,4","5,6","7,8"],
                                   ["AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01"],
                                   ["0.000001","0.000001","0.000001","0.000001"])
            #Output the columns file 
            columns.to_csv(validation_col_file,sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(np.transpose(
                                  np.array([["M_0"],
                                            ["7"]])))
            #Output the columns file 
            columns.to_csv(validation_col_file,sep=' ',index=None,header=None,mode='a')
        elif self.bands == 'grizy':
            #grizy bands
            expressions = ("g-r","g-i",
                           "g-z","g-y","r-i","r-z","r-y","i-z","i-y",
                           "z-y","z","g-r-(r-i)",
                           "r-i-(i-z)","i-z-(z-y)")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(["g_SDSS","r_SDSS","i_SDSS","z_SDSS","y_SDSS"],
                                   ["1,2","3,4","5,6","7,8","9,10"],
                                   ["AB","AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01","0.01"],
                                   ["0.000001","0.000001","0.000001","0.000001","0.000001"])
            #Output the columns file 
            columns.to_csv(validation_col_file,sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(np.transpose(
                                  np.array([["M_0"],
                                            ["7"]])))
            #Output the columns file 
            columns.to_csv(validation_col_file,sep=' ',index=None,header=None,mode='a')
        elif self.bands == 'ugriz':
            #ugrizy bands
            expressions = ("u-g","u-r","u-i","u-z","g-r","g-i",
                           "g-z","r-i","r-z","i-z",
                           "z","u-g-(g-r)","g-r-(r-i)",
                           "r-i-(i-z)")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(["u_SDSS","g_SDSS","r_SDSS","i_SDSS","z_SDSS"],
                                   ["1,2","3,4","5,6","7,8","9,10"],
                                   ["AB","AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01","0.01"],
                                   ["0.000001","0.000001","0.000001","0.000001","0.000001"])
            #Output the columns file 
            columns.to_csv(validation_col_file,sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(np.transpose(
                                  np.array([["M_0"],
                                            ["9"]])))
            #Output the columns file 
            columns.to_csv(validation_col_file,sep=' ',index=None,header=None,mode='a')
        elif self.bands == 'ugrizy':
            #ugrizy bands
            expressions = ("u-g","u-r","u-i","u-z","u-y","g-r","g-i",
                           "g-z","g-y","r-i","r-z","r-y","i-z","i-y",
                           "z-y","z","u-g-(g-r)","g-r-(r-i)",
                           "r-i-(i-z)","i-z-(z-y)")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(["u_SDSS","g_SDSS","r_SDSS","i_SDSS","z_SDSS","y_SDSS"],
                                   ["1,2","3,4","5,6","7,8","9,10","11,12"],
                                   ["AB","AB","AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01","0.01","0.01"],
                                   ["0.000001","0.000001","0.000001","0.000001","0.000001","0.000001"])
            #Output the columns file 
            columns.to_csv(validation_col_file,sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(np.transpose(
                                  np.array([["M_0"],
                                            ["9"]])))
            #Output the columns file 
            columns.to_csv(validation_col_file,sep=' ',index=None,header=None,mode='a')

            
        #Number of tomographic bins 
        n_bin = self.opt['bins']

        print("Preparing the data")
        data = pd.DataFrame.from_dict(data)
        print(data.shape)

        #print("Adding redshift info to training data")
        #if os.path.exists("validation_bpz.cat"):
        #    print("Read the BPZ inputs")
        #    cols = data.columns
        #    shape=data.shape
        #    data = pd.read_csv("validation_bpz.cat",sep=' ',header=None)
        #    data.columns = cols
        #else:
        #    print("Outputting the BPZ input cat")
        #    np.savetxt(f"validation_bpz.cat",data,fmt='%3.5f')

        #if not os.path.exists("validation_bpz.bpz"): 
        #    print("Running BPZ on the validation data")
        #    curdir=os.getcwd()
        #    os.chdir('bpz-1.99.3/')
        #    os.system("echo '#NEW' > bpz_run.py")
        #    os.system("echo import sys >> bpz_run.py")
        #    os.system("echo import os >> bpz_run.py")
        #    os.system("echo 'os.environ[\"HOME\"]=\""+curdir+"\"' >>bpz_run.py")
        #    os.system("echo 'os.environ[\"BPZPATH\"]=\""+curdir+"/bpz-1.99.3/\"' >> bpz_run.py")
        #    os.system("echo 'os.environ[\"NUMERIX\"]=\"numpy\"' >> bpz_run.py")
        #    argv = ["\'bpz.py\'", "\'../validation_bpz.cat\'",
        #        "-PRIOR", "NGVS", "-SPECTRA", "CWWSB4.list",
        #        "-ZMIN", "0.001", "-ZMAX", "7.000",
        #        "-INTERP", "10", "-NEW_AB", "no",
        #        "-ODDS", "0.68", "-MIN_RMS", "0.0", "-INTERACTIVE", "no",
        #        "-PROBS_LITE", "no",
        #        "-VERBOSE", "yes",
        #        "-CHECK", "no"]
        #    os.system("echo 'sys.argv=[\""+"\",\"".join(argv)+"\"]' >> bpz_run.py")
        #    os.system("echo 'exec(open(\"bpz.py\").read())' >> bpz_run.py")
        #    res = os.system("python2 bpz_run.py")
        #    os.chdir('../')

        #print("Read the BPZ results")
        ##bpz_res = pd.read_fwf("validation_bpz.bpz",comment="#",skiprows=62,header=None)
        #os.system("grep -v '^#' validation_bpz.bpz | sed 's/  / /g' | sed 's/  / /g' | \
        #        sed 's/  / /g' | sed 's/^ //g' | sed 's/ $//g' > validation_bpz.asc")
        #bpz_res = pd.read_csv("validation_bpz.asc",sep=' ',header=None)
        #print(bpz_res)

        #print("Adding BPZ Photoz info to validation data")
        #print(data)
        #data['Z_B'] = bpz_res[[1]]
        #print(data)

        #Construct the validation data frame (just a python-to-R data conversion)
        print("Converting the data to R format")
        with localconverter(ro.default_converter + pandas2ri.converter):
              data_df = ro.conversion.py2rpy(data)

        print("Parsing the validation data into the SOM groupings")
        #Generate the validation associations/groups
        print(base.summary(data_df))
        print(base.dim(data_df))
        group_prop=kohonen.generate_kohgroup_property(som=self.train_som,data=data_df,
            expression="nrow(data)",expr_label="N",returnMatrix=True,
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

        print(base.summary(group_bins))
        print(base.length(group_bins))
        #extract the validation som (just for convenience)
        valid_som = group_prop.rx2('som')

        #Assign the sources, by group, to tomographic bins
        print("Output source tomographic bin assignments")
        valid_bin = base.unsplit(group_bins,FactorVector(valid_som.rx2['clust.classif'],
            levels=base.seq(self.opt['num_groups'])),drop=False)
        #valid_bin = valid_som.rx2['clust.classif']
        valid_bin = np.array(valid_bin)-1

        return valid_bin

