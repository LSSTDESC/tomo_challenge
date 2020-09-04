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
from rpy2.robjects.vectors import StrVector, IntVector, DataFrame, FloatVector, FactorVector
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

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
import execnet

def call_python_version(Version, Script):
    gw = execnet.makegateway("popen//python=python%s" % Version)
    channel = gw.remote_exec("""
          import sys
          def call_script():
            exec(open("%s").read())
          channel.send(call_script())
          """ % (Script))
    channel.send("")
    return channel.receive()

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
        property_labels = ("mean_z_true","med_z_true","sd_z_true","mad_z_true","N","iqr_z_true","mean_ZB","median_ZB")
        property_expressions = ("mean(data$redshift_true)","median(data$redshift_true)","sd(data$redshift_true)",
                                "mad(data$redshift_true)","nrow(data)",
                                "diff(quantile(data$redshift_true,probs=pnorm(c(-2,2))))","mean(data$Z_B)","median(data$Z_B)")
        #Define the SOM variables
        if self.bands == 'riz':
            #riz bands
            expressions = ("r-i","r-z","i-z",
                           "z","r-i-(i-z)","Z_B")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(np.transpose(np.array([
                                   ["r_SDSS","i_SDSS","z_SDSS"],
                                   ["1,2","3,4","5,6"],
                                   ["AB","AB","AB"],
                                   ["0.01","0.01","0.01"],
                                   ["0.00","0.00","0.00"]])))
            #Output the columns file 
            columns.to_csv(r"training_bpz.columns",sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(np.transpose(
                                  np.array([["M_0","Z_S"],
                                            ["5","7"]])))
            #Output the columns file 
            columns.to_csv(r"training_bpz.columns",sep=' ',index=None,header=None,mode='a')
        elif self.bands == 'griz':
            #griz bands
            expressions = ("g-r","g-i",
                           "g-z","r-i","r-z","i-z",
                           "z","g-r-(r-i)",
                           "r-i-(i-z)","Z_B")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(["g_SDSS","r_SDSS","i_SDSS","z_SDSS"],
                                   ["1,2","3,4","5,6","7,8"],
                                   ["AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01"],
                                   ["0.000001","0.000001","0.000001","0.000001"])
            #Output the columns file 
            columns.to_csv(r"training_bpz.columns",sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(["M_0","Z_S"],
                                   ["7","9"])
            #Output the columns file 
            columns.to_csv(r"training_bpz.columns",sep=' ',index=None,header=None,mode='a')
        elif self.bands == 'grizy':
            #grizy bands
            expressions = ("g-r","g-i",
                           "g-z","g-y","r-i","r-z","r-y","i-z","i-y",
                           "z-y","z","g-r-(r-i)",
                           "r-i-(i-z)","i-z-(z-y)","Z_B")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(["g_SDSS","r_SDSS","i_SDSS","z_SDSS","y_SDSS"],
                                   ["1,2","3,4","5,6","7,8","9,10"],
                                   ["AB","AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01","0.01"],
                                   ["0.000001","0.000001","0.000001","0.000001","0.000001"])
            #Output the columns file 
            columns.to_csv(r"training_bpz.columns",sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(["M_0","Z_S"],
                                   ["7","11"])
            #Output the columns file 
            columns.to_csv(r"training_bpz.columns",sep=' ',index=None,header=None,mode='a')
        elif self.bands == 'ugriz':
            #ugrizy bands
            expressions = ("u-g","u-r","u-i","u-z","g-r","g-i",
                           "g-z","r-i","r-z","i-z",
                           "z","u-g-(g-r)","g-r-(r-i)",
                           "r-i-(i-z)","Z_B")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(["u_SDSS","g_SDSS","r_SDSS","i_SDSS","z_SDSS"],
                                   ["1,2","3,4","5,6","7,8","9,10"],
                                   ["AB","AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01","0.01"],
                                   ["0.000001","0.000001","0.000001","0.000001","0.000001"])
            #Output the columns file 
            columns.to_csv(r"training_bpz.columns",sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(["M_0","Z_S"],
                                   ["9","11"])
            #Output the columns file 
            columns.to_csv(r"training_bpz.columns",sep=' ',index=None,header=None,mode='a')
        elif self.bands == 'ugrizy':
            #ugrizy bands
            expressions = ("u-g","u-r","u-i","u-z","u-y","g-r","g-i",
                           "g-z","g-y","r-i","r-z","r-y","i-z","i-y",
                           "z-y","z","u-g-(g-r)","g-r-(r-i)",
                           "r-i-(i-z)","i-z-(z-y)","Z_B")
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(["u_SDSS","g_SDSS","r_SDSS","i_SDSS","z_SDSS","y_SDSS"],
                                   ["1,2","3,4","5,6","7,8","9,10","11,12"],
                                   ["AB","AB","AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01","0.01","0.01"],
                                   ["0.000001","0.000001","0.000001","0.000001","0.000001","0.000001"])
            #Output the columns file 
            columns.to_csv(r"training_bpz.columns",sep=' ',index=None,header=None,mode='w')
            columns = pd.DataFrame(["M_0","Z_S"],
                                   ["9","13"])
            #Output the columns file 
            columns.to_csv(r"training_bpz.columns",sep=' ',index=None,header=None,mode='a')

        print("Preparing the data for BPZ")
        #Output the BPZ input catalogue 
        training_data = pd.DataFrame.from_dict(training_data)
        print("Adding redshift info to training data")
        training_data['redshift_true'] = training_z
        if os.path.exists("training_bpz.cat"):
            print("Read the BPZ inputs")
            cols = training_data.columns
            training_data = pd.read_csv("training_bpz.cat",sep=' ')
            training_data.columns = cols
            training_z = training_data[['redshift_true']]
        else:
            if sparse_frac < 1:
                print("Sparse Sampling the training data")
                cut = np.random.uniform(0, 1, training_z.size) < sparse_frac
                training_data = training_data[cut]

            print("Outputting the BPZ input cat")
            np.savetxt(f"training_bpz.cat",training_data,fmt='%3.5f')

        if not os.path.exists("bpz-1.99.3/"): 
            os.system("bash INSTALL.sh")
        
        if not os.path.exists("training_bpz.bpz"): 
            print("Running BPZ on the training data")
            curdir=os.getcwd()
            os.chdir('bpz-1.99.3/')
            os.system("echo '#NEW' > bpz_run.py")
            os.system("echo import sys >> bpz_run.py")
            os.system("echo import os >> bpz_run.py")
            os.system("echo 'os.environ[\"HOME\"]=\""+curdir+"\"' >>bpz_run.py")
            os.system("echo 'os.environ[\"BPZPATH\"]=\""+curdir+"/bpz-1.99.3/\"' >> bpz_run.py")
            os.system("echo 'os.environ[\"NUMERIX\"]=\"numpy\"' >> bpz_run.py")
            argv = ["\'bpz.py\'", "\'../training_bpz.cat\'"]
            os.system("echo 'sys.argv=[\""+"\",\"".join(argv)+"\"]' >> bpz_run.py")
            os.system("echo 'exec(open(\"bpz.py\").read())' >> bpz_run.py")
            res = os.system("python2 bpz_run.py")
            os.chdir('../')

        print("Read the BPZ results")
        bpz_res = pd.read_fwf("training_bpz.bpz",comment="#",skiprows=62,header=None)

        print("Adding BPZ Photoz info to training data")
        training_data['Z_B'] = bpz_res[[1]]

        #Construct the training data frame (just a python-to-R data conversion)
        print("Converting the training data to R format")
        with localconverter(ro.default_converter + pandas2ri.converter):
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
            props = cell_prop.rx2['property']
            props.rx[base.which(base.is_na(props))] = -1
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
        
        #Construct the BPZ columns file 
        if self.bands == 'riz':
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(["r_SDSS","i_SDSS","z_SDSS"],
                                   ["1,5","2,6","3,7"],
                                   ["AB","AB","AB"],
                                   ["0.01","0.01","0.01"],
                                   ["0.000001","0.000001","0.000001"])
            #Output the columns file 
            np.savetxt(f"validation_bpz.columns",columns,fmt='%3.5f')
            columns = pd.DataFrame(["M_0","Z_S"],
                                   ["3","8"])
            #Output the columns file 
            np.savetxt(f"validation_bpz.columns",columns,fmt='%d',append=True)
        elif self.bands == 'griz':
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(["g_SDSS","r_SDSS","i_SDSS","z_SDSS"],
                                   ["1,5","2,6","3,7","4,8"],
                                   ["AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01"],
                                   [0.,0.,0.,0.])
            #Output the columns file 
            np.savetxt(f"validation_bpz.columns",columns,fmt='%3.5f')
            columns = pd.DataFrame(["M_0","Z_S"],
                                   ["4","9"])
            #Output the columns file 
            np.savetxt(f"validation_bpz.columns",columns,fmt='%d',append=True)
        elif self.bands == 'grizy':
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(["g_SDSS","r_SDSS","i_SDSS","z_SDSS","y_SDSS"],
                                   ["1,6","2,7","3,8","4,9","5,10"],
                                   ["AB","AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01","0.01"],
                                   [0.,0.,0.,0.,0.])
            #Output the columns file 
            np.savetxt(f"validation_bpz.columns",columns,fmt='%3.5f')
            columns = pd.DataFrame(["M_0","Z_S"],
                                   ["4","13"])
            #Output the columns file 
            np.savetxt(f"validation_bpz.columns",columns,fmt='%d',append=True)
        elif self.bands == 'ugriz':
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(["u_SDSS","g_SDSS","r_SDSS","i_SDSS","z_SDSS"],
                                   ["1,6","2,7","3,8","4,9","5,10"],
                                   ["AB","AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01","0.01"],
                                   [0.,0.,0.,0.,0.])
            #Output the columns file 
            np.savetxt(f"validation_bpz.columns",columns,fmt='%3.5f')
            columns = pd.DataFrame(["M_0","Z_S"],
                                   ["5","11"])
            #Output the columns file 
            np.savetxt(f"validation_bpz.columns",columns,fmt='%d',append=True)
        elif self.bands == 'ugrizy':
            # Filter columns AB/Vega zp_error zp_offset
            columns = pd.DataFrame(["u_SDSS","g_SDSS","r_SDSS","i_SDSS","z_SDSS","y_SDSS"],
                                   ["1,7","2,8","3,9","4,10","5,11","6,12"],
                                   ["AB","AB","AB","AB","AB","AB"],
                                   ["0.01","0.01","0.01","0.01","0.01","0.01"],
                                   ["0.000001","0.000001","0.000001","0.000001","0.000001","0.000001"])
            #Output the columns file 
            np.savetxt(f"validation_bpz.columns",columns,fmt='%3.5f')
            columns = pd.DataFrame(["M_0","Z_S"],
                                   ["5","13"])
            #Output the columns file 
            np.savetxt(f"validation_bpz.columns",columns,fmt='%d',append=True)

            
        #Output the columns file 
        np.savetxt(f"validation_bpz.columns",columns,fmt='%3.5f')

        #Number of tomographic bins 
        n_bin = self.opt['bins']

        print("Preparing the data")
        data = pd.DataFrame.from_dict(data)

        print("Outputting the BPZ input cat")
        np.savetxt(f"validation_bpz.cat",data,fmt='%3.5f')

        print("Running BPZ on the validation data")
        call_python_version("2.7","bpz","bpz",['validation_bpz.cat'])

        print("Read the BPZ results")
        bpz_res = pd.read_csv("validation_bpz.bpz",sep=' ')

        print("Adding BPZ Photoz info to validation data")
        data['Z_B'] = bpz_res[["Z_B"]]

        #Construct the validation data frame (just a python-to-R data conversion)
        print("Converting the data to R format")
        with localconverter(ro.default_converter + pandas2ri.converter):
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

