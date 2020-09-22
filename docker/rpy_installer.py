import rpy2
import rpy2.robjects as ro
import rpy2.robjects.packages as rpack
from rpy2.robjects.vectors import (
    StrVector,
    IntVector,
    DataFrame,
    FloatVector,
    FactorVector,
)
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter



packnames = [
    "shiny",
    "data.table",
    "itertools",
    "foreach",
    "doParallel",
    "RColorBrewer",
    "devtools",
    "matrixStats",
    "kohonen",
    "RANN"
]
base = ro.packages.importr("base")
utils = ro.packages.importr("utils")
stats = ro.packages.importr("stats")
gr = ro.packages.importr("graphics")
dev = ro.packages.importr("grDevices")
utils.chooseCRANmirror(ind=1)

utils.install_packages(StrVector(["gh"]))
# Selectively install what needs to be installed.
names_to_install = [x for x in packnames if not rpack.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
base.Sys_setenv(TAR=base.system("which tar", intern=True))
devtools=ro.packages.importr("devtools")

devtools.install_github("AngusWright/helpRfuncs")
devtools.install_github("AngusWright/kohonen/kohonen")

devtools = ro.packages.importr("devtools")
kohonen = ro.packages.importr("kohonen")
rann=ro.packages.importr("RANN")

