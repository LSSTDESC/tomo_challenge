import yaml
import tabulate
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# set default styles
plt.style.use('StyleSheet.mplstyle')



method_names = [
    "Autokeras_LSTM",
    "CNN",
    "ComplexSOM",
    "ENSEMBLE1",
    "funbins",
    "GPzBinning",
    "IBandOnly",
    "JaxCNN",
    "JaxResNet",
    "LGBM",
    "LSTM",
#     "MineCraft",
    "mlpqna",
    "myCombinedClassifiers",
    "NeuralNetwork1",
    "NeuralNetwork2",
    "PCACluster",
    "Random",
    "RandomForest",
    "SimpleSOM",
    "TensorFlow_FFNN",
    "TCN",
    "UTOPIA",
    "ZotBin",
    "ZotNet",
]

metrics = ["SNR_ww", "SNR_gg", "SNR_3x2", "FOM_ww", "FOM_gg", "FOM_3x2", "FOM_DETF_ww", "FOM_DETF_gg", "FOM_DETF_3x2"]


class EverythingIsZero:
    def __getitem__(self, x):
        return 0
    
class EverythingIsNan:
    def __getitem__(self, x):
        return np.nan


def make_table(results, metric):
    N = np.array([3, 5, 7, 9])
    row_format = r"{0} & {1} & {2}    & {3}    & {4}    & {5}             & {6}             & {7}             & {8}\\"
    data = {}
    for row in results:
        name  = row['method']
        bands = row['bands']
        bins = row['bins']
        data[name, bands, bins] = row[metric]


    for name in method_names:
        disp_name = rf"{{\sc {name} }}".replace("_", r"\_")
        row = [disp_name]
        for i, bands in enumerate(['riz', 'griz']):
            for n in N:
                val  = data.get((name, bands, n), np.nan)
                if np.isnan(val) or val == 0:
                    row.append(f"--")
                else:
                    row.append(f"{val:.1f}")

        print(row_format.format(*row))


def load_table(base):
    N = np.array([3, 5, 7, 9])
    data = []
    for name in method_names:
        for i, bands in enumerate(['riz', 'griz']):
            for n in N:
                row = [name, bands, n]
                fn = f"{base}/metrics/{name}_{n}_{bands}_0.npy.yml"
                try:
                    info =  yaml.safe_load(open(fn))
                except:
                    info = EverythingIsZero()
                for metric in  metrics:
                    row.append(info[metric])
                data.append(row)
    results = Table(rows=data, names=["method", "bands", "bins"] + metrics)
    return results

def pair_riz_griz(data):
    riz = data['bands']=='riz'
    griz = data['bands']=='griz'
    assert (data['method'][riz] == data['method'][griz]).all()
    assert (data['bins'][riz] == data['bins'][griz]).all()
    return data[riz], data[griz]


def plot_g_band_loss(dc2, buzzard, filename):
    fig = plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(8,4))

    cmap = matplotlib.cm.Dark2
    riz, griz = pair_riz_griz(dc2)
    x = riz['FOM_DETF_3x2']
    y = riz['FOM_DETF_3x2'] / griz['FOM_DETF_3x2']
    c = riz['bins']
    for i, nbin in enumerate([3,5,7,9]):
        s = riz['bins']==nbin
        ax[0].plot(x[s], y[s], '.', markersize=8, color=cmap.colors[i])

    riz, griz = pair_riz_griz(buzzard)
    x = riz['FOM_DETF_3x2']
    y = riz['FOM_DETF_3x2'] / griz['FOM_DETF_3x2']
    c = riz['bins']
    for i, nbin in enumerate([3,5,7,9]):
        s = riz['bins']==nbin
        ax[1].plot(x[s], y[s], '.', markersize=8, color=cmap.colors[i], label=f'$n_b={nbin}$')
    ax[1].legend(ncol=2, frameon=True)
    ax[0].axhline(1, color='k')
    ax[1].axhline(1, color='k')

    ax[0].set_xlabel("riz FoM (CosmoDC2)")
    ax[1].set_xlabel("riz FoM (Buzzard)")
    ax[0].set_ylabel("(riz FoM) / (griz FoM)")
    ax[0].set_ylim(0.5, 1.2)
    ax[1].set_ylim(0.5, 1.2)
    ax[0].set_xlim(0, 150)
    ax[1].set_xlim(0, 100)
    ax[1].set_yticklabels([])
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
 

if __name__ == '__main__':
    dc2 = load_table('cosmodc2')
    buzzard = load_table('buzzard')
    plot_g_band_loss(dc2, buzzard, "g_band_loss.pdf")

    # pylab.scatter(dc2['bins'], dc2['FOM_DETF_3x2'])
    # pylab.show()
    #dc2.sort("FOM_ww")
    # pylab.show()
    # buzzard = load_table('buzzard')
    # print(buzzard)
    # make_table(dc2, 'FOM_DETF_3x2')
    # make_table(dc2, 'FOM_ww')
