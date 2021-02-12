import os
import yaml
import tabulate
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import subprocess

# set default styles
plt.style.use('StyleSheet.mplstyle')


blue_color =  '#1f77b4'
orange_color =  '#ff7f0e'


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
    "PQNLD",
    "Random",
    "RandomForest",
    "SimpleSOM",
    "TensorFlow_FFNN",
    "TCN",
    "UTOPIA",
    "ZotBin",
    "ZotNet",
]


methods_with_trained_edges = [
    "ComplexSOM",
    "JaxCNN",
    "JaxResNet",
    "NeuralNetwork1",
    "NeuralNetwork2",
    "PCACluster",
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


def make_table(results, metric, filename):
    N = np.array([3, 5, 7, 9])
    row_format = r"{0} & {1} & {2}    & {3}    & {4}    & {5}             & {6}             & {7}             & {8}\\"
    data = {}
    for row in results:
        name  = row['method']
        bands = row['bands']
        bins = row['bins']
        data[name, bands, bins] = row[metric]

    f = open(filename, 'w')
    for name in method_names:
        disp_name = rf"{{\sc {name} }}".replace("_", r"\_").replace("myCombinedClassifiers", "Stacked Generalization")
        row = [disp_name]
        for i, bands in enumerate(['riz', 'griz']):
            for n in N:
                val  = data.get((name, bands, n), np.nan)
                if np.isnan(val):
                    row.append(f"--")
                elif val == 0:
                    row.append(f"*")
                else:
                    row.append(f"{val:.1f}")

        f.write(row_format.format(*row))
        f.write("\n")
    f.close()


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


def color_plot(data, x, y, ax):
    c = data['bins']
    cmap = matplotlib.cm.Dark2
    for i, nbin in enumerate([3,5,7,9]):
        s = data['bins']==nbin
        ax.plot(x[s], y[s], '.', markersize=8, color=cmap.colors[i], label=f'$n_b={nbin}$')


def plot_metric_comparisons(dc2, buzzard, filename):
    assert (dc2['method'] == buzzard['method']).all()
    assert (dc2['bins'] == buzzard['bins']).all()


    fig, ax = plt.subplots(4, 1, figsize=(4,9), sharex=True)
    x = dc2['FOM_DETF_3x2'].copy()
    x[x==0] = np.nan
    y = buzzard['FOM_DETF_3x2'].copy()
    y[y==0] = np.nan
    color_plot(dc2, x, y, ax[0])
    ax[0].set_ylabel("Buzzard FOM")
    ax[0].set_ylim(0)
    ax[0].legend(frameon=True)


    x = dc2['FOM_DETF_3x2'].copy()
    x[x==0] = np.nan
    y = dc2['FOM_3x2'].copy() / 1000
    y[y==0] = np.nan
    color_plot(dc2, x, y, ax[1])
    ax[1].set_ylabel(r"$\Omega_c - \sigma_8$ FOM / 1000")
    ax[1].set_ylim(0)


    x = dc2['FOM_DETF_3x2'].copy()
    x[x==0] = np.nan
    y = dc2['FOM_DETF_ww'].copy()
    y[y==0] = np.nan
    color_plot(dc2, x, y, ax[2])
    ax[2].set_ylabel(r"Lensing FOM")
    ax[2].set_ylim(0)


    x = dc2['FOM_DETF_3x2'].copy()
    x[x==0] = np.nan
    y = dc2['SNR_3x2'].copy()
    y[y==0] = np.nan

    color_plot(dc2, x, y, ax[3])
    ax[3].set_ylabel(r"SNR Metric")
    ax[3].set_ylim(500)


    ax[-1].set_xlabel("CosmoDC2 DETF 3x2pt FOM")
    ax[0].set_xlim(0, 180)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)
    fig.savefig(filename)
    plt.close(fig)



def plot_g_band_loss(dc2, buzzard, filename):
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
 

def plot_edge_type_comparison(data, filename):
    _, data = pair_riz_griz(data)
    fig, ax = plt.subplots(2, 1, figsize=(5,8), sharex=True)

    untrained = [m for m in method_names if m not in methods_with_trained_edges]

    def get(method):
        s = data['method'] == method
        m = data[s]
        x = m['bins']
        y = m['FOM_DETF_3x2']
        z = m['FOM_DETF_ww']
        y[y==0] = np.nan
        z[z==0] = np.nan
        return x, y, z


    color =  blue_color
    alpha = 0.5
    lw = 1
    style =  "--"
    need_legend = True
    for method in untrained:
        x, y, z = get(method)
        label = "Fixed edges" if need_legend else None
        need_legend = False
        ax[0].plot(x, y, color, alpha=alpha, lw=lw, ls=style, label=label)
        ax[1].plot(x, z, color, alpha=alpha, lw=lw, ls=style)

    color =  orange_color
    alpha = 1.0
    lw = 3
    style =  "-"
    need_legend = True

    for method in methods_with_trained_edges:
        x, y, z = get(method)
        label = "Trained edges" if need_legend else None
        need_legend = False
        ax[0].plot(x, y, color, alpha=alpha, lw=lw, ls=style, label=label)
        ax[1].plot(x, z, color, alpha=alpha, lw=lw, ls=style)

    ax[0].tick_params(which='minor', length=0, axis='x')
    ax[1].tick_params(which='minor', length=0, axis='x')
    ax[0].legend(frameon=True)
    ax[1].set_xlabel("Number of bins")
    ax[0].set_ylabel("3x2pt metric")
    ax[1].set_ylabel("Lensing metric")
    ax[1].set_xlim(3, 9)
    ax[0].set_ylim(0, 175)
    ax[1].set_ylim(0, 1.25)
    ax[1].set_xticks([3, 5, 7, 9])

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    fig.savefig(filename)
    plt.close(fig)


def plot_funbins_nz(filename):
    riz_file = "./cosmodc2/bins/funbins_9_riz_0.npy"
    griz_file = "./cosmodc2/bins/funbins_9_griz_0.npy"
    z_file = "./cosmodc2/bins/z.npz"


    if not os.path.exists(riz_file):
        print("Data not downloaded for n(z) plot - not overwriting")
        return

    fig, ax = plt.subplots(2, 1, figsize=(6,6), sharex=True, sharey=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    riz = np.load(riz_file)
    griz = np.load(griz_file)
    z = np.load(z_file)['arr_0']


    nbin = 9

    for i in range(nbin):
        w = np.where(griz == i)
        weight = np.repeat(1e-5, w[0].size)
        ax[0].hist(z[w], bins=50, histtype='step', ls='-', color=colors[i], weights=weight, lw=3)

    for i in range(nbin):
        w = np.where(riz == i)
        weight = np.repeat(1e-5, w[0].size)
        ax[1].hist(z[w], bins=50, histtype='step', ls='-', color=colors[i], weights=weight, lw=3)

    ax[0].set_ylabel('riz counts / $10^5$')
    ax[1].set_ylabel('griz counts / $10^5$')
    ax[1].set_xlabel('Redshift')
    ax[0].set_xlim(0, 3)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    fig.savefig(filename)
    plt.close(fig)



def make_tex_tables(dc2, buzzard, dirname):
    for name, data in [("dc2", dc2), ("buzzard", buzzard)]:
        for metric in metrics:
            fn = f'{dirname}/table_{metric}_{name}.tex'
            print(fn)
            make_table(data, metric, fn)


if __name__ == '__main__':
    dc2 = load_table('cosmodc2')
    buzzard = load_table('buzzard')
    plot_metric_comparisons(dc2, buzzard, "metric_comparisons.pdf")
    plot_g_band_loss(dc2, buzzard, "g_band_loss.pdf")
    make_tex_tables(dc2, buzzard, 'tables')
    plot_edge_type_comparison(dc2, "edge_comparison.pdf")
    plot_funbins_nz('funbins_nz.pdf')