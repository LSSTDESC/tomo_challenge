import os
import yaml
import tabulate
import numpy as np
from astropy.table import Table
import matplotlib
import subprocess
import h5py
import collections
import matplotlib.pyplot as plt
import scipy.stats
# set default styles


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

methods_with_fixed_edges = [m for m in method_names if m not in methods_with_trained_edges]


methods_reordered = methods_with_trained_edges + methods_with_fixed_edges
method_names = methods_reordered
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
        disp_name = rf"{{\sc {name} }}".replace('TensorFlow_FFNN', 'FFNN').replace("_", r"\_").replace("myCombinedClassifiers", "Stacked Generalization")
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
        if name == methods_with_trained_edges[-1]:
            f.write("\\hline\n")
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


def plot_metric_grid(data, data_set, bands, filename):
    fig, axes = plt.subplots(3, 3, figsize=(10,9), sharex=True, sharey=True)
    min_sel = (data['bands'] == bands) & (data['method'] == 'Random')
    ymin_data = data[min_sel]

    # pick out equal
    if data_set == "dc2":
        ymax_data = Table.read(f"output_dc2_FIEGE_validation_riz.txt", format='ascii')
    else:
        ymax_data = Table.read(f"output_buzzard_FIEGE_validation_griz.txt", format='ascii')

    cmap = plt.get_cmap("tab10")
    x = np.array([3, 5, 7, 9])
    for i, metric in enumerate(metrics):
        ymin = ymin_data[metric]
        ymax = ymax_data[metric][:4]
        ymax_equalz = ymax_data[metric][4:]
        ax = axes[i//3, i%3]
        method_sets = [methods_with_fixed_edges, methods_with_trained_edges]
        for j, method_set in enumerate(method_sets):
            c = cmap(j)
            for method in method_set:
                if method == "Random":
                    continue
                sel = (data['bands'] == bands) & (data['method'] == method)
                y = data[sel][metric]
                b = data[sel]['bins']
                s = y != 0
                ax.plot(x[s], ((y - ymin) / (ymax - ymin))[s], color=c, lw=1)
                ax.set_xticks([3, 5, 7, 9])
                ax.tick_params(axis='x', which='minor', bottom=False)
                ax.text(3.5, 1.7, metric, bbox={'facecolor': 'white'})
                if i//3 == 0 :
                    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
                if i == 3:
                    ax.set_ylabel("Normalized metric")
        ax.plot(x, ((ymax_equalz - ymin) / (ymax - ymin)), color=cmap(2), lw=3)
        ax.plot(x, [0, 0, 0, 0], color='k', lw=3)
        ax.plot(x, [1, 1, 1, 1], color='r', lw=3)

    ax.set_ylim(-0.01, 1.98)
    ax.set_xlim(3, 9)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.025, wspace=0.06)

    legend_elements = [
        plt.Line2D([0], [0], color='k', lw=3),
        plt.Line2D([0], [0], color='r', lw=3),
        plt.Line2D([0], [0], color=cmap(2), lw=3),
        plt.Line2D([0], [0], color=cmap(0), lw=1),
        plt.Line2D([0], [0], color=cmap(1), lw=1)
    ]
    axes[0, 0].legend(legend_elements, ["Random", "Equal-N Truth", "Equal-z Truth", "Fixed edges", "Trained edges"], loc='upper right')

    fig.savefig(filename)
    plt.close()

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
    ugrizy_file = "./cosmodc2/bins/funbins_ugrizy.npy"
    z_file = "./cosmodc2/bins/z.npz"


    if not os.path.exists(riz_file):
        print("Data not downloaded for n(z) plot - not overwriting")
        return

    fig, ax = plt.subplots(3, 1, figsize=(6,9), sharex=True, sharey=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    riz = np.load(riz_file)
    griz = np.load(griz_file)
    ugrizy = np.load(ugrizy_file)
    z = np.load(z_file)['arr_0']


    nbin = 9

    for i in range(nbin):
        w = np.where(riz == i)
        weight = np.repeat(1e-5, w[0].size)
        ax[0].hist(z[w], bins=50, histtype='step', ls='-', color=colors[i], weights=weight, lw=3)

    for i in range(nbin):
        w = np.where(griz == i)
        weight = np.repeat(1e-5, w[0].size)
        ax[1].hist(z[w], bins=50, histtype='step', ls='-', color=colors[i], weights=weight, lw=3)


    for i in range(nbin):
        w = np.where(ugrizy == i)
        weight = np.repeat(1e-5, w[0].size)
        ax[2].hist(z[w], bins=50, histtype='step', ls='-', color=colors[i], weights=weight, lw=3)

    ax[0].set_ylabel('riz counts / $10^5$')
    ax[1].set_ylabel('griz counts / $10^5$')
    ax[2].set_ylabel('ugrizy counts / $10^5$')
    ax[2].set_xlabel('Redshift')
    ax[0].set_xlim(0, 3)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    fig.savefig(filename)
    plt.close(fig)

def make_initial_nz(filename):
    dc2_file = "dc2_validation.hdf5"
    buzzard_file = "buzzard_validation.hdf5"
    if not os.path.exists(buzzard_file):
        print("Data not downloaded for colour-colour plot - not overwriting")
        print("Download from https://portal.nersc.gov/cfs/lsst/txpipe/tomo_challenge_data/ugrizy_buzzard/validation.hdf5")
        print("and https://portal.nersc.gov/cfs/lsst/txpipe/tomo_challenge_data/griz/validation.hdf5")
        return

    #buzz = h5py.File(buzzard_file)
    #dc2 = h5py.File(dc2_file)
    buzz = h5py.File(buzzard_file, 'r')
    dc2 = h5py.File(dc2_file, 'r')
    buzz_z = buzz['redshift_true'][:]
    dc2_z = dc2['redshift_true'][:]
    fig, ax = plt.subplots(figsize=(5,4))
    w1 = np.repeat(1e-5, buzz_z.size)
    w2 = np.repeat(1e-5, dc2_z.size)
    ax.hist(buzz_z, bins=50, histtype='step', label='Buzzard', linewidth=3, weights=w1)
    del buzz_z
    dc2_z = dc2['redshift_true'][:]
    ax.hist(dc2_z, bins=50, histtype='step', label='CosmoDC2', linewidth=3, weights=w2, alpha=0.7)
    ax.set_xlabel("z")
    ax.set_ylabel("Counts / $10^5$")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def colour_colour_cat(ax, cat, colour, thin, label, alpha):
    if 'g_mag' in cat.keys():
        fmt = '{}_mag'
    else:
        fmt = 'mcal_mag_{}'
    g = cat[fmt.format('g')][:][::thin]
    r = cat[fmt.format('r')][:][::thin]
    i = cat[fmt.format('i')][:][::thin]
    z = cat[fmt.format('z')][:][::thin]

    gr = g - r
    ri = r - i
    gz = g - z
    iz = i - z

    ax[0].scatter(gr, ri, c=colour, s=0.5, label=label, alpha=alpha)
    ax[1].scatter(gz, iz, c=colour, s=0.5, label=label, alpha=alpha)

    ax[0].set_xlabel("g - r")
    ax[0].set_ylabel("r - i")
    ax[1].set_xlabel("g - z")
    ax[1].set_ylabel("i - z")


def make_colour_colour(filename):
    dc2_file = "dc2_validation.hdf5"
    buzzard_file = "buzzard_validation.hdf5"
    if not os.path.exists(buzzard_file):
        print("Data not downloaded for colour-colour plot - not overwriting")
        print("Download from https://portal.nersc.gov/cfs/lsst/txpipe/tomo_challenge_data/ugrizy_buzzard/validation.hdf5")
        print("and https://portal.nersc.gov/cfs/lsst/txpipe/tomo_challenge_data/griz/validation.hdf5")
        return


    fig, ax = plt.subplots(2, 1, figsize=(4, 6))

    buzz = h5py.File(buzzard_file, 'r')
    dc2 = h5py.File(dc2_file, 'r')

    colour_colour_cat(ax, buzz, blue_color, 4000, "Buzzard", alpha=1)
    colour_colour_cat(ax, dc2, orange_color, 8000, "CosmoDC2", alpha=0.5)

    ax[0].legend(frameon=True)
    
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def make_tex_tables(dc2, buzzard, dirname):
    for name, data in [("dc2", dc2), ("buzzard", buzzard)]:
        for metric in metrics:
            fn = f'{dirname}/table_{metric}_{name}.tex'
            print(fn)
            make_table(data, metric, fn)


def find_best(data, nbin, methods):

    best = collections.defaultdict(lambda: (-np.inf, ""))
    for (name, bands, bins, metric), score in data.items():
        if name not in methods:
            continue
        if bins != nbin:
            continue
        # print(metric, bands, best[metric, bands, bins], (score, name), max(best[metric, bands, bins], (score, name)))
        best[metric, bands] = max(best[metric, bands], (score, name))
    return best

def make_9bin_table(results, filename):
    N = np.array([3, 5, 7, 9])
    data = {}
    for row in results:
        name  = row['method']
        bands = row['bands']
        bins = row['bins']
        for metric in metrics:
            data[name, bands, bins, metric] = row[metric]

    n = 9
    best1 = find_best(data, n, methods_with_trained_edges)
    best2 = find_best(data, n, methods_with_fixed_edges)
    f = open(filename, 'w')

    for name in methods_reordered:
        disp_name = rf"{{\sc {name} }}".replace('TensorFlow_FFNN', 'FFNN').replace("_", r"\_").replace("myCombinedClassifiers", "Stacked Generalization")
        row = [disp_name]
        for i, bands in enumerate(['riz', 'griz']):
            for j, metric in enumerate(metrics):
                val  = data.get((name, bands, n, metric), np.nan)
                if np.isnan(val):
                    row.append(f"--")
                elif val == 0:
                    row.append(f"*")
                else:
                    if best1[metric, bands][1] == name or best2[metric, bands][1] == name:
                        row.append(f"\\textbf{{{val:.1f}}}")
                        print("best ", metric, bands, name)
                    else:
                        row.append(f"{val:.1f}")

        f.write(" & ".join(row))
        f.write("\\\\ \n")
        if name == methods_with_trained_edges[-1]:
            f.write("\\hline\n")
    f.close()


def make_metric_grid(data_set, bands):
    
    # cut doen to the correct bands
    data_set = data_set[data_set['bands'] == bands]
    metrics = ['SNR_ww','SNR_gg','SNR_3x2','FOM_ww','FOM_gg','FOM_3x2','FOM_DETF_ww','FOM_DETF_gg','FOM_DETF_3x2']
    bins = [3, 5, 7, 9]
    
    nmetric = len(metrics)
    nbin = len(bins)

    nrow = nbin * nmetric
    row_meanings = [(b, metric) for b in bins for metric in metrics]
    row_labels = [f'n={b} {metric}' for b in bins for metric in metrics]
    
    methods = np.unique(data_set['method'])
    ncol = len(methods)

    grid = np.zeros((nrow, ncol), dtype=int)
    for i, (row, row_label) in enumerate(zip(row_meanings, row_labels)):
        b, metric = row
        d = data_set[data_set['bins']==b]
        # check that the ordering is the same.  should be
        assert (d['method'] == method_names).all()
        ranks = len(method_names) - scipy.stats.rankdata(d[metric]) + 1
        grid[i, :] = ranks
    
    return row_labels, grid
        
def plot_rank_grid(dcz, buzz, filename):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 13))
    # make_metric_grid(dc2, 'riz')
    for i, (data_set_name, data_set) in enumerate([("CosmoDC2", dc2), ("Buzzard", buzz)]):
        for j, bands in enumerate(["riz", "griz"]):
            row_labels, grid = make_metric_grid(data_set, bands)
            im = axes[i, j].imshow(grid.T, cmap='summer_r')
            ax = axes[i, j]
            
            if i == 0:
                ax.title.set_text(f"{bands}")
            
            xlabels = row_labels if i == 1 else []
            
            ylabels = method_names if j == 0 else []
            ax.set_yticks(np.arange(len(method_names)))
            ax.set_yticklabels(ylabels, fontsize=10)        
            ax.tick_params('y', length=0, width=1)
            ax.tick_params('y', length=0, width=1, which='minor')

            # The major ticks, which we use for the text
            ax.set_xticks(np.arange(len(row_labels)))
            ax.set_xticklabels(xlabels, fontsize=10, rotation=90)
            ax.tick_params('x', length=0, width=1, which='major')
            
            # The minor, which we use for the tick itself
            ax.set_xticks(np.arange(len(row_labels))+0.5, minor=True)        
            ax.set_xticklabels([], minor=True)
            minor_length = 10 if i == 1 else 0
            ax.tick_params('x', length=minor_length, width=1, which='minor')

    fig.subplots_adjust(bottom=0.25, hspace=-0.03, wspace=0.03)

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
    cbar_ax.text(12, -12, "Rank", fontsize=20)
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar_ax.grid(which='both', color='black')
    cbar_ax.tick_params('both', length=0, which='both')

    label_ax = fig.add_axes([0.95, 0.05, 0.05, 0.9])
    label_ax.axis('off')
    label_ax.text(-0.75, 0.64, "CosmoDC2", rotation=-90., fontsize=25)
    label_ax.text(-0.75, 0.32, "Buzzard", rotation=-90., fontsize=25)

    fig.savefig(filename)
    plt.close(fig)

if __name__ == '__main__':
    matplotlib.use('agg')
    plt.style.use('StyleSheet.mplstyle')

    dc2 = load_table('cosmodc2')
    buzzard = load_table('buzzard')
    plot_metric_grid(dc2, "dc2", "riz", "metric_grid_dc2_riz.pdf")
    plot_metric_grid(dc2, "dc2", "griz", "metric_grid_dc2_griz.pdf")
    plot_metric_grid(buzzard, "buzzard", "riz", "metric_grid_buzzard_riz.pdf")
    plot_metric_grid(buzzard, "buzzard", "griz", "metric_grid_buzzard_griz.pdf")
    plot_metric_comparisons(dc2, buzzard, "metric_comparisons.pdf")
    plot_g_band_loss(dc2, buzzard, "g_band_loss.pdf")
    make_tex_tables(dc2, buzzard, 'tables')
    plot_edge_type_comparison(dc2, "edge_comparison.pdf")
    plot_funbins_nz('funbins_nz.pdf')
    make_colour_colour("colour_colour.pdf")
    make_initial_nz("initial_data.pdf")
    make_9bin_table(buzzard, "9bin_buzzard.tex")
    make_9bin_table(dc2, "9bin_dc2.tex")
    plot_rank_grid(dc2, buzzard, "rank_grid.pdf")



