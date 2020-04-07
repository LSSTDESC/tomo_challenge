import matplotlib.pyplot as plt
import numpy as np
def compute_score(tomo_bin, z):
    """
    """
    return 0


def plot_distributions(z, tomo_bin, filename, nominal_edges=None):
    fig = plt.figure()
    for i in range(int(tomo_bin.max())+1):
        w = np.where(tomo_bin == i)
        plt.hist(z[w], bins=50)

    # Plot verticals at nominal edges, if given
    if nominal_edges is not None:
        for x in nominal_edges:
            plt.axvline(x, color='k', linestyle=':')

    plt.savefig(filename)
    plt.close()
