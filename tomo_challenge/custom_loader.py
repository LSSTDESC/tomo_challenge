import numpy as np
import h5py

def custom_loader(fname):
        data = h5py.File(fname, 'r')
        r_mag = data['r_mag']
        g_mag = data['g_mag']
        i_mag = data['i_mag']
        z_mag = data['z_mag']
        redshift = data['redshift_true']
        all_mags = np.vstack([g_mag, r_mag, i_mag, z_mag])
        all_mags = all_mags.T
        mask = (all_mags != np.inf).all(axis=1)
        all_mags = all_mags[mask,:]
        redshift = redshift[mask]
        gr_color = all_mags[:,0] - all_mags[:,1]
        ri_color = all_mags[:,1] - all_mags[:,2]
        iz_color = all_mags[:,2] - all_mags[:,3]
        all_colors = np.vstack([gr_color, ri_color, iz_color])
        all_colors = all_colors.T
        return np.hstack([all_mags, all_colors]), redshift

def custom_redshift_loader(fname, bands=None, errors=None, colors=None):
    _, redshift = custom_loader(fname)
    return redshift

def custom_data_loader(fname):
    features, _ = custom_loader(fname)
    return features
