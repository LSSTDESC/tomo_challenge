from .. import jax_metrics as j_metrics
from flax import nn, optim, serialization, jax_utils
import jax.random as rand
import jax.numpy as jnp
import jax
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
import h5py
import tensorflow as tf

def get_classifier(n_features, n_bins):
    class Resnet(nn.Module):
        def apply(self, x):
            #1st block
            conv_x = nn.Conv(x, features=64, kernel_size=(8,), padding='SAME')
            conv_x = nn.BatchNorm(conv_x)
            conv_x = nn.leaky_relu(conv_x)
            
            conv_y = nn.Conv(conv_x, 64, kernel_size=(5,), padding='SAME')
            conv_y = nn.BatchNorm(conv_y)
            conv_y = nn.leaky_relu(conv_y)
                
            conv_z = nn.Conv(conv_y, 64, kernel_size=(3,), padding='SAME')
            conv_z = nn.BatchNorm(conv_z)
                
            short_y = nn.Conv(x, 64, kernel_size=(1,), padding='SAME')
            short_y = nn.BatchNorm(short_y)
                
            output_1 = nn.leaky_relu(short_y + conv_z)
                
                #2nd block
            conv_x = nn.Conv(output_1, features=64*2, kernel_size=(8,), padding='SAME')
            conv_x = nn.BatchNorm(conv_x)
            conv_x = nn.leaky_relu(conv_x)
                
            conv_y = nn.Conv(conv_x, 64*2, kernel_size=(5,), padding='SAME')
            conv_y = nn.BatchNorm(conv_y)
            conv_y = nn.leaky_relu(conv_y)
                
            conv_z = nn.Conv(conv_y, 64*2, kernel_size=(3,), padding='SAME')
            conv_z = nn.BatchNorm(conv_z)
                
            short_y = nn.Conv(output_1, 64*2, kernel_size=(1,), padding='SAME')
            short_y = nn.BatchNorm(short_y)
                
            output_2 = nn.leaky_relu(short_y + conv_z)
                
                #3rd block
            conv_x = nn.Conv(output_2, features=64*2, kernel_size=(8,), padding='SAME')
            conv_x = nn.BatchNorm(conv_x)
            conv_x = nn.leaky_relu(conv_x)
                
            conv_y = nn.Conv(conv_x, 64*2, kernel_size=(5,), padding='SAME')
            conv_y = nn.BatchNorm(conv_y)
            conv_y = nn.leaky_relu(conv_y)
                
            conv_z = nn.Conv(conv_y, 64*2, kernel_size=(3,), padding='SAME')
            conv_z = nn.BatchNorm(conv_z)
                
            short_y = nn.BatchNorm(output_2)
                
            output_3 = nn.leaky_relu(short_y + conv_z)
                
            gap = jnp.mean(output_3, axis=1)
            return nn.softmax(nn.Dense(gap, n_bins))
    _, initial_params = Resnet.init_by_shape(rand.PRNGKey(0), [((1, n_features, 1), jnp.float32)])
    return nn.Model(Resnet, initial_params)


#def get_classifier(n_features, n_bins):
#    _, initial_params = Resnet(n_bins).init_by_shape( rand.PRNGKey(0), [((1, n_features,1), jnp.float32)])
#    return nn.Model(Resnet(n_bins), initial_params)

def get_batches(features, redshift, batch_size, train=True):
    if train:
        dataset = tf.data.Dataset.from_tensor_slices((features, redshift))
        dataset = dataset.shuffle(buffer_size=2048).batch(batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(features).batch(batch_size)
    return dataset

def train(n_bins, training_data, training_z, batch_size=512, epochs=40, lr=0.001):
    model = get_classifier(training_data.shape[1], n_bins)
    optimizer = optim.Adam(learning_rate=lr).create(model)

    @jax.jit
    def train_step(optimizer, x, y):
             # This is the loss function
        def loss_fn(model):
                 # Apply classifier to features
            w = model(x)
                 # returns - score, because we want to maximize score
            return 1000./ j_metrics.compute_snr_score(w, y)
             # Compute gradients
        loss, g = jax.value_and_grad(loss_fn)(optimizer.target)
             # Perform gradient descent
        optimizer = optimizer.apply_gradient(g)
        return optimizer, loss
    
    #def get_batches():
    #    train_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_z))
    #    train_dataset = train_dataset.shuffle(buffer_size=2048).batch(batch_size)
    #    return train_dataset

    losses = []
    
    with nn.stochastic(rand.PRNGKey(0)):
        for e in range(epochs):
            print('starting epoch {}'.format(e))
            batches = get_batches(training_data, training_z, batch_size)
            for i, (x_train, labels) in enumerate(batches.as_numpy_iterator()):
                optimizer, loss = train_step(optimizer, x_train, labels)
            losses.append(loss)
            print('Epoch {}\nLoss = {}'.format(e, loss))
        model = optimizer.target
    return losses, model

def predict(model, data, batch_size):
    batches = get_batches(data, 0., batch_size, train=False)
    preds = []
    for i, x_test in enumerate(batches.as_numpy_iterator()):
        #print(x_test.shape)
        p = model(x_test)
        preds.append(p)
    result = np.concatenate([p for p in preds], axis=0)
    return result

def load_data(n_bins, fname, no_inf=True, take_colors=True, cutoff=0.0):

    data = h5py.File(fname, 'r')
    r_mag = data['r_mag']
    g_mag = data['g_mag']
    i_mag = data['i_mag']
    z_mag = data['z_mag']
    redshift = data['redshift_true']
    all_mags = np.vstack([g_mag, r_mag, i_mag, z_mag])
    all_mags = all_mags.T
    if no_inf:
        mask = (all_mags != np.inf).all(axis=1)
        all_mags = all_mags[mask,:]
        redshift = redshift[mask]
    else:
        bad = ~np.isfinite(all_mags)
        all_mags[bad] = 30.0
    gr_color = all_mags[:,0] - all_mags[:,1]
    ri_color = all_mags[:,1] - all_mags[:,2]
    iz_color = all_mags[:,2] - all_mags[:,3]
    all_colors = np.vstack([gr_color, ri_color, iz_color])
    all_colors = all_colors.T
    p = np.linspace(0, 100, n_bins+1)
    z_edges = np.percentile(redshift, p)
    train_bin = np.zeros(all_mags.shape[0])
    for i in range(n_bins):
        z_low = z_edges[i]
        z_high = z_edges[i+1]
        train_bin[(redshift > z_low) & (redshift <= z_high)] = i
    if cutoff != 0.0:
        cut = np.random.uniform(0, 1, all_mags.shape[0]) < cutoff
        train_bin = train_bin[cut].reshape(-1,1)
        all_mags = all_mags[cut]
        all_colors = all_colors[cut]
        redshift = redshift[cut]
    else:
        train_bin = train_bin.reshape(-1,1)
    if take_colors:
        return np.hstack([all_mags, all_colors]), redshift, train_bin.astype(int), z_edges
    else:
        return mags, redshift, train_bin.astype(int), z_edges

def prepare_data(data, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        return np.expand_dims(scaled, axis=-1), scaler
    else:
        scaled = scaler.transform(data)
        return np.expand_dims(scaled, axis=-1)


if __name__ == '__main__':
    n_bins = int(sys.argv[4])
    print('preparing data')
    feats_train, z_train, bins_train, edges_train = load_data(n_bins, sys.argv[1], no_inf=False)

    x_train, scaler = prepare_data(feats_train)

    #print(x_train.shape)
    #model = get_classifier(x_train.shape[1], n_bins)
    print('training')
    losses, trained_model = train(n_bins, x_train, z_train, epochs=20)
    print('training finished')
    val_feats, z_val, val_bins, edges_val = load_data(n_bins, sys.argv[2], no_inf=False)
    x_val = prepare_data(val_feats, scaler=scaler)
    #print(x_val.shape)
    print('predicting')
    preds = predict(trained_model, x_val, 512)

    #classes = preds.argmax(axis=1)
    print('saving')
    output_dir = sys.argv[3]
    np.save(output_dir+'y_pred.npy', preds)
