from .. import jax_metrics as j_metrics
from flax import nn, optim, serialization, jax_utils
import jax.random as rand
import jax.numpy as jnp
import jax
import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
import h5py
import tensorflow as tf

def get_classifier(n_features, n_bins):
    class AutoLSTM(nn.Module):
        def apply(self, x):
            batch_size = x.shape[0]
            net = nn.Conv(x, features=32, kernel_size=(5,), padding='SAME')
            net = nn.leaky_relu(net)
            net = nn.Conv(net, 512, kernel_size=(5,), padding='SAME')
            net = nn.leaky_relu(net)
            net = nn.Conv(net, 128, kernel_size=(5,), padding='SAME')
            net = nn.leaky_relu(net)

            carry = nn.LSTMCell.initialize_carry(rand.PRNGKey(0), (batch_size,), 128)
            new_carry, output_1 = jax_utils.scan_in_dim(nn.LSTMCell.partial(), carry, net, axis=1)
            _, output_2 = jax_utils.scan_in_dim(nn.LSTMCell.partial(), new_carry, output_1, axis=1)

            net_dense = output_2.reshape((output_2.shape[0], -1))
            net_dense = nn.Dense(net_dense, 512)
            net_dense = nn.BatchNorm(net_dense)
            net_dense = nn.leaky_relu(net_dense)

            net_dense = nn.dropout(net_dense, 0.25)

            net_dense = nn.dropout(net_dense, 0.5)
            return nn.softmax(nn.Dense(net_dense, n_bins))

    with nn.stochastic(rand.PRNGKey(0)):
        _, initial_params = AutoLSTM.init_by_shape(rand.PRNGKey(0), [((1, n_features, 1), jnp.float32)])
    return nn.Model(AutoLSTM, initial_params)


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



def train(n_features, n_bins, training_data, training_z, batch_size=512, epochs=20, lr=0.001):
    model = get_classifier(n_features, n_bins)
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

def predict(model, data, batch_size=512):

    batches = get_batches(data, 0., batch_size, train=False)
    preds = []
    n_batches = data.shape[0] // batch_size
    with nn.stochastic(rand.PRNGKey(0)):
        for i, x_test in enumerate(batches.as_numpy_iterator()):
            print('batch {} of {}'.format(i, n_batches))
            p = model(x_test)
            preds.append(p)
    result = np.concatenate([p for p in preds], axis=0)
    return result

def load_data(n_bins, fname, no_inf=True, take_colors=True, cutoff=0.0):

    #print(fname)
    #sys.exit()
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
        all_mags[bad] = 30
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
    if scaler == None:
        scaler = MinMaxScaler()
        result = scaler.fit_transform(data)
        result = np.expand_dims(result, axis=-1)
        return result, scaler
    else:
        result = scaler.transform(data)
        result = np.expand_dims(result, axis=-1)
        return result


if __name__ == '__main__':
    n_bins = int(sys.argv[4])
    n_features = 7
    print('preparing data')
    feats_train, z_train, bins_train, edges_train = load_data(n_bins, sys.argv[1], no_inf=False)

    x_train, scaler = prepare_data(feats_train)

    print('training')
    losses, trained_model = train(n_features, n_bins, x_train, z_train, epochs=20)
    print('training finished')

    val_feats, z_val, bins_val, edges_val = load_data(n_bins, sys.argv[2], no_inf=False)
    x_val = prepare_data(val_feats, scaler=scaler)
    print(x_val.shape)
    print('predicting')
    preds = predict(trained_model, x_val, 512)

    #classes = preds.argmax(axis=1)
    print('saving')
    output_dir = sys.argv[3]
    np.save(output_dir+'y_pred.npy', preds)
