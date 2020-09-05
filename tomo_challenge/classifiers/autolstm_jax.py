import tomo_challenge.jax_metrics as j_metrics
from flax import nn, optim, serialization, jax_utils
import jax.random as rand
import jax.numpy as jnp
import jax

class AutoLSTM(nn.Module):
    def __init__(self, n_bins):
        self.bins = n_bins
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
        return nn.softmax(nn.Dense(gap, self.bins))



def get_classifier(n_features, n_bins):
    with nn.stochastic(rand.PRNGKey(0)):
        _, initial_params = AutoLSTM(n_bins).init_by_shape( rand.PRNGKey(1), [((1, n_features,1), jnp.float32)])
    return nn.Model(AutoLSTM(n_bins), initial_params)

def get_batches(features, redshift, batch_size, train=True):
    if train:
        dataset = tf.data.Dataset.from_tensor_slices((features, redshift))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = dataset.shuffle(buffer_size=2048).batch(batch_size)
    return dataset

def train(n_features, n_bins, training_data, training_z, batch_size=20, epochs=20, lr=0.001):
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

def predict(model, data, batch_size):
    batches = get_batches(data, 0., train=False)
    preds = []
    with nn.stochastic(rand.PRNGKey(0)):
        for i, x_test in enumerate(batches.as_numpy_iterator()):
            p = model(x_test)
            preds.append(p)
        result = np.concatenate([p for p in preds], axis=0)
    return result
