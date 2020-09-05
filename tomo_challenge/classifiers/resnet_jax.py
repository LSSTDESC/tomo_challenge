import tomo_challenge.jax_metrics as j_metrics
from flax import nn, optim, serialization, jax_utils
import jax.random as rand
import jax.numpy as jnp
import jax

class Resnet(nn.Module):
    def __init__(self, n_bins):
        self.bins = n_bins
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
        return nn.softmax(nn.Dense(gap, self.bins))





def get_batches(features, redshift, batch_size, train=True):
    if train:
        dataset = tf.data.Dataset.from_tensor_slices((features, redshift))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = dataset.shuffle(buffer_size=2048).batch(batch_size)
    return dataset

def train(resnet,n_features, n_bins, training_data, training_z, batch_size=20, epochs=20, lr=0.001):
    def get_classifier(n_features, n_bins):
        _, initial_params = resnet.init_by_shape(rand.PRNGKey(0), [((1, n_features, 1), jnp.float32)])
        return nn.Model(resnet, initial_params)

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
    for i, x_test in enumerate(batches.as_numpy_iterator()):
        p = model(x_test)
        preds.append(p)
    result = np.concatenate([p for p in preds], axis=0)
    return result
