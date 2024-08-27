# TODO implement all the different losses as well as an autoencoder / vae (maybe base use a base class if they both use lots of the same components)
# e.g KL and others ones specified on the notebook 29 - rewatch the video

import torch.functional as F
from torch.nn import Sequential, Linear, Conv1d, Conv2d, Conv3d, SiLU, BatchNorm1d, Module
import torch.nn.init as init

# TODO get autoencoder / vae paper and comment this
class Autoenc(Module):
    def __init__(self):
        super().__init__()
        num_inputs, num_hidden, num_latents = 784, 400, 200 

        # SiLU - Sigmoid linear unit is a smooth function which can help provide non zero gradients
        # It helps maintain gradient flow and the benefit of the smoothness in comparison to the RELU is that the negatives
        # are great with alleviating dead neurons that are infamous with ReLUs
        self.encode = Sequential(
            Sequential(Linear(num_inputs, num_hidden, bias=True), SiLU(), BatchNorm1d(num_hidden)),
            Sequential(Linear(num_hidden, num_hidden, bias=True), SiLU(), BatchNorm1d(num_hidden)),
            Sequential(Linear(num_inputs, num_latents, bias=True), SiLU(), BatchNorm1d(num_latents)),
        )

        self.decode = Sequential(
            Sequential(Linear(num_latents, num_hidden, bias=True), SiLU(), BatchNorm1d(num_hidden)),
            Sequential(Linear(num_hidden, num_hidden, bias=True), SiLU(), BatchNorm1d(num_hidden)),
            # We choose to not have an activation function here at the end of the autoencoder decoding stage so that we don't limit the dynamic range
            # that the autoencoder can output - it will be directly raw pixel values
            Sequential(Linear(num_inputs, num_inputs, bias=True), BatchNorm1d(num_inputs)),
        )

        # initialisse the weights
        self.initialise_weights(leak=0.2)

    def initialise_weights(self, leak=0.0):
        # Check that this doesn't just go over sequential
        for module in self.modules():
            print(module)
            if isinstance(module, (Conv1d, Conv2d, Conv3d, Linear)): 
                init.kaiming_normal_(module.weight, a=leak)

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)

# TODO learn these and comment them - check their papers and add them to the read me
# Space around that point and the space it suses should be 0 mean adn 1 variance which should decode down to that image
# so that image given those latents should decode to that
def kld_loss(inp, x):
    x_hat,mu,lv = inp
    return -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()

# binary cross entropy loss - with logits we dont have to do the softmax
def bce_loss(inp, x): 
    return F.binary_cross_entropy_with_logits(inp[0], x)

# KL divergence - mu and log variance - number that says is mu close to 0 and is lv close to 1
def vae_loss(inp, x): 
    return kld_loss(inp, x) + bce_loss(inp,x)

ae = Autoenc()
