# TODO implement all the different losses as well as an autoencoder / vae (maybe base use a base class if they both use lots of the same components)
# e.g KL and others ones specified on the notebook 29 - rewatch the video

# TODO get autoencoder / vae paper and comment this

import torch
import torch.functional as F
import torch.nn.init as init

from torch.nn import Sequential, Linear, Conv1d, Conv2d, Conv3d, SiLU, BatchNorm1d, Module
from torch.nn import BCEWithLogitsLoss, MSELoss, KLDivLoss
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam

# Import any other dataset that you want to train on
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

class AutoEncoder(Module):
    def __init__(self, leak=0.15):
        super().__init__()
        num_inputs, num_hidden, num_latents = 784, 400, 64

        # SiLU - Sigmoid linear unit is a smooth function which can help provide non zero gradients
        # It helps maintain gradient flow and the benefit of the smoothness in comparison to the RELU is that the negatives
        # are great with alleviating dead neurons that are infamous with ReLUs
        self.encode = Sequential(
            Sequential(Linear(num_inputs, num_hidden, bias=True), SiLU(), BatchNorm1d(num_hidden)),
            Sequential(Linear(num_hidden, num_hidden, bias=True), SiLU(), BatchNorm1d(num_hidden)),
            Sequential(Linear(num_hidden, num_latents, bias=True), SiLU(), BatchNorm1d(num_latents)),
        )

        self.decode = Sequential(
            Sequential(Linear(num_latents, num_hidden, bias=True), SiLU(), BatchNorm1d(num_hidden)),
            Sequential(Linear(num_hidden, num_hidden, bias=True), SiLU(), BatchNorm1d(num_hidden)),
            # We choose to not have an activation function here at the end of the autoencoder decoding stage so that we don't limit the dynamic range
            # that the autoencoder can output - it will be directly raw pixel values
            Sequential(Linear(num_hidden, num_inputs, bias=True), BatchNorm1d(num_inputs)),
        )

        # initialisse the weights
        self.initialise_weights(leak=leak)

    def initialise_weights(self, leak=0.0):
        # Check that this doesn't just go over sequential
        for module in self.modules():
            if isinstance(module, Linear): 
                init.kaiming_normal_(module.weight, a=leak)

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)
    
class VAE(Module):
    """
    VAEs use approximation methods, here being approximating thed posterior distribution by sampling the mean and the log variance.
    This removes the intractability from the problem where we before would have to sample all of our latents for the denominator of bayes.
    These two are the two networks we define in the centre cvalled self.mean and self.log_variance

    There a re two temrs when training our vae - the expectation of the log of the posterior is the reconstruction loss as it will push the vae
    towards reconstructing the original input from the compressed latent representation.
    The KL Divergence here can be thought of as a regularisation term where it will also try to distribute the latents as a standard normal distribution
    
    Key Advantages that VAEs provide over auatoencoders:
        - VAEs allow you to generate new samples by sampling from the latents - regions of space will be contained separate distributions and so is much more generalised
        - VAEs allow you to control the kind of distribution you want around the latent space as you can use the KL-divergence to control it
    """

    def __init__(self, leak=0.15):
        super().__init__()
        # Make sure that the latents that you use in this matches the size of the UNet
        num_inputs, num_hidden, num_latents = 784, 400, 64 

        # SiLU - Sigmoid linear unit is a smooth function which can help provide non zero gradients
        # It helps maintain gradient flow and the benefit of the smoothness in comparison to the RELU is that the negatives
        # are great with alleviating dead neurons that are infamous with ReLUs
        self.encode = Sequential(
            Sequential(Linear(num_inputs, num_hidden, bias=True), SiLU(), BatchNorm1d(num_hidden)),
            Sequential(Linear(num_hidden, num_hidden, bias=True), SiLU(), BatchNorm1d(num_hidden)),
        )
        
        # TODO read paper and figure out why this is
        # Both the mean and variance sides have no bias and no activation function
        self.mean = Sequential(Linear(num_hidden, num_latents, bias=False), BatchNorm1d(num_latents))
        self.log_variance = Sequential(Linear(num_hidden, num_latents, bias=False), BatchNorm1d(num_latents))

        self.decode = Sequential(
            Sequential(Linear(num_latents, num_hidden, bias=True), SiLU(), BatchNorm1d(num_hidden)),
            Sequential(Linear(num_hidden, num_hidden, bias=True), SiLU(), BatchNorm1d(num_hidden)),
            # We choose to not have an activation function here at the end of the autoencoder decoding stage so that we don't limit the dynamic range
            # that the autoencoder can output - it will be directly raw pixel values
            Sequential(Linear(num_hidden, num_inputs, bias=True), BatchNorm1d(num_inputs)),
        )

        # initialisse the weights
        self.initialise_weights(leak=leak)

    def initialise_weights(self, leak=0.0):
        # Check that this doesn't just go over sequential
        for module in self.modules():
            if isinstance(module, Linear): 
                init.kaiming_normal_(module.weight, a=leak)

    def forward(self, x):
        x = self.encode(x)
        # Here we get the mean and log variance from the vae
        mean, log_variance = self.mean(x), self.log_variance(x)

        # We are now sampling using the calculated values
        # Using the reparameterisation trick to sample - here we calculate the standard deviation (the 0.5 can be inside since its a log)
        std = (0.5 * log_variance).exp()
        z = mean + std * torch.randn_like(log_variance)

        return self.decode(z)

class VAELoss(Module):

    def __init__(self):
        super().__init__()
        
        # Defining the two loss functions that will be used - reconstruction error as well as teh 
        self.bce = BCEWithLogitsLoss()
        self.kl_div = KLDivLoss()
        
    def forward(self, output, image):
        # Combine both losses
        return self.bce(output, image) + self.kl_div(output, image)

if __name__ == "__main__":
    autoencoder = AutoEncoder()
    vae = VAE()
    # Alternatively you can use
    # model.load_state_dict(torch.load("training/saved_models/{model_name}.pth"))

    # Gathering our training and testing dataset using torch.utils.data
    training_dataset = FashionMNIST(root="dataset", train=True, download=True, transform=ToTensor())
    testing_dataset = FashionMNIST(root="dataset", train=False, download=True, transform=ToTensor())
    training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
    testing_dataloader = DataLoader(testing_dataset, batch_size=64, shuffle=True)

    # Use any of the following - MLELoss, KLDivLoss, BCEWithLogitsLoss, VAELoss
    # Here we use binary cross entropy with logits since we are working with grayscale images whree the pixel values are normalised to 0a nd 1
    model = vae
    loss_function = VAELoss()
    optimiser = Adam(model.parameters(), lr=0.001)

    # Simpler pytorch training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for data in training_dataloader:
            image, label = data
            image = image.view(image.size(0), -1)
            
            # Forward pass
            output = model(image)
            loss = loss_function(output, image)

            # Backward pass and optimization
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), "training/saved_models/vae.pth")
