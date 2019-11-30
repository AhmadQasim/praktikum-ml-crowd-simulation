"""
Implementation of a Variational Auto-encoder in PyTorch
"""

import numpy as np
import fire
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch.distributions.normal import Normal
from multiprocessing import set_start_method
from torch.nn import functional as F

torch.set_default_tensor_type(torch.cuda.FloatTensor)

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class VAE:
    def __init__(self):
        self.latent_vector_size = 32
        self.print_on_epoch = [1, 5, 25, 50]

        self.model = Model(self.latent_vector_size)
        self.batch_size = 128
        self.test_count = 16
        self.classes = 10

        self.train_mnist_dataloader = None
        self.test_mnist_dataloader = None
        self.mnist_epochs = 5
        self.learning_rate = 0.001
        self.model_opt = Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.generated_loss = torch.nn.BCELoss(reduction="sum")
        self.dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        self.model_path = 'models/vae.hdf5'

    def load_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        train_set = MNIST(root='./mnist', train=True, download=True, transform=transform)
        self.train_mnist_dataloader = torch.utils.data.DataLoader(train_set,
                                                                  batch_size=self.batch_size,
                                                                  shuffle=True, num_workers=0)

        test_set = MNIST(root='./mnist', train=False, download=True, transform=transform)
        self.test_mnist_dataloader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size,
                                                                 shuffle=False, num_workers=0)

    def sample_and_save_image(self, epoch_num):
        sample_vector = torch.randn(self.test_count, self.latent_vector_size)
        img = self.model.decode(sample_vector)
        self.plot_grid(img, epoch_num, "generated")

        torch.save(self.model.state_dict(), self.model_path)

    def generate_and_plot_results(self, epoch_num):
        real, _ = next(iter(self.test_mnist_dataloader))
        real = real[0:self.test_count, :]
        real = real.cuda().squeeze().view(-1, 784)
        with torch.no_grad():
            mean, logvar = self.model.encode(real)
            latent_vector = self.reparametrize(mean, logvar)
            generated = self.model.decode(latent_vector)

        self.plot_grid(generated, epoch_num, "reconstructed")
        self.plot_grid(real, epoch_num, "real")

    def plot_grid(self, images, epoch_num, t):
        fig = plt.figure(figsize=(28, 28))
        columns = np.sqrt(self.test_count)
        rows = np.sqrt(self.test_count)
        images = images.view(-1, 28, 28)
        images = images.cpu().detach().numpy()
        images = images * 255.0
        for i in range(1, int(columns) * int(rows) + 1):
            ax = fig.add_subplot(rows, columns, i)
            ax.axis('off')
            plt.imshow(images[i - 1], cmap="gray_r")
        plt.suptitle("The " + t + " images at Epoch: " + str(epoch_num) + " and latent dimensions: " +
                     str(self.latent_vector_size), fontsize=40)
        plt.savefig("plots/vae/" + t + "_epochs" + str(epoch_num) + "_latent" + str(self.latent_vector_size))

    @staticmethod
    def reparametrize(mean, logvar):
        std = torch.exp(0.5 * logvar)

        # sample unit gaussian vector
        sample_vector = torch.randn_like(std)
        latent_vector = (sample_vector * std) + mean

        return latent_vector

    def train(self):
        elbo_loss_log = []

        self.load_data()
        self.model.train()
        for epoch in range(self.mnist_epochs):

            for i, data in enumerate(self.train_mnist_dataloader, 0):
                real, real_labels = data
                real = real.cuda().squeeze().view(-1, 784)

                # run encoder
                self.model_opt.zero_grad()
                mean, logvar = self.model.encode(real)
                latent_vector = self.reparametrize(mean, logvar)

                # calculate loss
                generated_loss = self.generated_loss(self.model.decode(latent_vector), real)
                latent_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                total_loss = generated_loss + latent_loss
                total_loss.backward()

                # optimize
                self.model_opt.step()

            # print results
            print("Epoch: ", epoch + 1, " Loss: ", total_loss)

            if epoch + 1 in self.print_on_epoch:
                self.plot_latent_rep(epoch_num=epoch+1)
                self.generate_and_plot_results(epoch_num=epoch+1)
                self.sample_and_save_image(epoch_num=epoch+1)

            elbo_loss_log.append(-self.elbo_test_loss())

        plt.figure()
        plt.plot(range(self.mnist_epochs), elbo_loss_log)
        plt.xlabel('Epochs')
        plt.ylabel('-Loss(ELBO)')
        plt.title('The plot of elbo loss (test set) with latent dim ' + str(self.latent_vector_size) +
                  "and latent dimensions: " + str(self.latent_vector_size))
        plt.savefig('plots/vae/loss_elbo_latent' + str(self.latent_vector_size))

        print('Finished Training')

    def elbo_test_loss(self):
        total_elbo_loss = 0

        for i, data in enumerate(self.test_mnist_dataloader, 0):
            real, real_labels = data
            real = real.cuda().squeeze().view(-1, 784)

            with torch.no_grad():

                # run encoder
                mean, logvar = self.model.encode(real)

                # calculate loss
                latent_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

                total_elbo_loss += latent_loss

        return total_elbo_loss

    def plot_latent_rep(self, epoch_num):
        plt.figure()
        for i, data in enumerate(self.test_mnist_dataloader, 0):
            real, real_labels = data
            real = real.cuda().squeeze().view(-1, 784)

            for real_label in torch.unique(real_labels):
                real_mask = real_labels[real_labels == real_label]
                real_imgs = real[real_mask]

                with torch.no_grad():
                    mean, logvar = self.model.encode(real_imgs)
                    latent_vector = self.reparametrize(mean, logvar)

                plt.scatter(latent_vector[:, 0].cpu().numpy(),
                            latent_vector[:, 1].cpu().numpy())

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("The latent representation at Epoch: " + str(epoch_num) + " and latent dimensions: " +
                  str(self.latent_vector_size))
        plt.savefig("plots/vae/" + "latent_epoch" + str(epoch_num) + "_latent" + str(self.latent_vector_size))


class Model(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Model, self).__init__()

        self.fc1_encoder = torch.nn.Linear(784, 256)
        self.fc2_encoder = torch.nn.Linear(256, 256)
        self.fc_mean_encoder = torch.nn.Linear(256, latent_dim)
        self.fc_logvar_encoder = torch.nn.Linear(256, latent_dim)

        self.fc1_decoder = torch.nn.Linear(latent_dim, 256)
        self.fc2_decoder = torch.nn.Linear(256, 256)
        self.fc_output = torch.nn.Linear(256, 784)

    def encode(self, x):
        x = F.relu(self.fc1_encoder(x))
        x = F.relu(self.fc2_encoder(x))
        mean = self.fc_mean_encoder(x)
        logvar = self.fc_logvar_encoder(x)

        return mean, logvar

    def decode(self, x):
        x = F.relu(self.fc1_decoder(x))
        x = F.relu(self.fc2_decoder(x))
        x = torch.sigmoid(self.fc_output(x))

        return x


# wrapping to avoid Windows 10 error
def main():
    fire.Fire(VAE)


if __name__ == "__main__":
    main()
