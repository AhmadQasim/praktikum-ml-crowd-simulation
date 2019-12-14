"""
Implementation of a Variational Auto-encoder in PyTorch
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.nn import functional as F

torch.set_default_tensor_type(torch.cuda.FloatTensor)


class VAE:
    def __init__(self,
                 latent_vector_size,
                 print_output,
                 batch_size,
                 learning_rate,
                 epochs,
                 train_dataloader,
                 test_dataloader,
                 dataset_dims,
                 test_count,
                 kl_annealing,
                 beta_limit,
                 reconstruction_loss_scale):

        # logging
        self.print_on_epoch = [1, 5, 25, 50]
        self.print_output = print_output

        # dataset related parameters
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.batch_size = batch_size
        self.test_count = test_count
        self.dataset_dims = dataset_dims

        # model training
        self.latent_vector_size = latent_vector_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta_limit = beta_limit
        self.reconstruction_loss_scale = reconstruction_loss_scale
        self.generated_loss = torch.nn.MSELoss(reduction="sum")

        # prior distribution gaussian
        self.dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        # intializing the model
        self.model = Model(self.latent_vector_size, self.dataset_dims)
        self.model_opt = Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.model_path = 'models/task4.hdf5'

        # KL annealing
        self.kl_annealing = kl_annealing
        self.beta = 0
        self.annealed_epochs = int(self.epochs / 5)
        self.beta_scaling = torch.linspace(0, beta_limit, self.epochs - self.annealed_epochs)

    @staticmethod
    def reparametrize(mean, logvar):

        """
        Perform Re-parametrization

        :param logvar: the logvar value
        :param mean: the mean value

        :return: reparametrized latent vector object
        """

        std = torch.exp(0.5 * logvar)

        # sample unit gaussian vector
        sample_vector = torch.randn_like(std)
        latent_vector = (sample_vector * std) + mean

        return latent_vector

    def train(self):

        """
        Train the VAE model

        :return: the trained VAE model object
        """

        elbo_loss_log = []

        self.model.train()
        for epoch in range(self.epochs):

            if epoch > self.annealed_epochs and self.kl_annealing:
                self.beta = self.beta_scaling[epoch - self.annealed_epochs]
            else:
                self.beta = 1

            for i, data in enumerate(self.train_dataloader, 0):
                real, _ = data
                real = real.cuda().squeeze().view(-1, self.dataset_dims)

                # run encoder
                self.model_opt.zero_grad()
                mean, logvar = self.model.encode(real)
                latent_vector = self.reparametrize(mean, logvar)

                # calculate loss
                generated_loss = self.generated_loss(self.model.decode(latent_vector), real) * \
                                 self.reconstruction_loss_scale
                latent_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                total_loss = generated_loss + (self.beta * latent_loss)
                total_loss.backward()

                # optimize
                self.model_opt.step()

            # print results
            print("Latent Loss: ", latent_loss)
            print("Epoch: ", epoch + 1, " Loss: ", total_loss)

            if epoch + 1 in self.print_on_epoch and self.print_output:
                self.plot_latent_rep(epoch_num=epoch+1)
                self.reconstruct_and_plot_results(epoch_num=epoch + 1)
                self.generate_and_plot_results(epoch_num=epoch + 1)

            elbo_loss_log.append(-self.calculate_total_elbo_loss())

        fig = plt.figure()
        plt.plot(range(self.epochs), elbo_loss_log)
        plt.xlabel('Epochs')
        plt.ylabel('-Loss(ELBO)')
        plt.savefig('plots/task3/loss_elbo_latent' + str(self.latent_vector_size))
        plt.close(fig)

        torch.save(self.model.state_dict(), self.model_path)
        print('Finished Training')

        return self.model

    def test(self):

        """
        Reconstruct data using the VAE model from test set

        :return: the reconstructed data converted from pytorch tensor to numpy array
        """

        reconstructed_data = torch.zeros(size=(1, self.dataset_dims))

        for i, data in enumerate(self.test_dataloader, 0):
            real, real_labels = data
            real = real.cuda().squeeze().view(-1, self.dataset_dims)

            with torch.no_grad():

                mean, logvar = self.model.encode(real)
                latent_vector = self.reparametrize(mean, logvar)

                generated = self.model.decode(latent_vector)
                reconstructed_data = torch.cat([reconstructed_data, generated], dim=0)

        return reconstructed_data[1:, :].cpu().detach().numpy()

    def generate(self, count=None):

        """
        Generate data using the trained VAE model

        :param count: the number of data points to be generated, if None given then the method uses
        self.test_count parameter to determine the number
        :return: the generated data
        """

        if count is None:
            sample_vector = torch.randn(self.test_count, self.latent_vector_size)
        else:
            sample_vector = torch.randn(count, self.latent_vector_size)
        data = self.model.decode(sample_vector)

        return data

    def generate_and_plot_results(self, epoch_num):

        """
        Generate data using the trained VAE model and plot the results

        :param epoch_num: the number of current epoch during the training phase,
        if/at which the generation is being done
        :return: null
        """

        img = self.generate()
        self.plot_grid(img, epoch_num, "generated")

    def reconstruct_and_plot_results(self, epoch_num):

        """
        Reconstruct data using the trained VAE model and plot the results

        :param epoch_num: the number of current epoch during the training phase,
        if/at which the reconstruction is being done
        :return: null
        """

        real, _ = next(iter(self.test_dataloader))
        real = real[0:self.test_count, :]
        real = real.cuda().squeeze().view(-1, self.dataset_dims)
        with torch.no_grad():
            mean, logvar = self.model.encode(real)
            latent_vector = self.reparametrize(mean, logvar)
            generated = self.model.decode(latent_vector)

        self.plot_grid(generated, epoch_num, "reconstructed")
        self.plot_grid(real, epoch_num, "real")

    def plot_grid(self, images, epoch_num, t):

        """
        The main plotting function for plotting a grid of output results

        :param epoch_num: the number of current epoch during the training phase,
        if/at which the the results are being plotted
        :param images: the images to be plotted
        :param t: the mode i.e. reconstruction, generation
        :return: null
        """

        fig = plt.figure(figsize=(5, 5))
        columns = np.sqrt(self.test_count)
        rows = np.sqrt(self.test_count)
        images = images.view(-1, 28, 28)
        images = images.cpu().detach().numpy()
        images = images * 255.0
        for i in range(1, int(columns) * int(rows) + 1):
            ax = fig.add_subplot(rows, columns, i)
            ax.axis('off')
            plt.imshow(images[i - 1], cmap="gray_r")
        plt.grid()
        plt.savefig("plots/task3/" + t + "_epochs" + str(epoch_num) + "_latent" + str(self.latent_vector_size))
        plt.close(fig)

    def calculate_total_elbo_loss(self):

        """
        Calculate the ELBO loss

        :return: the total elbo loss for the test set
        """

        total_elbo_loss = 0

        for i, data in enumerate(self.test_dataloader, 0):
            real, _ = data
            real = real.cuda().squeeze().view(-1, self.dataset_dims)

            with torch.no_grad():

                # run encoder
                mean, logvar = self.model.encode(real)
                latent_vector = self.reparametrize(mean, logvar)

                # calculate loss
                generated_loss = self.generated_loss(self.model.decode(latent_vector), real)
                latent_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

                total_elbo_loss += (latent_loss + generated_loss)

        print("ELBO loss", total_elbo_loss)

        return total_elbo_loss

    def plot_latent_rep(self, epoch_num):

        """
        Plot the latent representation of the first two dimensions of latent space

        :return: null
        """

        fig = plt.figure()
        for i, data in enumerate(self.test_dataloader, 0):
            real, real_labels = data
            real = real.cuda().squeeze().view(-1, self.dataset_dims)

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
        plt.savefig("plots/task3/" + "latent_epoch" + str(epoch_num) + "_latent" + str(self.latent_vector_size))
        plt.close(fig)


class Model(torch.nn.Module):
    def __init__(self, latent_dim, dataset_dims):
        super(Model, self).__init__()

        self.dataset_dims = dataset_dims

        self.fc1_encoder = torch.nn.Linear(self.dataset_dims, 256)
        self.fc2_encoder = torch.nn.Linear(256, 256)
        self.fc_mean_encoder = torch.nn.Linear(256, latent_dim)
        self.fc_logvar_encoder = torch.nn.Linear(256, latent_dim)

        self.fc1_decoder = torch.nn.Linear(latent_dim, 256)
        self.fc2_decoder = torch.nn.Linear(256, 256)
        self.fc_output = torch.nn.Linear(256, self.dataset_dims)

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
