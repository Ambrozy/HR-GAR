import torch
import torch.nn as nn
import time
import numpy as np
import utils_train.log as log


class Trainer(object):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, device):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device
        self.d_out_shape = None
        self.bce_loss = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.gan_lambda = 1e-2

    def __calc_d_out_shape(self, dataloader):
        hr_batch_images = next(iter(dataloader))[1].to(self.device)
        d_out = self.discriminator(hr_batch_images)
        self.d_out_shape = d_out.shape

    def train(self, dataloader, epochs, save_path, save_label, per_epoch_plot=10):
        const_images = next(iter(dataloader))[0][:10].to(self.device)
        history = []
        params = {}
        self.__calc_d_out_shape(dataloader)

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            params = self.train_one_epoch(dataloader)
            history.append(list(params.values()))

            log.save_weights(self.generator, self.discriminator, save_path, save_label)
            if per_epoch_plot > 0 and (epoch + 1) % per_epoch_plot == 0:
                log.plot_results(self.generator, const_images)
            print()

        values = np.array(history).transpose((1, 0))
        return dict(zip(params.keys(), values))

    def train_one_epoch(self, dataloader):
        n_batches = len(dataloader)
        start_time = time.time()
        params = {}

        y_real_full = torch.full(self.d_out_shape, 1., dtype=torch.float, device=self.device)
        y_fake_full = torch.full(self.d_out_shape, 1., dtype=torch.float, device=self.device)

        for batch_idx, (x_images, y_images) in enumerate(dataloader):
            x_images, y_images = x_images.to(self.device), y_images.to(self.device)
            y_real = y_real_full[:len(x_images)]
            y_fake = y_fake_full[:len(x_images)]

            # train discriminator
            y_pred_images = self.generator(x_images)
            d_out = self.discriminator_train_on_batch((y_images, y_real), (y_pred_images.detach(), y_fake))

            # train generator
            y_real = y_real_full[:len(x_images)]
            g_out = self.gan_train_on_batch(y_pred_images, y_images, y_real)

            params = {**d_out, **g_out}
            log.print_log(batch_idx, n_batches, time.time() - start_time, params)
        log.print_log(n_batches, n_batches, time.time() - start_time, params)
        return params

    def discriminator_train_on_batch(self, real, fake):
        x_real, y_real = real
        x_fake, y_fake = fake

        x = torch.vstack([x_real, x_fake])
        y = torch.vstack([y_real, y_fake])

        self.discriminator.zero_grad()
        output = self.discriminator(x)
        loss, losses = self.discriminator_loss(y, output)
        loss.backward()
        self.d_optimizer.step()

        metrics = self.discriminator_metrics(y, output)
        return {**losses, **metrics}

    def discriminator_loss(self, y_true, y_pred):
        loss = self.bce_loss(y_pred, y_true)
        return loss, {'d_loss': loss.item()}

    def discriminator_metrics(self, y_true, y_pred):
        return {}

    def gan_train_on_batch(self, y_pred_images, y_images, y_dis):
        self.generator.zero_grad()
        output = self.discriminator(y_pred_images)
        loss, losses = self.generator_loss(y_images, y_pred_images, y_dis, output)
        loss.backward()
        self.g_optimizer.step()

        metrics = self.generator_metrics(y_images, y_pred_images, y_dis, output)
        return {**losses, **metrics}

    def generator_loss(self, y_img_true, y_img_pred, y_dis_true, y_dis_pred):
        loss1 = self.mse(y_img_pred, y_img_true)
        loss2 = self.bce_loss(y_dis_pred, y_dis_true)
        loss = loss1 + self.gan_lambda * loss2
        return loss, {'loss': loss.item(), 'img_loss': loss1.item(), 'g_loss': loss2.item()}

    def generator_metrics(self, y_img_true, y_img_pred, y_dis_true, y_dis_pred):
        return {}
