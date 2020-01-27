import argparse
import gc
import logging
import os
import sys
import time
os.chdir('../..')
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from data.loader import data_loader
from losses import gan_g_loss, gan_d_loss, l2_loss
from losses import displacement_error, final_displacement_error

from networks.Generator import Generator
from networks.Discriminator import Discriminator
from utils import int_tuple, bool_flag, get_total_norm
from utils import relative_to_abs, get_dset_path

torch.backends.cudnn.benchmark = True

class Trainer:
    def __init__(self):
        self.dataset_name = "eth"
        self.delim = "tab"
        self.obs_len = 8
        self.pred_len = 12
        self.skip = 1

        self.batch_size = 64
        self.num_iterations = 10000
        self.num_epochs = 200

        self.embedding_dim = 64
        self.num_layers = 1
        self.dropout = 0.0
        self.batch_norm = 0
        self.ffnn_dim = 1024

        self.encoder_h_dim_g = 64
        self.decoder_h_dim_g = 128
        self.noise_dim = (0, )
        self.noise_type = "gaussian"
        self.noise_mix_type = "ped"
        self.clipping_threshold_g = 0
        self.g_learning_rate = 5e-4
        self.g_steps = 1

        self.pooling_type = "pool_net"
        self.pool_every_timestep = 1

        self.bottleneck_dim = 1024

        self.neighborhood_size = 2.0
        self.grid_size = 8

        self.d_type = "local"
        self.encoder_h_dim_d = 64
        self.d_learning_rate = 5e-4
        self.d_steps = 2
        self.clipping_threshold_d = 0

        self.l2_loss_weight = 0
        self.best_k = 1

        self.output_dir = "./"
        self.print_every = 25
        self.checkpoint_every = 100
        self.checkpoint_name = "checkpoint"
        self.checkpoint_start_from = None
        self.restore_from_checkpoint = 1
        self.num_samples_check = 5000

        self.use_gpu = 1
        self.timing = 0
        self.gpu_num = "0"

        FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
        logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
        self.logger = logging.getLogger(__name__)

    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight)

    def get_dtypes(self):
        long_dtype = torch.LongTensor
        float_dtype = torch.FloatTensor
        if self.use_gpu == 1:
            long_dtype = torch.cuda.LongTensor
            float_dtype = torch.cuda.FloatTensor
        return long_dtype, float_dtype

    def discriminator_step(self, batch, generator, discriminator, d_loss_fn, optimizer_d):
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch
        losses = {}
        loss = torch.zeros(1).to(pred_traj_gt)

        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
        traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

        scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
        scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

        data_loss = d_loss_fn(scores_real, scores_fake)
        losses['D_data_loss'] = data_loss.item()
        loss += data_loss
        losses['D_total_loss'] = loss.item()

        optimizer_d.zero_grad()
        loss.backward()
        if self.clipping_threshold_d > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(), self.clipping_threshold_d)
        optimizer_d.step()

        return losses

    def generator_step(self, batch, generator, discriminator, g_loss_fn, optimizer_g):
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, seq_start_end) = batch
        losses = {}
        loss = torch.zeros(1).to(pred_traj_gt)
        g_l2_loss_rel = []

        loss_mask = loss_mask[:, self.obs_len:]

        for _ in range(self.best_k):
            generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

            pred_traj_fake_rel = generator_out
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            if self.l2_loss_weight > 0:
                g_l2_loss_rel.append(self.l2_loss_weight * l2_loss(
                    pred_traj_fake_rel,
                    pred_traj_gt_rel,
                    loss_mask,
                    mode='raw'))

        g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
        if self.l2_loss_weight > 0:
            g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
            for start, end in seq_start_end.data:
                _g_l2_loss_rel = g_l2_loss_rel[start:end]
                _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
                _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                    loss_mask[start:end])
                g_l2_loss_sum_rel += _g_l2_loss_rel
            losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
            loss += g_l2_loss_sum_rel

        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

        scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
        discriminator_loss = g_loss_fn(scores_fake)

        loss += discriminator_loss
        losses['G_discriminator_loss'] = discriminator_loss.item()
        losses['G_total_loss'] = loss.item()

        optimizer_g.zero_grad()
        loss.backward()
        if self.clipping_threshold_g > 0:
            nn.utils.clip_grad_norm_(
                generator.parameters(), self.clipping_threshold_g
            )
        optimizer_g.step()

        return losses

    def check_accuracy(self, loader, generator, discriminator, d_loss_fn, limit=False):
        d_losses = []
        metrics = {}
        g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
        disp_error, disp_error_l, disp_error_nl = ([],) * 3
        f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
        total_traj, total_traj_l, total_traj_nl = 0, 0, 0
        loss_mask_sum = 0
        generator.eval()
        with torch.no_grad():
            for batch in loader:
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                 non_linear_ped, loss_mask, seq_start_end) = batch
                linear_ped = 1 - non_linear_ped
                loss_mask = loss_mask[:, self.obs_len:]

                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

                g_l2_loss_abs, g_l2_loss_rel = self.cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                                                                  pred_traj_fake_rel, loss_mask)
                ade, ade_l, ade_nl = self.cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped)
                fde, fde_l, fde_nl = self.cal_fde(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped)

                traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
                traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
                traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
                traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

                scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
                scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

                d_loss = d_loss_fn(scores_real, scores_fake)
                d_losses.append(d_loss.item())

                g_l2_losses_abs.append(g_l2_loss_abs.item())
                g_l2_losses_rel.append(g_l2_loss_rel.item())
                disp_error.append(ade.item())
                disp_error_l.append(ade_l.item())
                disp_error_nl.append(ade_nl.item())
                f_disp_error.append(fde.item())
                f_disp_error_l.append(fde_l.item())
                f_disp_error_nl.append(fde_nl.item())

                loss_mask_sum += torch.numel(loss_mask.data)
                total_traj += pred_traj_gt.size(1)
                total_traj_l += torch.sum(linear_ped).item()
                total_traj_nl += torch.sum(non_linear_ped).item()
                if limit and total_traj >= self.num_samples_check:
                    break

        metrics['d_loss'] = sum(d_losses) / len(d_losses)
        metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
        metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

        metrics['ade'] = sum(disp_error) / (total_traj * self.pred_len)
        metrics['fde'] = sum(f_disp_error) / total_traj
        if total_traj_l != 0:
            metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * self.pred_len)
            metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
        else:
            metrics['ade_l'] = 0
            metrics['fde_l'] = 0
        if total_traj_nl != 0:
            metrics['ade_nl'] = sum(disp_error_nl) / (
                    total_traj_nl * self.pred_len)
            metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
        else:
            metrics['ade_nl'] = 0
            metrics['fde_nl'] = 0

        generator.train()
        return metrics

    @staticmethod
    def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel, loss_mask):
        g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode='sum')
        g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum')

        return g_l2_loss_abs, g_l2_loss_rel

    @staticmethod
    def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
        ade = displacement_error(pred_traj_fake, pred_traj_gt)
        ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
        ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)

        return ade, ade_l, ade_nl

    @staticmethod
    def cal_fde(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
        fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
        fde_l = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], linear_ped)
        fde_nl = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped)

        return fde, fde_l, fde_nl

    def main(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_num
        train_path = get_dset_path(self.dataset_name, 'train')
        val_path = get_dset_path(self.dataset_name, 'val')

        long_dtype, float_dtype = self.get_dtypes(self)

        self.logger.info("Initializing train dataset")
        train_dset, train_loader = data_loader(self, train_path)
        self.logger.info("Initializing val dataset")
        _, val_loader = data_loader(self, val_path)

        iterations_per_epoch = len(train_dset) / self.batch_size / self.d_steps
        if self.num_epochs:
            self.num_iterations = int(iterations_per_epoch * self.num_epochs)

        self.logger.info(
            'There are {} iterations per epoch'.format(iterations_per_epoch)
        )

        generator = Generator(
            pred_len=self.pred_len,
            embedding_dim=self.embedding_dim,
            encoder_h_dim=self.encoder_h_dim_g,
            decoder_h_dim=self.decoder_h_dim_g,
            mlp_dim=self.ffnn_dim,
            num_layers=self.num_layers,
            noise_dim=self.noise_dim,
            noise_type=self.noise_type,
            noise_mix_type=self.noise_mix_type,
            pooling_type=self.pooling_type,
            pool_every_timestep=self.pool_every_timestep,
            dropout=self.dropout,
            bottleneck_dim=self.bottleneck_dim,
            neighborhood_size=self.neighborhood_size,
            grid_size=self.grid_size,
            batch_norm=self.batch_norm)

        generator.apply(self.init_weights)
        generator.type(float_dtype).train()
        # self.logger.info('Here is the generator:')
        # self.logger.info(generator)

        discriminator = Discriminator(
            self.embedding_dim,
            self.encoder_h_dim_d,
        )

        discriminator.apply(self.init_weights)
        discriminator.type(float_dtype).train()
        # self.logger.info('Here is the discriminator:')
        # self.logger.info(discriminator)

        g_loss_fn = gan_g_loss
        d_loss_fn = gan_d_loss

        optimizer_g = optim.Adam(generator.parameters(), lr=self.g_learning_rate)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=self.d_learning_rate)

        # Maybe restore from checkpoint
        restore_path = None
        if self.restore_from_checkpoint == 1:
            restore_path = os.path.join(self.output_dir,
                                        '%s_with_model.pt' % self.checkpoint_name)

        if restore_path is not None and os.path.isfile(restore_path):
            self.logger.info('Restoring from checkpoint {}'.format(restore_path))
            checkpoint = torch.load(restore_path)
            generator.load_state_dict(checkpoint['g_state'])
            discriminator.load_state_dict(checkpoint['d_state'])
            optimizer_g.load_state_dict(checkpoint['g_optim_state'])
            optimizer_d.load_state_dict(checkpoint['d_optim_state'])
            t = checkpoint['counters']['t']
            epoch = checkpoint['counters']['epoch']
            checkpoint['restore_ts'].append(t)
        else:
            # Starting from scratch, so initialize checkpoint data structure
            t, epoch = 0, 0
            checkpoint = {
                'self': self.__dict__,
                'G_losses': defaultdict(list),
                'D_losses': defaultdict(list),
                'losses_ts': [],
                'metrics_val': defaultdict(list),
                'metrics_train': defaultdict(list),
                'sample_ts': [],
                'restore_ts': [],
                'norm_g': [],
                'norm_d': [],
                'counters': {
                    't': None,
                    'epoch': None,
                },
                'g_state': None,
                'g_optim_state': None,
                'd_state': None,
                'd_optim_state': None,
                'g_best_state': None,
                'd_best_state': None,
                'best_t': None,
                'g_best_nl_state': None,
                'd_best_state_nl': None,
                'best_t_nl': None,
            }
        t0 = None
        while t < self.num_iterations:
            gc.collect()
            d_steps_left = self.d_steps
            g_steps_left = self.g_steps
            epoch += 1
            self.logger.info('Starting epoch {}'.format(epoch))
            for batch in train_loader:
                if d_steps_left > 0:
                    step_type = 'd'
                    losses_d = self.discriminator_step(self, batch, generator,
                                                       discriminator, d_loss_fn,
                                                       optimizer_d)
                    checkpoint['norm_d'].append(
                        get_total_norm(discriminator.parameters()))
                    d_steps_left -= 1
                elif g_steps_left > 0:
                    step_type = 'g'
                    losses_g = self.generator_step(self, batch, generator,
                                                   discriminator, g_loss_fn,
                                                   optimizer_g)
                    checkpoint['norm_g'].append(
                        get_total_norm(generator.parameters())
                    )
                    g_steps_left -= 1

                if d_steps_left > 0 or g_steps_left > 0:
                    continue

                if t % self.print_every == 0:
                    self.logger.info('t = {} / {}'.format(t + 1, self.num_iterations))
                    for k, v in sorted(losses_d.items()):
                        self.logger.info('  [D] {}: {:.3f}'.format(k, v))
                        checkpoint['D_losses'][k].append(v)
                    for k, v in sorted(losses_g.items()):
                        self.logger.info('  [G] {}: {:.3f}'.format(k, v))
                        checkpoint['G_losses'][k].append(v)
                    checkpoint['losses_ts'].append(t)

                if t > 0 and t % self.checkpoint_every == 0:
                    checkpoint['counters']['t'] = t
                    checkpoint['counters']['epoch'] = epoch
                    checkpoint['sample_ts'].append(t)

                    # Check stats on the validation set
                    self.logger.info('Checking stats on val ...')
                    metrics_val = self.check_accuracy(val_loader, generator, discriminator, d_loss_fn)
                    self.logger.info('Checking stats on train ...')
                    metrics_train = self.check_accuracy(train_loader, generator, discriminator, d_loss_fn, limit=True)

                    for k, v in sorted(metrics_val.items()):
                        self.logger.info('  [val] {}: {:.3f}'.format(k, v))
                        checkpoint['metrics_val'][k].append(v)
                    for k, v in sorted(metrics_train.items()):
                        self.logger.info('  [train] {}: {:.3f}'.format(k, v))
                        checkpoint['metrics_train'][k].append(v)

                    min_ade = min(checkpoint['metrics_val']['ade'])
                    min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                    if metrics_val['ade'] == min_ade:
                        self.logger.info('New low for avg_disp_error')
                        checkpoint['best_t'] = t
                        checkpoint['g_best_state'] = generator.state_dict()
                        checkpoint['d_best_state'] = discriminator.state_dict()

                    if metrics_val['ade_nl'] == min_ade_nl:
                        self.logger.info('New low for avg_disp_error_nl')
                        checkpoint['best_t_nl'] = t
                        checkpoint['g_best_nl_state'] = generator.state_dict()
                        checkpoint['d_best_nl_state'] = discriminator.state_dict()

                    # Save another checkpoint with model weights and
                    # optimizer state
                    checkpoint['g_state'] = generator.state_dict()
                    checkpoint['g_optim_state'] = optimizer_g.state_dict()
                    checkpoint['d_state'] = discriminator.state_dict()
                    checkpoint['d_optim_state'] = optimizer_d.state_dict()
                    checkpoint_path = os.path.join(
                        self.output_dir, '%s_with_model.pt' % self.checkpoint_name
                    )
                    self.logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                    torch.save(checkpoint, checkpoint_path)
                    self.logger.info('Done.')

                    # Save a checkpoint with no model weights by making a shallow
                    # copy of the checkpoint excluding some items
                    checkpoint_path = os.path.join(
                        self.output_dir, '%s_no_model.pt' % self.checkpoint_name)
                    self.logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                    key_blacklist = [
                        'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                        'g_optim_state', 'd_optim_state', 'd_best_state',
                        'd_best_nl_state'
                    ]
                    small_checkpoint = {}
                    for k, v in checkpoint.items():
                        if k not in key_blacklist:
                            small_checkpoint[k] = v
                    torch.save(small_checkpoint, checkpoint_path)
                    self.logger.info('Done.')

                t += 1
                d_steps_left = self.d_steps
                g_steps_left = self.g_steps
                if t >= self.num_iterations:
                    break
