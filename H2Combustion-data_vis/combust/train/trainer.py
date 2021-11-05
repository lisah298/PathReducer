import os
import numpy as np
import pandas as pd
import torch
import time
import shutil

from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR


class Trainer:
    """
    Parameters
    ----------
    """
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 device,
                 output_path,
                 script_name,
                 lr_scheduler,
                 checkpoint_interval=1,
                 validation_interval=1):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        shutil.copyfile(script_name, os.path.join(output_path,script_name))

        if lr_scheduler[0] == 'plateau':
            self.scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                               mode='min',
                                               patience=lr_scheduler[2],
                                               factor=lr_scheduler[3],
                                               min_lr=lr_scheduler[4])
        elif lr_scheduler[0] == 'decay':
            lambda1 = lambda epoch: np.exp(-epoch * lr_scheduler[1])
            self.scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda1)

        self.lr_scheduler = lr_scheduler

        self.checkpoint_interval = checkpoint_interval
        self.validation_interval = validation_interval

        # checkpoints
        self.epoch = 0  # number of epochs of any steps that model has gone through so far
        self.data_point = 0  # number of data points that model has seen so far
        self.log_loss = {
            'epoch': [],
            'loss': [],
            'val_ae_energy': [],
            'val_ae_forces': [],
            'test_loss': [],
            'lr': [],
            'time': []
        }
        self.best_val_loss = float("inf")

    def print_layers(self):
        total_n_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
                if len(param.shape) > 1:
                    total_n_params += param.shape[0] * param.shape[1]
                else:
                    total_n_params += param.shape[0]
        print('\n total trainable parameters: %i\n' % total_n_params)

    def store_checkpoint(self, input, steps):
        self.log_loss['epoch'].append(self.epoch)
        self.log_loss['loss'].append(np.sqrt(input[0]))
        self.log_loss['val_ae_energy'].append(input[1])
        self.log_loss['val_ae_forces'].append(input[2])
        self.log_loss['test_loss'].append(0.0)
        self.log_loss['lr'].append(input[3])
        self.log_loss['time'].append(input[4])

        pd.DataFrame(self.log_loss).to_csv(os.path.join(
            self.output_path, 'log.csv'),
                                           index=False)

        print(
            "[%d, %3d] loss_rmse: %.5f; val_ae_energy: %.5f; val_ae_forces: %.5f; test_loss: %.5f; lr: %.7f; epoch_time: %.3f"
            % (self.epoch, steps, np.sqrt(
                input[0]), input[1], input[2], 0.0, input[3], input[4]))

    def _optimizer_to_device(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def metric_se(self, preds, data):
        """squared error"""
        diff = preds - data
        se = diff**2

        # diff_forces = preds[1] - batch_data.forces
        # diff_forces = diff_forces ** 2

        # err_sq = torch.sum(diff_energy) + torch.sum(diff_forces)

        return torch.sum(se)

    def metric_ae(self, preds, data):
        """absolute error"""
        ae = torch.abs(preds - data)

        # diff_forces = preds[1] - batch_data.forces
        # diff_forces = diff_forces ** 2

        # err_sq = torch.sum(diff_energy) + torch.sum(diff_forces)

        return torch.sum(ae)

    def train(self,
              train_generator,
              epochs,
              steps,
              val_generator=None,
              val_steps=None):
        """
        The main function to train model for the given number of epochs (and steps per epochs).
        The implementation allows for resuming the training with different data and number of epochs.

        Parameters
        ----------
        epochs: int
            number of training epochs

        steps: int
            number of steps to call it an epoch (designed for nonstop data generators)


        """
        self.model.to(self.device)
        self._optimizer_to_device()

        running_val_loss = []
        for _ in range(epochs):
            t0 = time.time()

            # record total number of epochs so far
            self.epoch += 1

            # training
            running_loss = 0.0
            for s in range(steps):
                self.optimizer.zero_grad()

                train_batch = next(train_generator)

                preds = self.model(train_batch)
                loss = self.loss_fn(preds, train_batch)
                loss.backward()
                self.optimizer.step()
                current_loss = loss.item()
                running_loss += current_loss

                # update number of data points
                nb = train_batch.energy.size()[0]
                self.data_point += nb

                print(
                    "Epoch %i/%i - %i/%i - loss: %.5f - sqr(running_loss): %.5f"
                    % (self.epoch, epochs, s, steps, current_loss,
                       np.sqrt(running_loss / (s + 1))))

                del train_batch

            running_loss /= steps

            # validation
            val_loss_energy = 0.0
            val_loss_force = 0.0
            if val_generator is not None and \
                self.epoch % self.validation_interval == 0:

                n_val = 0
                for _ in range(val_steps):

                    val_batch = next(val_generator)

                    val_preds = self.model(val_batch)
                    val_loss_energy += (self.metric_ae(
                        val_preds[0], val_batch.energy).data.cpu().numpy())
                    val_loss_force += (self.metric_ae(
                        val_preds[1], val_batch.forces).data.cpu().numpy())
                    n_val += val_batch.energy.size()[0]
                    n_atoms = val_batch.atoms.size()[1]

                    del val_batch

                val_loss_energy /= n_val
                val_loss_force /= (n_val * n_atoms * 3)
                val_loss = val_loss_energy + val_loss_force
            else:
                val_loss = float("inf")

            # best model
            if self.best_val_loss > val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(),
                           os.path.join(self.output_path, 'best_model'))
            elif val_generator is None:
                if self.epoch % self.validation_interval == 0:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            self.output_path, 'model_every-%i-epochs' %
                            self.validation_interval))

            # learning rate decay
            if self.lr_scheduler[0] == 'plateau':
                running_val_loss.append(val_loss)
                if len(running_val_loss) > self.lr_scheduler[1]:
                    running_val_loss.pop(0)
                accum_val_loss = np.mean(running_val_loss)
                self.scheduler.step(accum_val_loss)
            elif self.lr_scheduler[0] == 'decay':
                self.scheduler.step()
                accum_val_loss = 0.0

            # checkpoint
            if self.epoch % self.checkpoint_interval == 0:

                for i, param_group in enumerate(
                        self.scheduler.optimizer.param_groups):
                    old_lr = float(param_group["lr"])

                self.store_checkpoint(
                    (running_loss, val_loss_energy, val_loss_force, old_lr,
                     time.time() - t0), steps)
