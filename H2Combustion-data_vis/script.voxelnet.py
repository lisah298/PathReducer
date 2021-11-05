import os
import numpy as np
import torch
from torch.optim import Adam

from combust.data import ExtensiveEnvironment
from combust.data import extensive_train_loader, extensive_irc_loader
from combust.layers import ShellProvider, ManyBodyVoxel
from combust.models import MRDenseNet, Voxel3D, Testnet
from combust.train import Trainer

device = torch.device('cuda:0')
model_mode = 'channelize'  # channelize, embed_elemental, split_elemental, shared_elemental
output_dir = 'local/voxelnet/%s_g16_ch2_rot8_energy.001'%(model_mode)
script_name = 'script.voxelnet.py'

n_data = 10000

derivative = True
rho_tradeoff = 0.001

batch_size = 16
epochs = 25
n_rotations = 7
steps = int(n_data / batch_size) * (n_rotations + 1)

normalizer = (-210, 12)
atom_types = [1, 8]
# grid_length = [1., 2., 4., 6., 14.]
grid_length = [14.]
grid_size = 16
sigma = 1 / 3

dropout = 0.1
lr = 1e-3
weight_decay = 3e-5

checkpoint_interval=1
validation_interval=1

# lr_scheduler = ('plateau', 5, 3, 0.7, 1e-6)   # '', val_loss_length, patience, decay, min
lr_scheduler = ('decay', 0.3)  # '', decay

# data
dir_path = "/home/moji/Dropbox/AIMD/04/combined/"
env = ExtensiveEnvironment()
train_gen = extensive_train_loader(dir_path=dir_path,
                                   env_provider=env,
                                   data_size=10000,
                                   batch_size=batch_size,
                                   n_rotations=n_rotations,
                                   freeze_rotations=True,
                                   device=device,
                                   shuffle=True,
                                   drop_last=False)

val_gen = extensive_train_loader(dir_path=dir_path,
                                   env_provider=env,
                                   data_size=-10000,
                                   batch_size=4,
                                   n_rotations=n_rotations,
                                   freeze_rotations=True,
                                   device=device,
                                   shuffle=True,
                                   drop_last=False)


dir_path = "/home/moji/Dropbox/IRC/04_H2O+O_2OH.t/"
test_gen = extensive_irc_loader(dir_path=dir_path,
                               env_provider=env,
                               batch_size=4,
                               n_rotations=0,
                               device=device,
                               shuffle=False,
                               drop_last=False)

# model
shell = ShellProvider()
mb_voxel = ManyBodyVoxel(mode = model_mode,
                         atom_types=atom_types,
                         grid_length=grid_length,
                         grid_size=grid_size,
                         sigma=torch.tensor(sigma,
                                            device=device,
                                            dtype=torch.float))
if model_mode=='channelize':
    in_channels = len(atom_types) * len(grid_length)
    mr_densenet = MRDenseNet(in_channels=in_channels, dropout=dropout)
elif model_mode=='embed_elemental':
    in_channels = len(grid_length)
    mr_densenet = MRDenseNet(in_channels=in_channels, dropout=dropout)
elif model_mode=='split_elemental':
    in_channels = len(grid_length)
    mr_densenet = torch.nn.ModuleList([
        MRDenseNet(in_channels=in_channels, dropout=dropout)
        for _ in range(len(atom_types))
    ])
elif model_mode=='shared_elemental':
    in_channels = len(grid_length)
    mr_densenet = torch.nn.ModuleList([
        MRDenseNet(in_channels=in_channels, dropout=dropout)
    ]
        * len(atom_types)
    )

model = Voxel3D(shell,
                mb_voxel,
                mr_densenet,
                mode=model_mode,
                normalizer=normalizer,
                device=device,
                derivative=derivative,
                create_graph=False)

# optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)


# loss
def custom_loss(preds, batch_data):
    # compute the mean squared error on the energies
    diff_energy = preds[0] - batch_data.energy
    assert diff_energy.shape[1] == 1
    err_sq_energy = torch.mean(diff_energy**2)

    # compute the mean squared error on the forces
    diff_forces = preds[1] - batch_data.forces
    err_sq_forces = torch.mean(diff_forces**2)

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    direction_diff = 1 - cos(preds[1], batch_data.forces)
    direction_diff *= torch.norm(batch_data.forces, p=2, dim=-1)
    direction_loss = torch.mean(direction_diff)

    print('energy loss: ', err_sq_energy, 'force loss: ', err_sq_forces,
          'direction loss: ', direction_loss)

    # build the combined loss function
    err_sq = rho_tradeoff * err_sq_energy + \
             (1 - rho_tradeoff) * err_sq_forces + \
             direction_loss * 10

    return err_sq


# training
trainer = Trainer(model=model,
                  loss_fn=custom_loss,
                  optimizer=optimizer,
                  device=device,
                  output_path=output_dir,
                  script_name=script_name,
                  lr_scheduler=lr_scheduler,
                  checkpoint_interval=checkpoint_interval,
                  validation_interval=validation_interval)

trainer.print_layers()

trainer.train(train_generator=train_gen,
              epochs=epochs,
              steps=steps,
              val_generator=val_gen,
              val_steps=50)

print('done!')
