from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer_interpo_enc_dec_percp_gp import MUNIT_Trainer
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torch
try:
    from itertools import izip as zip
except ImportError:
    pass
import os
import sys
import tensorboardX
import shutil
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/cat2dog.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", type=bool, default=True)
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

trainer = LSGANs_Trainer(config)
trainer.cuda()
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()


# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory, checkpoint_iter = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))


iterations, previous_least_loss = trainer.resume_iter(checkpoint_iter, '_00300000', hyperparameters=config)


train_loader_a = enumerate(train_loader_a)
train_loader_b = enumerate(train_loader_b)

path_ab = "./translated_images/edge2bag/"
path_ba = "./translated_images/bag2edge/"

files = glob.glob(path_ab+'*')
for f in files:
    os.remove(f)

files = glob.glob(path_ba+'*')
for f in files:
    os.remove(f)

with torch.no_grad():
    trainer.eval()

    for it, (images_a, images_b) in enumerate(zip(test_loader_a, test_loader_b)):
        c_a, s_a = trainer.encoder(images_a.cuda().detach())
        c_b, s_b = trainer.encoder(images_b.cuda().detach())

        trainer.v = torch.ones(s_a.size())
        s_a_interp = trainer.interp_net_ba(s_b, s_a, trainer.v)
        s_b_interp = trainer.interp_net_ab(s_a, s_b, trainer.v)

        x_ab = trainer.decoder(c_a, s_b_interp)
        x_ba = trainer.decoder(c_b, s_a_interp)

        vutils.save_image(x_ab.data, path_ab+'ab_'+str(it)+'_.jpg', padding=0, normalize=True)
        vutils.save_image(x_ba.data, path_ba+'ba_'+str(it)+'_.jpg', padding=0, normalize=True)




