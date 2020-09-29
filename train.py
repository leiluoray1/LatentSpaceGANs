from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import LSGANs_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: 
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
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
test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory, checkpoint_iter = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

# Start training
iterations, previous_least_loss = 0, float("inf")
if opts.resume:
    iterations, previous_least_loss = trainer.resume_iter(checkpoint_iter, '_0300000', hyperparameters=config)


train_loader_a = enumerate(train_loader_a)
train_loader_b = enumerate(train_loader_b)

while True:
    try:
        with Timer("Elapsed time in update: %f"):
            for i in range(3):
                images_a = next(train_loader_a)[1].cuda().detach()
                images_b = next(train_loader_b)[1].cuda().detach()

                trainer.dis_update(images_a, images_b, config)

            trainer.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()


            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(image_outputs, display_size, image_directory, 'train_current')

            # Save network weights
            if trainer.total_loss < previous_least_loss:
                print("Better model:", trainer.total_loss)
                previous_least_loss = trainer.total_loss
                trainer.best_iter = iterations
                trainer.save_better_model(checkpoint_directory)

            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save_at_iter(checkpoint_iter, iterations)

            iterations += 1
            trainer.total_loss = 0
            if iterations >= max_iter:
                sys.exit('Finish training')

    except StopIteration:
        train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
        train_loader_a = enumerate(train_loader_a)
        train_loader_b = enumerate(train_loader_b)


