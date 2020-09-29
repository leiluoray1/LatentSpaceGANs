from networks import Encoder, Decoder, Interpolator, Perceptural_loss, Discriminator
from utils import weights_init, get_model, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import glob

class LSGANs_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(LSGANs_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.encoder = Encoder(hyperparameters['input_dim_a'], hyperparameters['gen'])
        self.decoder = Decoder(hyperparameters['input_dim_a'], hyperparameters['gen'])
        self.dis_a = Discriminator()  
        self.dis_b = Discriminator()  
        self.interp_net_ab = Interpolator()
        self.interp_net_ba = Interpolator()
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        enc_params = list(self.encoder.parameters())
        dec_params = list(self.decoder.parameters())
        dis_a_params = list(self.dis_a.parameters()) 
        dis_b_params = list(self.dis_b.parameters())
        interperlator_ab_params = list(self.interp_net_ab.parameters())
        interperlator_ba_params = list(self.interp_net_ba.parameters())

        self.enc_opt = torch.optim.Adam([p for p in enc_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dec_opt = torch.optim.Adam([p for p in dec_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])  
        self.dis_a_opt = torch.optim.Adam([p for p in dis_a_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_b_opt = torch.optim.Adam([p for p in dis_b_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.interp_ab_opt = torch.optim.Adam([p for p in interperlator_ab_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.interp_ba_opt = torch.optim.Adam([p for p in interperlator_ba_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.enc_scheduler = get_scheduler(self.enc_opt, hyperparameters)
        self.dec_scheduler = get_scheduler(self.dec_opt, hyperparameters)
        self.dis_a_scheduler = get_scheduler(self.dis_a_opt, hyperparameters)
        self.dis_b_scheduler = get_scheduler(self.dis_b_opt, hyperparameters)
        self.interp_ab_scheduler = get_scheduler(self.interp_ab_opt, hyperparameters)
        self.interp_ba_scheduler = get_scheduler(self.interp_ba_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))


        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        self.total_loss = 0
        self.best_iter = 0
        self.perceptural_loss = Perceptural_loss()

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()

        c_a, s_a_fake = self.encoder(x_a)
        c_b, s_b_fake = self.encoder(x_b)
        
        # decode (cross domain)
        s_ab_interp = self.interp_net_ab(s_a_fake, s_b_fake, self.v)
        s_ba_interp = self.interp_net_ba(s_b_fake, s_a_fake, self.v)
        x_ba = self.decoder(c_b, s_a_interp)
        x_ab = selfdecoder(c_a, s_b_interp)
        self.train()
        return x_ab, x_ba
    
    def zero_grad(self):
        self.dis_a_opt.zero_grad()
        self.dis_b_opt.zero_grad()
        self.dec_opt.zero_grad()
        self.enc_opt.zero_grad()
        self.interp_ab_opt.zero_grad()
        self.interp_ba_opt.zero_grad()

    def dis_update(self, x_a, x_b, hyperparameters):
        self.zero_grad()

        # encode
        c_a, s_a = self.encoder(x_a)
        c_b, s_b = self.encoder(x_b)


        # decode (cross domain)
        self.v = torch.ones(s_a.size())
        s_a_interp = self.interp_net_ba(s_b, s_a, self.v)
        s_b_interp = self.interp_net_ab(s_a, s_b, self.v)
        x_ba = self.decoder(c_b, s_a_interp)
        x_ab = self.decoder(c_a, s_b_interp)


        x_a_feature = self.dis_a(x_a)
        x_ba_feature = self.dis_a(x_ba)
        x_b_feature = self.dis_b(x_b)
        x_ab_feature = self.dis_b(x_ab)
        self.loss_dis_a = (x_ba_feature - x_a_feature).mean()
        self.loss_dis_b = (x_ab_feature - x_b_feature).mean()

        # gradient penality
        self.loss_dis_a_gp = self.dis_a.calculate_gradient_penalty(x_ba, x_a)
        self.loss_dis_b_gp = self.dis_b.calculate_gradient_penalty(x_ab, x_b)


        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + \
                              hyperparameters['gan_w'] * self.loss_dis_b + \
                              hyperparameters['gan_w'] * self.loss_dis_a_gp + \
                              hyperparameters['gan_w'] * self.loss_dis_b_gp


        self.loss_dis_total.backward()
        self.total_loss += self.loss_dis_total.item()
        self.dis_a_opt.step()
        self.dis_b_opt.step()


    def gen_update(self, x_a, x_b, hyperparameters):
        self.zero_grad()

        # encode
        c_a, s_a = self.encoder(x_a)
        c_b, s_b = self.encoder(x_b)

        # decode (within domain)
        x_a_recon = self.decoder(c_a, s_a)
        x_b_recon = self.decoder(c_b, s_b)

        # decode (cross domain)
        self.v = torch.ones(s_a.size())
        s_a_interp = self.interp_net_ba(s_b, s_a, self.v)
        s_b_interp = self.interp_net_ab(s_a, s_b, self.v)
        x_ba = self.decoder(c_b, s_a_interp)
        x_ab = self.decoder(c_a, s_b_interp)

        # encode again
        c_b_recon, s_a_recon = self.encoder(x_ba)
        c_a_recon, s_b_recon = self.encoder(x_ab)

        # decode again 
        x_aa = self.decoder(c_a_recon, s_a) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bb = self.decoder(c_b_recon, s_b) if hyperparameters['recon_x_cyc_w'] > 0 else None


        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aa, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bb, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0

        # perceptual loss
        self.loss_gen_vgg_a = self.perceptural_loss(x_a_recon, x_a) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.perceptural_loss(x_b_recon, x_b) if hyperparameters['vgg_w'] > 0 else 0

        self.loss_gen_vgg_aa = self.perceptural_loss(x_aa, x_a) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_bb = self.perceptural_loss(x_bb, x_b) if hyperparameters['vgg_w'] > 0 else 0
        
        # GAN loss
        x_ba_feature = self.dis_a(x_ba)
        x_ab_feature = self.dis_b(x_ab)
        self.loss_gen_adv_a = -x_ba_feature.mean() 
        self.loss_gen_adv_b = -x_ab_feature.mean() 

        
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_aa + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_bb + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b

        
        self.loss_gen_total.backward()
        self.total_loss += self.loss_gen_total.item()
        self.dec_opt.step()
        self.enc_opt.step()
        self.interp_ab_opt.step()
        self.interp_ba_opt.step()


    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ab, x_ba, x_aa, x_bb = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a = self.encoder(x_a[i].unsqueeze(0))
            c_b, s_b = self.encoder(x_b[i].unsqueeze(0))
            x_a_recon.append(self.decoder(c_a, s_a))
            x_b_recon.append(self.decoder(c_b, s_b))
            
            self.v = torch.ones(s_a.size())
            s_a_interp = self.interp_net_ba(s_b, s_a, self.v)
            s_b_interp = self.interp_net_ab(s_a, s_b, self.v)

            x_ab_i = self.decoder(c_a, s_b_interp)
            x_ba_i = self.decoder(c_b, s_a_interp)

            c_a_recon, s_b_recon = self.encoder(x_ab_i)
            c_b_recon, s_a_recon = self.encoder(x_ba_i)


            x_ab.append(self.decoder(c_a, s_b_interp.unsqueeze(0)))
            x_ba.append(self.decoder(c_b, s_a_interp.unsqueeze(0)))
            x_aa.append(self.decoder(c_a_recon, s_a.unsqueeze(0)))
            x_bb.append(self.decoder(c_b_recon, s_b.unsqueeze(0)))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ab, x_aa = torch.cat(x_ab), torch.cat(x_aa)
        x_ba, x_bb= torch.cat(x_ba), torch.cat(x_bb)

        self.train()

        return x_a, x_a_recon, x_ab, x_aa, x_b, x_b_recon, x_ba, x_bb


    def update_learning_rate(self):
        if self.dis_a_scheduler is not None:
            self.dis_a_scheduler.step()
        if self.dis_b_scheduler is not None:
            self.dis_b_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.enc_scheduler is not None:
            self.enc_scheduler.step()
        if self.dec_scheduler is not None:
            self.dec_scheduler.step()
        if self.interpo_ab_scheduler is not None:
            self.interpo_ab_scheduler.step()
        if self.interpo_ba_scheduler is not None:
            self.interpo_ba_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load encode
        model_name = get_model(checkpoint_dir, "encoder")
        state_dict = torch.load(model_name)
        self.encoder.load_state_dict(state_dict)

        # Load decode
        model_name = get_model(checkpoint_dir, "decoder")
        state_dict = torch.load(model_name)
        self.decoder.load_state_dict(state_dict)

        # Load discriminator a
        model_name = get_model(checkpoint_dir, "dis_a")
        state_dict = torch.load(model_name)
        self.dis_a.load_state_dict(state_dict)

        # Load discriminator a
        model_name = get_model(checkpoint_dir, "dis_b")
        state_dict = torch.load(model_name)
        self.dis_b.load_state_dict(state_dict)

        # Load interperlator ab
        model_name = get_model(checkpoint_dir, "interp_ab")
        state_dict = torch.load(model_name)
        self.interp_net_ab.load_state_dict(state_dict)

        # Load interperlator ba
        model_name = get_model(checkpoint_dir, "interp_ba")
        state_dict = torch.load(model_name)
        self.interp_net_ba.load_state_dict(state_dict)

        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.enc_opt.load_state_dict(state_dict['enc_opt'])
        self.dec_opt.load_state_dict(state_dict['dec_opt'])
        self.dis_a_opt.load_state_dict(state_dict['dis_a_opt'])
        self.dis_b_opt.load_state_dict(state_dict['dis_b_opt'])
        self.interp_ab_opt.load_state_dict(state_dict['interp_ab_opt'])
        self.interp_ba_opt.load_state_dict(state_dict['interp_ba_opt'])

        self.best_iter = state_dict['best_iter']
        self.total_loss = state_dict['total_loss']

        # Reinitilize schedulers
        self.dis_a_scheduler = get_scheduler(self.dis_a_opt, hyperparameters, self.best_iter)
        self.dis_b_scheduler = get_scheduler(self.dis_b_opt, hyperparameters, self.best_iter)
        self.enc_scheduler = get_scheduler(self.enc_opt, hyperparameters, self.best_iter)
        self.dec_scheduler = get_scheduler(self.dec_opt, hyperparameters, self.best_iter)
        self.interpo_ab_scheduler = get_scheduler(self.interp_ab_opt, hyperparameters, self.best_iter)
        self.interpo_ba_scheduler = get_scheduler(self.interp_ba_opt, hyperparameters, self.best_iter)
        print('Resume from iteration %d' % self.best_iter)
        return self.best_iter, self.total_loss


    def resume_iter(self, checkpoint_dir, surfix, hyperparameters):
        # Load encode
        state_dict = torch.load(os.path.join(checkpoint_dir, 'encoder'+surfix+'.pt'))
        self.encoder.load_state_dict(state_dict)

        # Load decode
        state_dict = torch.load(os.path.join(checkpoint_dir, 'decoder'+surfix+'.pt'))
        self.decoder.load_state_dict(state_dict)

        # Load discriminator a 
        state_dict = torch.load(os.path.join(checkpoint_dir, 'dis_a'+surfix+'.pt'))
        self.dis_a.load_state_dict(state_dict)

        # # Load discriminator b
        state_dict = torch.load(os.path.join(checkpoint_dir, 'dis_b'+surfix+'.pt'))
        self.dis_b.load_state_dict(state_dict)

        state_dict = torch.load(os.path.join(checkpoint_dir, 'interp'+surfix+'.pt'))
        # print(state_dict)
        self.interp_net_ab.load_state_dict(state_dict['ab'])
        self.interp_net_ba.load_state_dict(state_dict['ba'])

        # Load interperlator ab
        state_dict = torch.load(os.path.join(checkpoint_dir, 'interp_ab'+surfix+'.pt'))
        self.interp_net_ab.load_state_dict(state_dict)
        
        # # Load interperlator ba
        state_dict = torch.load(os.path.join(checkpoint_dir, 'interp_ba'+surfix+'.pt'))
        self.interp_net_ba.load_state_dict(state_dict)

        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer'+surfix+'.pt'))
        self.enc_opt.load_state_dict(state_dict['enc_opt'])
        self.dec_opt.load_state_dict(state_dict['dec_opt'])
        self.dis_a_opt.load_state_dict(state_dict['dis_a_opt'])
        self.dis_b_opt.load_state_dict(state_dict['dis_b_opt'])
        self.interp_ab_opt.load_state_dict(state_dict['interp_ab_opt'])
        self.interp_ba_opt.load_state_dict(state_dict['interp_ba_opt'])

        self.best_iter = state_dict['best_iter']
        self.total_loss = state_dict['total_loss']

        # Reinitilize schedulers
        self.dis_a_scheduler = get_scheduler(self.dis_a_opt, hyperparameters, self.best_iter)
        self.dis_b_scheduler = get_scheduler(self.dis_b_opt, hyperparameters, self.best_iter)
        self.enc_scheduler = get_scheduler(self.enc_opt, hyperparameters, self.best_iter)
        self.dec_scheduler = get_scheduler(self.dec_opt, hyperparameters, self.best_iter)
        self.interpo_ab_scheduler = get_scheduler(self.interp_ab_opt, hyperparameters, self.best_iter)
        self.interpo_ba_scheduler = get_scheduler(self.interp_ba_opt, hyperparameters, self.best_iter)
        print('Resume from iteration %d' % self.best_iter)
        return self.best_iter, self.total_loss

    def save_better_model(self, snapshot_dir):
        # remove sub_optimal models
        files = glob.glob(snapshot_dir+'/*')
        for f in files:
            os.remove(f)
        # Save encoder, decoder, interpolator, discriminators, and optimizers
        encoder_name = os.path.join(snapshot_dir, 'encoder_%.4f.pt' % (self.total_loss))
        decoder_name = os.path.join(snapshot_dir, 'decoder_%.4f.pt' % (self.total_loss))
        interp_ab_name = os.path.join(snapshot_dir, 'interp_ab_%.4f.pt' % (self.total_loss))
        interp_ba_name = os.path.join(snapshot_dir, 'interp_ba_%.4f.pt' % (self.total_loss))
        dis_a_name = os.path.join(snapshot_dir, 'dis_a_%.4f.pt' % (self.total_loss))
        dis_b_name = os.path.join(snapshot_dir, 'dis_b_%.4f.pt' % (self.total_loss))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        
        torch.save(self.encoder.state_dict(), encoder_name)
        torch.save(self.decoder.state_dict(), decoder_name)
        torch.save(self.interp_net_ab.state_dict(), interp_ab_name)
        torch.save(self.interp_net_ba.state_dict(), interp_ba_name)
        torch.save(self.dis_a_opt.state_dict(), dis_a_name)
        torch.save(self.dis_b_opt.state_dict(), dis_b_name)
        torch.save({'enc_opt': self.enc_opt.state_dict(),
                    'dec_opt': self.dec_opt.state_dict(),
                    'dis_a_opt': self.dis_a_opt.state_dict(),
                    'dis_b_opt': self.dis_b_opt.state_dict(),
                    'interp_ab_opt': self.interp_ab_opt.state_dict(),
                    'interp_ba_opt': self.interp_ba_opt.state_dict(),
                    'best_iter': self.best_iter,
                    'total_loss': self.total_loss}, opt_name)


    def save_at_iter(self, snapshot_dir, iterations):

        encoder_name = os.path.join(snapshot_dir, 'encoder_%08d.pt' % (iterations + 1))
        decoder_name = os.path.join(snapshot_dir, 'decoder_%08d.pt' % (iterations + 1))
        interp_ab_name = os.path.join(snapshot_dir, 'interp_ab_%08d.pt' % (iterations + 1))
        interp_ba_name = os.path.join(snapshot_dir, 'interp_ba_%08d.pt' % (iterations + 1))
        dis_a_name = os.path.join(snapshot_dir, 'dis_a_%08d.pt' % (iterations + 1))
        dis_b_name = os.path.join(snapshot_dir, 'dis_b_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer_%08d.pt' % (iterations + 1))
        
        torch.save(self.encoder.state_dict(), encoder_name)
        torch.save(self.decoder.state_dict(), decoder_name)
        torch.save(self.interp_net_ab.state_dict(), interp_ab_name)
        torch.save(self.interp_net_ba.state_dict(), interp_ba_name)
        torch.save(self.dis_a_opt.state_dict(), dis_a_name)
        torch.save(self.dis_b_opt.state_dict(), dis_b_name)
        torch.save({'enc_opt': self.enc_opt.state_dict(),
                    'dec_opt': self.dec_opt.state_dict(),
                    'dis_a_opt': self.dis_a_opt.state_dict(),
                    'dis_b_opt': self.dis_b_opt.state_dict(),
                    'interp_ab_opt': self.interp_ab_opt.state_dict(),
                    'interp_ba_opt': self.interp_ba_opt.state_dict(),
                    'best_iter': self.best_iter,
                    'total_loss': self.total_loss}, opt_name)