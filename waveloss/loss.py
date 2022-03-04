from waveloss import discriminator, wavelet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReconstructionLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ReconstructionLoss, self).__init__()
        if (type == 'l1'):
            self.loss = nn.L1Loss()
        elif (type == 'l2'):
            self.loss = nn.MSELoss()
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')

    def forward(self, sr, hr):
        return self.loss(sr, hr)


class Rec_LL_Loss(nn.Module):
    def __init__(self, type='l1', use_cpu=False):
        super(Rec_LL_Loss, self).__init__()
        if (type == 'l1'):
            self.loss = nn.L1Loss()
        elif (type == 'l2'):
            self.loss = nn.MSELoss()
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')
        self.device = torch.device('cpu' if use_cpu else 'cuda')
        self.filter = wavelet.WavePool(3).to(self.device)

    def forward(self, sr, hr):
        LL_sr, _ = self.filter(sr)
        LL_hr, _ = self.filter(hr)
        return self.loss(LL_sr, LL_hr)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

    def forward(self, sr_relu5_1, hr_relu5_1):
        loss = F.mse_loss(sr_relu5_1, hr_relu5_1)
        return loss


def gram_matrix(x):
    b, ch, h, w = x.size()
    f = x.view(b, ch, h*w)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T)
    return G


class TPerceptualLoss(nn.Module):
    def __init__(self, use_S=True, type='l2'):
        super(TPerceptualLoss, self).__init__()
        self.use_S = use_S
        self.type = type

    def forward(self, map_lv3, map_lv2, map_lv1,
                sr_skips, S, T_lv3, T_lv2, T_lv1, skips_T):
        # S.size(): [N, 1, h, w]
        if (self.use_S):
            S_lv3 = torch.sigmoid(S)
            S_lv2 = torch.sigmoid(
                F.interpolate(S, size=(S.size(-2)*2,
                              S.size(-1)*2), mode='bicubic')
            )
            S_lv1 = torch.sigmoid(
                F.interpolate(S, size=(S.size(-2)*4,
                              S.size(-1)*4), mode='bicubic')
            )
        else:
            S_lv3, S_lv2, S_lv1 = 1., 1., 1.
        S_lv3 = S
        S_lv2 = F.interpolate(S, size=(S.size(-2)*2,
                              S.size(-1)*2), mode='bicubic')
        S_lv1 = F.interpolate(S, size=(S.size(-2)*4,
                              S.size(-1)*4), mode='bicubic')
        loss_texture = 0.
        loss_texture1 = 0.
        if (self.type == 'l1'):
            loss_texture += F.l1_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            loss_texture += F.l1_loss(map_lv2 * S_lv2, T_lv2 * S_lv2)
            for i in [0, 1, 2]:
                loss_texture1 += F.l1_loss(sr_skips['pool2'][i] * S_lv3,
                                           skips_T['T_lv3'][i] * S_lv3)
                loss_texture1 += F.l1_loss(sr_skips['pool1'][i] * S_lv2,
                                           skips_T['T_lv2'][i] * S_lv2)
            loss_texture += F.l1_loss(map_lv1 * S_lv1, T_lv1 * S_lv1)
            loss_texture /= 3.
            loss_texture1 /= 6.
        elif (self.type == 'l2'):
            loss_texture += F.mse_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            loss_texture += F.mse_loss(map_lv2 * S_lv2, T_lv2 * S_lv2)
            for i in [0, 1, 2]:
                loss_texture1 += F.mse_loss(sr_skips['pool2'][i] * S_lv3,
                                            skips_T['T_lv3'][i] * S_lv3)
                loss_texture1 += F.mse_loss(sr_skips['pool1'][i] * S_lv2,
                                            skips_T['T_lv2'][i] * S_lv2)
            loss_texture += F.mse_loss(map_lv1 * S_lv1, T_lv1 * S_lv1)
            loss_texture /= 3.
            loss_texture1 /= 6.

        return loss_texture, loss_texture1


class AdversarialLoss(nn.Module):
    def __init__(self, logger, use_cpu=False,
                 num_gpu=1, gan_type='WGAN_GP', gan_k=1,
                 lr_dis=1e-4, train_crop_size=40, in_channel=18):

        super(AdversarialLoss, self).__init__()
        self.logger = logger
        self.gan_type = gan_type
        self.gan_k = gan_k
        self.device = torch.device('cpu' if use_cpu else 'cuda')
        self.discriminator = discriminator.Discriminator(
            train_crop_size*4,
            in_channel=18
        ).to(self.device)
        if (num_gpu > 1):
            self.discriminator = nn.DataParallel(self.discriminator,
                                                 list(range(num_gpu)))
        if (gan_type in ['WGAN_GP', 'GAN']):
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=lr_dis
            )
        else:
            raise SystemExit('Error: no such type of GAN!')

        self.bce_loss = torch.nn.BCELoss().to(self.device)
        self.filter = wavelet.WavePool(3).to(self.device)

    def forward(self, fake, real, ref):
        fake_detach = fake.detach()
        _, fake_detach = self.filter(fake_detach)
        # wavelet real
        _, real = self.filter(real)
        # wavelet ref
        _, ref = self.filter(ref)
        fake_detach = torch.cat((fake_detach, ref), 1)
        real = torch.cat((real, ref), 1)

        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            if (self.gan_type.find('WGAN') >= 0):
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand(real.size(0), 1, 1, 1).to(self.device)
                    epsilon = epsilon.expand(real.size())
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            elif (self.gan_type == 'GAN'):
                valid_score = torch.ones(real.size(0), 1).to(self.device)
                fake_score = torch.zeros(real.size(0), 1).to(self.device)
                real_loss = self.bce_loss(torch.sigmoid(d_real), valid_score)
                fake_loss = self.bce_loss(torch.sigmoid(d_fake), fake_score)
                loss_d = (real_loss + fake_loss) / 2.

            # Discriminator update
            loss_d.backward()
            self.optimizer.step()

        _, fake = self.filter(fake)
        fake = torch.cat((fake, ref), 1)

        d_fake_for_g = self.discriminator(fake)
        if (self.gan_type.find('WGAN') >= 0):
            loss_g = -d_fake_for_g.mean()
        elif (self.gan_type == 'GAN'):
            loss_g = self.bce_loss(torch.sigmoid(d_fake_for_g), valid_score)

        # Generator loss
        return loss_g

    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict


def get_loss_dict(args, logger):
    loss = {}
    if (abs(args.rec_w - 0) <= 1e-8):
        raise SystemExit('NotImplementError: ReconstructionLoss must exist!')
    else:
        loss['rec_loss'] = ReconstructionLoss(type='l1')
    if (abs(args.rec_w - 0) > 1e-8):
        loss['rec_ll_loss'] = Rec_LL_Loss(type='l1', use_cpu=args.cpu)
    if (abs(args.per_w - 0) > 1e-8):
        loss['per_loss'] = PerceptualLoss()
    if (abs(args.tpl_w - 0) > 1e-8):
        loss['tpl_loss'] = TPerceptualLoss(use_S=args.tpl_use_S,
                                           type=args.tpl_type)
    if (abs(args.adv_w - 0) > 1e-8):
        loss['adv_loss'] = AdversarialLoss(
            logger=logger, use_cpu=args.cpu, num_gpu=args.num_gpu,
            gan_type=args.GAN_type, gan_k=args.GAN_k, lr_dis=args.lr_rate_dis,
            train_crop_size=args.train_crop_size//2, in_channel=9
        )
    return loss
