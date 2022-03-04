from utils import calc_psnr_and_ssim
from importlib import import_module
import os
import numpy as np
from imageio import imread, imsave
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class Trainer():

    def __init__(self, args, logger, dataloader, model, loss_all):
        self.args = args
        self.logger = logger
        self.tb_logger = SummaryWriter(log_dir=args.save_dir)
        self.dataloader = dataloader
        self.model = model
        self.loss_all = loss_all
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        Vgg19 = import_module('models.' + args.which_model + '.Vgg19')
        self.logger.info('Import model: [{}]'.format(args.which_model))
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        if ((not self.args.cpu) and (self.args.num_gpu > 1)):
            self.vgg19 = nn.DataParallel(
                self.vgg19,
                list(range(self.args.num_gpu))
            )
        self.params = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    self.model.MainNet.parameters()
                    if args.num_gpu == 1
                    else self.model.module.MainNet.parameters()
                ),
                "lr": args.lr_rate
            },
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    self.model.LTE.parameters()
                    if args.num_gpu == 1
                    else self.model.module.LTE.parameters()
                ),
                "lr": args.lr_rate_lte
            }
        ]
        self.optimizer = optim.Adam(self.params, betas=(
            args.beta1, args.beta2), eps=args.eps)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0
        self.min_lpips = float('inf')
        self.min_lpips_epoch = 0
        self.iter_idx = 0

    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            model_state_dict_save = {
                k: v for k, v in torch.load(model_path).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict)

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def train(self, current_epoch=0, is_init=False):
        self.model.train()
        self.logger.info('Current epoch learning rate: %e' % (
            self.optimizer.param_groups[0]['lr']))

        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()

            sample_batched = self.prepare(sample_batched)
            lr = sample_batched['LR']
            lr_sr = sample_batched['LR_sr']
            hr = sample_batched['HR']
            ref = sample_batched['Ref']
            ref_sr = sample_batched['Ref_sr']
            sr, S, T_lv3, T_lv2, T_lv1, skips_T = self.model(
                lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
            # calc loss
            is_print = ((i_batch + 1) % self.args.print_every == 0)

            if (is_init):
                rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)
                loss = rec_loss
            else:
                # only calculate the reconstruction loss of LL
                rec_loss = self.args.rec_w * \
                    self.loss_all['rec_ll_loss'](sr, hr)
                loss = rec_loss
            if (is_print):
                self.logger.info(
                    ('init ' if is_init else '') +
                    'epoch: ' + str(current_epoch) + '\t batch: ' +
                    str(i_batch+1)
                )
                self.logger.info('rec_loss: %.10f' % (rec_loss.item()))
                self.tb_logger.add_scalar(
                    'losses/rec_loss',
                    rec_loss.item(), self.iter_idx
                )
            if (not is_init):
                if ('per_loss' in self.loss_all):
                    sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
                    with torch.no_grad():
                        hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
                    per_loss = self.args.per_w * \
                        self.loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
                    loss += per_loss
                    if (is_print):
                        self.logger.info('per_loss: %.10f' % (per_loss.item()))
                        self.tb_logger.add_scalar(
                            'losses/per_loss',
                            per_loss.item(), self.iter_idx
                        )
                if ('tpl_loss' in self.loss_all):
                    sr_lv1, sr_lv2, sr_lv3, sr_skips = self.model(sr=sr)
                    res1, res2 = self.loss_all['tpl_loss'](
                        sr_lv3, sr_lv2, sr_lv1,
                        sr_skips, S, T_lv3, T_lv2, T_lv1, skips_T
                    )
                    tpl_loss = 1e-3 * res1 + self.args.tpl_w * res2
                    loss += tpl_loss
                    if (is_print):
                        self.logger.info('tpl_loss: %.10f' % (tpl_loss.item()))
                        self.tb_logger.add_scalar(
                            'losses/tpl_loss',
                            tpl_loss.item(), self.iter_idx
                        )
                if ('adv_loss' in self.loss_all):
                    adv_loss = self.args.adv_w * \
                        self.loss_all['adv_loss'](sr, hr, ref)
                    loss += adv_loss
                    if (is_print):
                        self.logger.info('adv_loss: %.10f' % (adv_loss.item()))
                        self.tb_logger.add_scalar(
                            'losses/adv_loss',
                            adv_loss.item(), self.iter_idx
                        )
            loss.backward()
            self.optimizer.step()
            self.iter_idx += 1

        self.scheduler.step()

        self.logger.info('saving the latest model...')
        tmp = self.model.state_dict()
        model_state_dict = {
            key.replace('module.', ''): tmp[key] for key in tmp if
            (('SearchNet' not in key) and ('_copy' not in key))
        }
        model_name = self.args.save_dir.strip('/')+'/model/model_latest.pth'
        torch.save(model_state_dict, model_name)

        if ((not is_init) and current_epoch % self.args.save_every == 0):
            self.logger.info('saving the model...')
            model_name = self.args.save_dir.strip('/') + '/model/model_' + \
                str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)

    def evaluate(self, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) +
                         ' evaluation process...')
        import lpips
        loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)

        if (self.args.dataset == 'CUFED'):
            self.model.eval()
            with torch.no_grad():
                psnr, ssim, cnt = 0., 0., 0
                lpips = 0.
                test_data_bar = tqdm(self.dataloader['test']['1'])
                for i_batch, sample_batched in enumerate(test_data_bar):
                    torch.cuda.empty_cache()
                    cnt += 1
                    sample_batched = self.prepare(sample_batched)
                    lr = sample_batched['LR']
                    lr_sr = sample_batched['LR_sr']
                    hr = sample_batched['HR']
                    ref = sample_batched['Ref']
                    ref_sr = sample_batched['Ref_sr']
                    sr, _, _, _, _, _ = self.model(lr=lr, lrsr=lr_sr,
                                                   ref=ref, refsr=ref_sr)
                    if (self.args.eval_save_results):
                        sr_save = (sr+1.) * 127.5
                        sr_save = np.transpose(
                            sr_save.squeeze().round().cpu().numpy(),
                            (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results',
                               str(i_batch).zfill(5)+'.png'), sr_save)

                    # calculate psnr and ssim
                    _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())

                    psnr += _psnr
                    ssim += _ssim
                    lpips += loss_fn_alex(hr.detach(), sr.detach())
                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                lpips_ave = lpips / cnt
                self.logger.info(
                    'Ref  PSNR (now): %.3f \t SSIM (now): %.4f \t LPIPS (now): %.4f'
                    % (psnr_ave, ssim_ave, lpips_ave)
                )
                self.tb_logger.add_scalar('metrics/psnr', psnr_ave,
                                          current_epoch)
                self.tb_logger.add_scalar('metrics/ssim', ssim_ave,
                                          current_epoch)
                self.tb_logger.add_scalar('metrics/lpips', lpips_ave,
                                          current_epoch)
                if (psnr_ave > self.max_psnr):
                    self.max_psnr = psnr_ave
                    self.max_psnr_epoch = current_epoch

                    if self.args.save_model:
                        self.logger.info('saving the best model...')
                        tmp = self.model.state_dict()
                        model_state_dict = {
                            key.replace('module.', ''):
                            tmp[key] for key in tmp
                            if (
                                ('SearchNet' not in key) and
                                ('_copy' not in key)
                            )
                        }
                        model_name = self.args.save_dir.strip('/') + \
                            '/model/model_best.pth'
                        torch.save(model_state_dict, model_name)

                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch

                if (lpips_ave < self.min_lpips):
                    self.min_lpips = lpips_ave
                    self.min_lpips_epoch = current_epoch

                self.logger.info(
                    'Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d) \t LPIPS (min): %.4f (%d)' 
                    %(self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch, self.min_lpips,self.min_lpips_epoch)
                )

        self.logger.info('Evaluation over.')

    def test(self):
        self.logger.info('Test process...')
        self.logger.info('lr path:     %s' % (self.args.lr_path))
        self.logger.info('ref path:    %s' % (self.args.ref_path))

        # LR and LR_sr
        LR = imread(self.args.lr_path)
        h1, w1 = LR.shape[:2]
        LR_sr = np.array(
            Image.fromarray(LR).resize((w1*4, h1*4), Image.BICUBIC))

        # Ref and Ref_sr
        Ref = imread(self.args.ref_path)
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2//4*4, w2//4*4
        Ref = Ref[:h2, :w2, :]
        Ref_sr = np.array(
            Image.fromarray(Ref).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr = np.array(
            Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))

        # change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        # rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        # to tensor
        LR_t = torch.from_numpy(LR.transpose((2, 0, 1)))\
            .unsqueeze(0).float().to(self.device)
        LR_sr_t = torch.from_numpy(LR_sr.transpose((2, 0, 1)))\
            .unsqueeze(0).float().to(self.device)
        Ref_t = torch.from_numpy(Ref.transpose((2, 0, 1)))\
            .unsqueeze(0).float().to(self.device)
        Ref_sr_t = torch.from_numpy(Ref_sr.transpose((2, 0, 1)))\
            .unsqueeze(0).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            sr, _, _, _, _, _ = self.model(lr=LR_t, lrsr=LR_sr_t,
                                           ref=Ref_t, refsr=Ref_sr_t)
            sr_save = (sr+1.) * 127.5
            sr_save = np.transpose(
                sr_save.squeeze().round().cpu().numpy(),
                (1, 2, 0)
            ).astype(np.uint8)
            save_path = os.path.join(self.args.save_dir, 'save_results',
                                     os.path.basename(self.args.lr_path))
            imsave(save_path, sr_save)
            self.logger.info('output path: %s' % (save_path))
        self.logger.info('Test over.')
