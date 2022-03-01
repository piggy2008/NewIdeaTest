import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.nn import functional as F
from matplotlib import pyplot as plt

import contextual_loss as cl

import joint_transforms
from config import msra10k_path, video_train_path, datasets_root, video_seq_gt_path, video_seq_path, saving_path
from water_dataset import WaterImageFolder, WaterImage2Folder, WaterImage3Folder, WaterImage4Folder
from underwater_model.model_SPOS import Water
from underwater_model.discriminator import Discriminator, PatchDiscriminator

from misc import AvgMeter, check_mkdir, VGGPerceptualLoss, Lab_Loss, GANLoss, VGG19_PercepLoss
from torch.backends import cudnn
import time
from utils.utils_mine import load_part_of_model, load_part_of_model2, load_MGA
# from module.morphology import Erosion2d
import random
import numpy as np

cudnn.benchmark = True

device_id = 0
device_id2 = 0

torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(2021)
np.random.seed(2021)
# torch.cuda.set_device(device_id)


time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
ckpt_path = saving_path
exp_name = 'WaterEnhance' + '_' + time_str

args = {
    'gnn': True,
    'choice': 8,
    # 'choice2': 4,
    'layers': 12,
    # 'layers2': 3,
    'en_channels': [64, 128, 256],
    'dim': 48,
    'distillation': False,
    'L2': False,
    'KL': True,
    'structure': True,
    'iter_num': 300000,
    'iter_save': 4000,
    'iter_start_seq': 0,
    'train_batch_size': 6,
    'last_iter': 0,
    'lr': 1e-4,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.925,
    'snapshot': '',
    # 'pretrain': os.path.join(ckpt_path, 'WaterEnhance_2022-02-14 18:11:18', '40000.pth'),
    'pretrain': '',
    # 'mga_model_path': 'pre-trained/MGA_trained.pth',
    # 'imgs_file': '/mnt/hdd/data/ty2',
    'imgs_file': '/home/ty/data/uw',
    # 'imgs_file': 'Pre-train/pretrain_all_seq_DAFB2_DAVSOD_flow.txt',
    # 'imgs_file2': 'Pre-train/pretrain_all_seq_DUT_TR_DAFB2.txt',
    # 'imgs_file': 'video_saliency/train_all_DAFB2_DAVSOD_5f.txt',
    # 'train_loader': 'video_image'
    # 'train_loader': 'flow_image3',
    # 'train_loader': 'video_sequence'
    'image_size': 320,
    'crop_size': 256,
    # 'self_distill': 0.1,
    # 'teacher_distill': 0.6
}
# if train_set2 is not None:
#     train_loader2 = DataLoader(train_set2, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)

criterion = nn.MSELoss()
criterion_l1 = nn.L1Loss()
criterion_perceptual = VGGPerceptualLoss().cuda()
# criterion_gan = GANLoss(gan_mode='vanilla').cuda()
# criterion_lab = Lab_Loss().cuda()
# criterion_context = cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4').cuda()
# criterion_tv = TVLoss(TVLoss_weight=10).cuda()
# erosion = Erosion2d(1, 1, 5, soft_max=False).cuda()

log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')

total_loss_record, loss0_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

def fix_parameters(parameters):
    for name, parameter in parameters:
        if name.find('linearp') >= 0 or name.find('linearr') >= 0 or name.find('decoder') >= 0:
            print(name, 'is not fixed')

        else:
            print(name, 'is fixed')
            parameter.requires_grad = False

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def main():
    # imgs_file = os.path.join(datasets_root, args['imgs_file'])
    # imgs_file = os.path.join(datasets_root, 'video_saliency/train_all_DAFB3_seq_5f.txt')

    joint_transform = joint_transforms.Compose([
        # joint_transforms.ImageResize(args['image_size']),
        joint_transforms.RandomCrop(args['crop_size']),
        joint_transforms.RandomHorizontallyFlip(),
    ])

    # joint_transform = joint_transforms.Compose_single([
    #     joint_transforms.ImageResize_numpy(args['image_size']),
    #     joint_transforms.RandomCrop_numpy(args['crop_size']),
    #     joint_transforms.RandomHorizontallyFlip_numpy(),
    # ])

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()
    # train_set = ImageFolder(msra10k_path, joint_transform, img_transform, target_transform)
    train_set = WaterImage2Folder(args['imgs_file'], joint_transform, img_transform, target_transform)
    # train_set = WaterImage4Folder(args['imgs_file'], 256)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=16, shuffle=True)

    net = Water(dim=args['dim']).cuda(device_id).train()
    net.apply(weights_init)

    # discriminator = PatchDiscriminator(3).cuda(device_id).train()
    # vgg = models.vgg19(pretrained=True).features
    # for param in vgg.parameters():
    #     param.requires_grad_(False)
    # vgg.to(device_id).eval()
    # net = warp().cuda(device_id).train()
    remains = []
    for name, param in net.named_parameters():
        # if 'base' in name:
        #     # param.requires_grad = False
        #     bkbone.append(param)
        # # # elif 'flow' in name or 'linearf' in name or 'decoder' in name:
        # # #     print('flow related:', name)
        # # #     flow_modules.append(param)
        # # elif 'flow' in name or 'linearf' in name or 'decoder' in name:
        # #     print('decoder related:', name)
        # #     flow_modules.append(param)
        # else:
        # print('remains:', name)
        remains.append(param)
    # fix_parameters(net.named_parameters())
    # optimizer = optim.SGD([
    #     {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
    #      'lr': 2 * args['lr']},
    #     {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
    #      'lr': args['lr'], 'weight_decay': args['weight_decay']}
    # ], momentum=args['momentum'])

    optimizer = optim.Adam([{'params': remains, 'lr': args['lr']}],
                         betas=(0.9, 0.999))
    # optimizer_d = optim.Adam([{'params': discriminator.parameters()}],
    #                        lr=args['lr'], betas=(0.9, 0.999))
    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 0.5 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']
        optimizer.param_groups[2]['lr'] = args['lr']

    # net = load_part_of_model(net, 'pre-trained/SNet.pth', device_id=device_id)
    if len(args['pretrain']) > 0:
        print('pretrain model from ' + args['pretrain'])
        net = load_part_of_model(net, args['pretrain'], device_id=device_id)
        # fix_parameters(student.named_parameters())

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, None, optimizer, None, train_loader)


def train(net, discriminator, optimizer, optimizer_d, train_loader):
    curr_iter = args['last_iter']
    while True:

        # loss3_record = AvgMeter()
        # dataloader_iterator = iter(train_loader2)
        for i, data in enumerate(train_loader):

            optimizer.param_groups[0]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                  ) ** args['lr_decay']
            # optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
            #                                                 ) ** args['lr_decay']
            # optimizer.param_groups[2]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
            #                                                 ) ** args['lr_decay']
            #
            # optimizer.param_groups[3]['lr'] = 0.1 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
            #                                                 ) ** args['lr_decay']
            #
            # inputs, flows, labels, pre_img, pre_lab, cur_img, cur_lab, next_img, next_lab = data
            rgb, hsv, lab, target, lab_target = data
            # rgb, lab = data
            # data2 = next(dataloader_iterator)
            # inputs2, labels2 = data2
            # train_single(net, inputs, flows, labels, optimizer, curr_iter, teacher)


            train_single2(net, None, rgb, lab, target, lab_target, None, optimizer, None, curr_iter)
            curr_iter += 1

            if curr_iter % args['iter_save'] == 0:
                print('taking snapshot ...')
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))

            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
                return


def train_single2(net, discriminator, rgb, lab, target, lab_target, depth, optimizer, optimizer_d, curr_iter):
    rgb = Variable(rgb).cuda(device_id)

    lab = Variable(lab).cuda(device_id)

    # segmentation = Variable(depth).cuda(device_id)
    labels = Variable(target).cuda(device_id)
    labels_lab = Variable(lab_target).cuda(device_id)
    # labels_64 = F.interpolate(labels, size=[64, 64], mode='bilinear')
    # labels_32 = F.interpolate(labels, size=[32, 32], mode='bilinear')
    # labels_lab3 = Variable(lab_target).cuda(device_id)
    # patch = (1, 256 // 2 ** 5, 256 // 2 ** 5)
    # valid = Variable(torch.Tensor(np.ones((labels.size(0), *patch))), requires_grad=False).cuda(device_id) # 全1
    # fake = Variable(torch.Tensor(np.zeros((labels.size(0), *patch))), requires_grad=False).cuda(device_id)  # 全0

    get_random_cand = lambda: tuple(np.random.randint(args['choice']) for i in range(args['layers']))
    # get_random_cand2 = lambda: tuple(np.random.randint(args['choice2']) for i in range(args['layers2']))
    # print(get_random_cand2() + get_random_cand())
    optimizer.zero_grad()

    # final, mid_ab, final2, inter_rgb, inter_lab = net(rgb, hsv, lab, depth, get_random_cand())
    final = net(rgb, lab, get_random_cand())
    # fake_image = torch.cat([lab, final], dim=1)
    # pred_fake = discriminator(fake_image)
    # loss_GAN = criterion_gan(pred_fake, True)

    loss0 = criterion(final, labels)
    loss1 = criterion_l1(final, labels)

    # loss0_2 = criterion(final2, labels)
    # loss1_2 = criterion_l1(final2, labels)

    # loss_mid_ab = criterion(mid_ab, labels_lab)
    # loss_mid_ab = criterion_l1(mid_ab, labels_lab)

    # loss0_lab = criterion(final_lab, labels_lab)
    # loss1_lab = criterion_l1(final_lab, labels_lab)
    #
    # loss0_lab3 = criterion(final_lab3, labels_lab3)
    # loss1_lab3 = criterion_l1(final_lab3, labels_lab3)

    loss7 = criterion_perceptual(final, labels)
    # loss7_2 = criterion_perceptual(final2, labels)
    # loss11 = criterion_tv(final)

    # loss5 = criterion(final2, labels)
    # loss6 = criterion_l1(final2, labels)

    # loss2 = criterion(inter_rgb, labels)
    # loss4 = criterion(inter_lab, labels_lab)

    # loss2_1 = criterion_l1(inter_rgb, labels)
    # loss4_1 = criterion_l1(inter_lab, labels_lab)

    # loss8 = criterion_perceptual(inter_rgb, labels)
    # loss10 = criterion_perceptual(inter_lab, labels_lab)
    # texture_features = get_features(rgb, vgg)
    # target_features = get_features(labels, vgg)
    # content_loss = torch.mean((texture_features['relu5_4'] - target_features['relu5_4']) ** 2)

    # loss1_third = criterion(third, labels_32)
    # loss2_third = criterion_l1(third, labels_32)

    # loss1_second = criterion(second, labels_64)
    # loss2_second = criterion_l1(second, labels_64)
    # total_loss = 1 * loss0 + 0.25 * loss1
    total_loss = 1 * loss0 + 0.25 * loss1  \
                 + 0.2 * loss7
                 # + 1 * loss0_2 + 0.25 * loss1_2 + 0.25 * loss7_2 \
                 # + loss1_third + 0.25 * loss2_third + loss1_second + 0.25 * loss2_second \
                 # + 0.5 * loss_GAN
    # total_loss = 1 * loss0 + 0.25 * loss1  \
    #              + 1 * loss0_2 + 0.25 * loss1_2 + 0.25 * loss7_2 \
    # distill_loss = loss6_k + loss7_k + loss8_k

    # total_loss = total_loss + 0.1 * distill_loss
    total_loss.backward()
    optimizer.step()

    # discriminator
    # optimizer_d.zero_grad()
    # # Real loss
    # real_image = torch.cat([lab, labels], dim=1)
    # pred_real = discriminator(real_image)
    # loss_real = criterion_gan(pred_real, True)
    #
    # # Fake loss
    # final2 = final.detach()
    # fake_image = torch.cat([lab, final2], dim=1)
    # # third = third.detach()
    # # second = second.detach()
    # pred_fake1 = discriminator(fake_image)
    # loss_fake = criterion_gan(pred_fake1, False)
    #
    # # Total loss
    # loss_D = 0.5 * (loss_real + loss_fake)
    # # loss_D=loss_real
    #
    # loss_D.backward(retain_graph=True)
    # optimizer_d.step()

    print_log(total_loss, loss0, loss1, loss7, args['train_batch_size'], curr_iter, optimizer)

    return

def print_log(total_loss, loss0, loss1, loss2, batch_size, curr_iter, optimizer, type='normal'):
    total_loss_record.update(total_loss.data, batch_size)
    loss0_record.update(loss0.data, batch_size)
    loss1_record.update(loss1.data, batch_size)
    loss2_record.update(loss2.data, batch_size)
    # loss3_record.update(loss3.data, batch_size)
    # loss4_record.update(loss4.data, batch_size)
    log = '[iter %d][%s], [total loss %.5f], [loss0 %.5f], [loss1 %.5f], [loss2 %.5f] ' \
          '[lr %.13f]' % \
          (curr_iter, type, total_loss_record.avg, loss0_record.avg, loss1_record.avg, loss2_record.avg,
           optimizer.param_groups[0]['lr'])
    print(log)
    open(log_path, 'a').write(log + '\n')

if __name__ == '__main__':
    main()
