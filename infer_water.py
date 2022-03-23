import numpy as np
import os

import torch
from PIL import Image, ImageCms
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F

from config import ecssd_path, hkuis_path, pascals_path, sod_path, dutomron_path, \
    davis_path, fbms_path, mcl_path, uvsd_path, visal_path, vos_path, segtrack_path, davsod_path, saving_path
from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure

from utils.utils_mine import load_part_of_model2, MaxMinNormalization, calculate_psnr, calculate_ssim

import time
from matplotlib import pyplot as plt
from underwater_model.model_uw import Water
from skimage import img_as_ubyte
import cv2
import random


torch.manual_seed(2020)

# set which gpu to use
device_id = 0
torch.cuda.set_device(device_id)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = saving_path


exp_name = 'WaterEnhance_2022-03-21 02:33:56'

args = {
    'gnn': True,
    'snapshot': '200000',  # your snapshot filename (exclude extension name)
    'crf_refine': False,  # whether to use crf to refine results
    'save_results': True,  # whether to save the resulting masks
    'en_channels': [64, 128, 256],
    'de_channels': 128,
    'dim': 48,
    # 'input_size': (380, 380),
    # 'image_path': '/mnt/hdd/data/ty2/input_test',
    # 'depth_path': '/mnt/hdd/data/ty2/depth_test',
    # 'gt_path': '/mnt/hdd/data/ty2/gt_test',
    'image_path': '/home/ty/data/LSUI/test_input',
    'depth_path': '/home/ty/data/LSUI/depth_test',
    'gt_path': '/home/ty/data/LSUI/test_gt',
    'segment_path': '/home/ty/data/uw/segment_input_test',
    'dataset': 'LSUI',
    'start': 0
}
# 3, 6, 6, 5, 0, 9, 9, 1, 3, 6, 6, 1 underwater
img_transform = transforms.Compose([
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()

def read_testset(dataset, image_path):
    if dataset == 'UIEB':
        images = os.listdir(image_path)
        uieb = []
        for img in images:
            if img.find('deep') > 0:
                continue
            uieb.append(img[:-4])
        return uieb
    elif dataset == 'LSUI':
        images = os.listdir(image_path)
        lsui = []
        # random_list = random.sample(range(0, len(images)), 504)
        for img in images:
            lsui.append(img[:-4])
        return lsui
    else:
        images = os.listdir(image_path)
        s1000 = []
        for img in images:
            s1000.append(img[:-4])
        return s1000

def main(snapshot):
    # net = R3Net(motion='', se_layer=False, dilation=False, basic_model='resnet50')

    net = Water(dim=args['dim'])
    # net = warp()
    if snapshot is None:
        print ('load snapshot \'%s\' for testing' % args['snapshot'])
        # net.load_state_dict(torch.load('pretrained/R2Net.pth', map_location='cuda:2'))
        # net = load_part_of_model2(net, 'pretrained/R2Net.pth', device_id=2)
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),
                                   map_location='cuda:' + str(device_id)))
    else:
        print('load snapshot \'%s\' for testing' % snapshot)
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, snapshot + '.pth'),
                                       map_location='cuda:' + str(device_id)))
    net.eval()
    net.cuda()
    results = {}
    factor = 8
    image_names = read_testset(args['dataset'], args['image_path'])
    with torch.no_grad():
        psnr_record = AvgMeter()
        ssim_record = AvgMeter()
        for name in image_names:
            # precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]

            # img_list = [i_id.strip() for i_id in open(imgs_path)]
            # print(args['image_path'], name + '.jpg')
            img = Image.open(os.path.join(args['image_path'], name + '.jpg')).convert('RGB')
            img = np.array(img)


            w = img.shape[0]
            h = img.shape[1]
            # img = cv2.resize(img, (256, 256))
            # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img_var = Variable(img_transform(img).unsqueeze(0), volatile=True).cuda()
            # hsv_var = Variable(img_transform(hsv).unsqueeze(0), volatile=True).cuda()
            lab_var = Variable(img_transform(lab).unsqueeze(0), volatile=True).cuda()
            # segmentation = Variable(img_transform(segmentation).unsqueeze(0), volatile=True).cuda()

            h, w = img_var.shape[2], img_var.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            img_var = F.pad(img_var, (0, padw, 0, padh), 'reflect')
            lab_var = F.pad(lab_var, (0, padw, 0, padh), 'reflect')

            prediction, _, _ = net(img_var, lab_var, [3, 6, 6, 5, 0, 9, 9, 1, 3, 6, 6, 1])
            prediction = prediction[:, :, :h, :w]

            # prediction = torch.unsqueeze(prediction, 0)
            # print(torch.unique(prediction))
            # precision = to_pil(prediction.data.squeeze(0).cpu())
            # prediction = np.array(precision)
            # prediction = prediction.astype('float')
            prediction = torch.clamp(prediction, 0, 1)
            prediction = prediction.permute(0, 2, 3, 1).cpu().detach().numpy()
            prediction = np.squeeze(prediction)
            # prediction = prediction[:, :, ::-1]
            # plt.style.use('classic')
            # plt.subplot(1, 2, 1)
            # plt.imshow(prediction)
            # plt.subplot(1, 2, 2)
            # plt.imshow(precision2[0])
            # plt.show()

            # prediction = MaxMinNormalization(prediction, prediction.max(), prediction.min()) * 255.0
            # prediction = prediction.astype('uint8')
            # prediction = cv2.resize(prediction, (h, w))
            gt = Image.open(os.path.join(args['gt_path'], name + '.jpg')).convert('RGB')
            gt = np.asarray(gt)
            # gt = cv2.resize(gt, (256, 256))
            print(gt.shape, '-----', prediction.shape)
            # prediction = cv2.cvtColor(prediction * 255.0, cv2.COLOR_LAB2RGB)
            # print(np.unique(prediction))
            psnr = calculate_psnr(prediction * 255.0, gt)
            ssim = calculate_ssim(prediction * 255.0, gt)

            if args['save_results']:
                save_path = os.path.join(ckpt_path, exp_name, '%s' % (args['snapshot']), args['dataset'])
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                prediction = img_as_ubyte(prediction)
                # print(np.unique(prediction))
                cv2.imwrite(os.path.join(save_path, name + '.png'), cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR))
                # Image.fromarray(prediction).save(os.path.join(save_path, name + '.png'))

            psnr_record.update(psnr)
            ssim_record.update(ssim)

        results[args['dataset']] = {'PSNR': psnr_record.avg, 'SSIM': ssim_record.avg}

    print ('test results:')
    print (results)
    log_path = os.path.join('result_water_all.txt')
    if snapshot is None:
        open(log_path, 'a').write(exp_name + ' ' + args['snapshot'] + '\n')
    else:
        open(log_path, 'a').write(exp_name + ' ' + snapshot + '\n')
    open(log_path, 'a').write(str(results) + '\n\n')


if __name__ == '__main__':
    if args['start'] > 0:
        for i in range(args['start'], 204000, 4000):
            main(str(i))
    else:
        main(None)
