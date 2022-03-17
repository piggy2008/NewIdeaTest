import numpy as np
import os

import torch
from PIL import Image, ImageCms
from torch.autograd import Variable
from torchvision import transforms

from config import ecssd_path, hkuis_path, pascals_path, sod_path, dutomron_path, \
    davis_path, fbms_path, mcl_path, uvsd_path, visal_path, vos_path, segtrack_path, davsod_path, saving_path
from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure

from utils.utils_mine import load_part_of_model2, MaxMinNormalization, calculate_psnr, calculate_ssim
import time
from matplotlib import pyplot as plt
from underwater_model.model_SPOS import Water
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

exp_name = 'WaterEnhance_2021-12-23 16:13:35'
args = {
    'gnn': True,
    'snapshot': '64000',  # your snapshot filename (exclude extension name)
    'crf_refine': False,  # whether to use crf to refine results
    'save_results': True,  # whether to save the resulting masks
    # 'input_size': (380, 380),
    # 'image_path': '/mnt/hdd/data/ty2/input_test',
    # 'depth_path': '/mnt/hdd/data/ty2/depth_test',
    # 'gt_path': '/mnt/hdd/data/ty2/gt_test',
    'image_path': '/home/ty/data/uw/input_test',
    'depth_path': '/home/ty/data/uw/depth_test',
    'gt_path': '/home/ty/data/uw/gt_test',
    'segment_path': '/home/ty/data/uw/segment_input_test',
    'dataset': 'UIEB',
    'start': 0
}

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
        random_list = random.sample(range(0, len(images)), 504)
        for num in random_list:
            lsui.append(images[num][:-4])
        return lsui
    else:
        images = os.listdir(image_path)
        s1000 = []
        for img in images:
            if img.find('deep') > 0:
                s1000.append(img[:-4])
        return s1000

def main(snapshot):
    # net = R3Net(motion='', se_layer=False, dilation=False, basic_model='resnet50')

    net = Water()
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
    image_names = read_testset(args['dataset'], args['image_path'])
    with torch.no_grad():
        psnr_record = AvgMeter()
        ssim_record = AvgMeter()
        for name in image_names:
            # precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]

            # img_list = [i_id.strip() for i_id in open(imgs_path)]
            img = Image.open(os.path.join(args['image_path'], name + '.png')).convert('RGB')
            img = np.array(img)
            # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            # fv = Image.open(os.path.join(args['segment_path'], 'FV', name + '.bmp')).convert('L')
            # hd = Image.open(os.path.join(args['segment_path'], 'HD', name + '.bmp')).convert('L')
            # ri = Image.open(os.path.join(args['segment_path'], 'RI', name + '.bmp')).convert('L')
            # ro = Image.open(os.path.join(args['segment_path'], 'RO', name + '.bmp')).convert('L')
            # wr = Image.open(os.path.join(args['segment_path'], 'WR', name + '.bmp')).convert('L')

            # fv = cv2.resize(np.array(fv), (224, 224))
            # hd = cv2.resize(np.array(hd), (224, 224))
            # ri = cv2.resize(np.array(ri), (224, 224))
            # ro = cv2.resize(np.array(ro), (224, 224))
            # wr = cv2.resize(np.array(wr), (224, 224))
            # segmentation = np.stack((fv, hd, ri, ro, wr), axis=-1)
            w = img.shape[0]
            h = img.shape[1]
            img = cv2.resize(img, (256, 256))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img_var = Variable(img_transform(img).unsqueeze(0), volatile=True).cuda()
            hsv_var = Variable(img_transform(hsv).unsqueeze(0), volatile=True).cuda()
            lab_var = Variable(img_transform(lab).unsqueeze(0), volatile=True).cuda()
            # segmentation = Variable(img_transform(segmentation).unsqueeze(0), volatile=True).cuda()
            prediction, prediction2, hsv_side, lab_side, _, _ = net(img_var, lab_var, [3, 3, 3, 3])
            # prediction = torch.unsqueeze(prediction, 0)
            # print(torch.unique(prediction))
            # precision = to_pil(prediction.data.squeeze(0).cpu())
            # prediction = np.array(precision)
            # prediction = prediction.astype('float')
            prediction = torch.clamp(prediction2, 0, 1)
            prediction = prediction.permute(0, 2, 3, 1).cpu().detach().numpy()
            prediction = np.squeeze(prediction)
            # plt.style.use('classic')
            # plt.subplot(1, 2, 1)
            # plt.imshow(prediction)
            # plt.subplot(1, 2, 2)
            # plt.imshow(precision2[0])
            # plt.show()

            # prediction = MaxMinNormalization(prediction, prediction.max(), prediction.min()) * 255.0
            # prediction = prediction.astype('uint8')
            prediction = cv2.resize(prediction, (w, h))
            gt = Image.open(os.path.join(args['gt_path'], name + '.png')).convert('RGB')
            gt = np.asarray(gt)
            # gt = cv2.resize(gt, (256, 256))
            print(gt.shape, '-----', prediction.shape)
            psnr = calculate_psnr(prediction * 255.0, gt)
            ssim = calculate_ssim(prediction * 255.0, gt)

            if args['save_results']:
                save_path = os.path.join(ckpt_path, exp_name, '%s' % (args['snapshot']), args['dataset'])
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                prediction = img_as_ubyte(prediction)
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
