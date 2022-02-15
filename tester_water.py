import torch
from config import msra10k_path, video_train_path, datasets_root, video_seq_gt_path, video_seq_path, saving_path, visal_path
from datasets import ImageFlowFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import joint_transforms
from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from utils.utils_mine import MaxMinNormalization, calculate_psnr, calculate_ssim
import tqdm
import os
from PIL import Image, ImageCms
from torch.autograd import Variable
import numpy as np
from infer_water import read_testset
import cv2


# assert torch.cuda.is_available()

train_loader = None
device_id = 0

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


@no_grad_wrapper
def get_cand_err(model, cand, args):
    # global train_loader

    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    target_transform = transforms.ToTensor()

    max_train_iters = args['max_train_iters']

    # print('clear bn statics....')
    # for m in model.modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         m.running_mean = torch.zeros_like(m.running_mean)
    #         m.running_var = torch.ones_like(m.running_var)

    # print('train bn with training set (BN sanitize) ....')
    # model.cuda(device_id).train()
    # dataloader_iterator = iter(train_loader)
    # for step in tqdm.tqdm(range(max_train_iters)):
    #     data = next(dataloader_iterator)
    #     inputs, flows, labels = data
    #     inputs = Variable(inputs).cuda(device_id)
    #     flows = Variable(flows).cuda(device_id)
    #     labels = Variable(labels).cuda(device_id)
    #     out1u, out2u, out2r, out3r, out4r, out5r = model(inputs, architecture=cand)
    #     # print('training:', step)
    #     del data, out1u, out2u, out2r, out3r, out4r, out5r

    print('starting test....')
    model.cuda(device_id).eval()
    image_names = read_testset(args['dataset'], args['image_path'])
    psnr_record = AvgMeter()
    ssim_record = AvgMeter()
    for name in image_names:
        # img_list = [i_id.strip() for i_id in open(imgs_path)]
        img = Image.open(os.path.join(args['image_path'], name + '.png')).convert('RGB')

        # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = np.array(img)
        img = cv2.resize(img, (256, 256))
        # depth = Image.open(os.path.join(args['depth_path'], name + '.png_depth_estimate.png')).convert('L')
        fv = Image.open(os.path.join(args['segment_path'], 'FV', name + '.bmp')).convert('L')
        hd = Image.open(os.path.join(args['segment_path'], 'HD', name + '.bmp')).convert('L')
        ri = Image.open(os.path.join(args['segment_path'], 'RI', name + '.bmp')).convert('L')
        ro = Image.open(os.path.join(args['segment_path'], 'RO', name + '.bmp')).convert('L')
        wr = Image.open(os.path.join(args['segment_path'], 'WR', name + '.bmp')).convert('L')

        fv = cv2.resize(np.array(fv), (256, 256))
        hd = cv2.resize(np.array(hd), (256, 256))
        ri = cv2.resize(np.array(ri), (256, 256))
        ro = cv2.resize(np.array(ro), (256, 256))
        wr = cv2.resize(np.array(wr), (256, 256))
        segmentation = np.stack((fv, hd, ri, ro, wr), axis=-1)


        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # srgb_profile = ImageCms.createProfile("sRGB")
        # lab_profile = ImageCms.createProfile("LAB")
        # rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        img_var = Variable(img_transform(img).unsqueeze(0), volatile=True).cuda()
        hsv_var = Variable(img_transform(hsv).unsqueeze(0), volatile=True).cuda()
        lab_var = Variable(img_transform(lab).unsqueeze(0), volatile=True).cuda()

        segmentation_var = Variable(img_transform(segmentation).unsqueeze(0), volatile=True).cuda()
        
        # temp = (1, 1, 0)

        prediction, prediction2, _, _ = model(img_var, lab_var, cand)

        # prediction = torch.unsqueeze(prediction, 0)
        # print(torch.unique(prediction))
        # precision = to_pil(prediction.data.squeeze(0).cpu())
        # prediction = np.array(precision)
        # prediction = prediction.astype('float')
        prediction = torch.clamp(prediction2, 0, 1)
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

        gt = Image.open(os.path.join(args['gt_path'], name + '.png')).convert('RGB')
        gt = np.asarray(gt)
        gt = cv2.resize(gt, (256, 256))
        # print(gt.shape, '-----', prediction.shape)
        psnr = calculate_psnr(prediction * 255.0, gt)
        ssim = calculate_ssim(prediction * 255.0, gt)


        psnr_record.update(psnr)
        ssim_record.update(ssim)


    print('psnr: {:.5f} ssim: {:.5f}'.format(psnr_record.avg, ssim_record.avg))

    return psnr_record.avg, ssim_record.avg

