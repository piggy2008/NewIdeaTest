import torch
from config import msra10k_path, video_train_path, datasets_root, video_seq_gt_path, video_seq_path, saving_path, visal_path
from datasets import ImageFlowFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import joint_transforms
from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from utils.utils_mine import MaxMinNormalization
import tqdm
import os
from PIL import Image
from torch.autograd import Variable
import numpy as np

# assert torch.cuda.is_available()

train_loader = None
device_id = 0

to_test = {'ViSal': os.path.join(visal_path, 'ViSal_test')}
gt_root = os.path.join(visal_path, 'GT')
flow_root = os.path.join(visal_path, 'flow')
imgs_path = os.path.join(visal_path, 'ViSal_test_single.txt')

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

    imgs_file = os.path.join(datasets_root, args['imgs_file'])
    # imgs_file = os.path.join(datasets_root, 'video_saliency/train_all_DAFB3_seq_5f.txt')

    joint_transform = joint_transforms.Compose([
        joint_transforms.ImageResize(args['image_size']),
        joint_transforms.RandomCrop(args['crop_size']),
        # joint_transforms.ColorJitter(hue=[-0.1, 0.1], saturation=0.05),
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.RandomRotate(10)
    ])

    input_size = (473, 473)

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()
    train_set = ImageFlowFolder(video_train_path, imgs_file, joint_transform, img_transform, target_transform)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)

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

    img_list = [i_id.strip() for i_id in open(imgs_path)]
    for name, root in to_test.items():
        precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
        mae_record = AvgMeter()
        video = ''
        for idx, img_name in enumerate(img_list):
            if video != img_name.split('/')[0]:
                video = img_name.split('/')[0]
                if name != 'VOS':
                    continue
                if name == 'VOS' or name == 'DAVSOD':
                    img = Image.open(os.path.join(root, img_name + '.png')).convert('RGB')
                else:
                    img = Image.open(os.path.join(root, img_name + '.jpg')).convert('RGB')
                flow = Image.open(os.path.join(flow_root, img_name + '.jpg')).convert('RGB')
                shape = img.size
                img = img.resize((args['crop_size'], args['crop_size']))
                flow = flow.resize((args['crop_size'], args['crop_size']))
                img_var = Variable(img_transform(img).unsqueeze(0), volatile=True).cuda()
                flow_var = Variable(img_transform(flow).unsqueeze(0), volatile=True).cuda()

                prediction2, prediction, _, _, _, _, _, _, _, _ = model(img_var, flow_var)
                prediction = torch.sigmoid(prediction)
            else:
                if name == 'VOS' or name == 'DAVSOD':
                    img = Image.open(os.path.join(root, img_name + '.png')).convert('RGB')
                else:
                    img = Image.open(os.path.join(root, img_name + '.jpg')).convert('RGB')
                if name == 'davis':
                    flow = Image.open(os.path.join(flow_root, img_name + '.jpg')).convert('RGB')
                else:
                    flow = Image.open(os.path.join(flow_root, img_name + '.jpg')).convert('RGB')
                # flow = Image.open(os.path.join(flow_root, img_name + '.jpg')).convert('RGB')
                shape = img.size
                img = img.resize((args['crop_size'], args['crop_size']))
                flow = flow.resize((args['crop_size'], args['crop_size']))
                img_var = Variable(img_transform(img).unsqueeze(0), volatile=True).cuda()
                flow_var = Variable(img_transform(flow).unsqueeze(0), volatile=True).cuda()

                prediction2, prediction, prediction3, _, _, _, _, _, _, _ = model(img_var, flow_var, architecture=cand)
                prediction = torch.sigmoid(prediction3)

            precision = transforms.ToPILImage()(prediction.data.squeeze(0).cpu())
            precision = precision.resize(shape)
            prediction = np.array(precision)
            prediction = prediction.astype('float')

            prediction = MaxMinNormalization(prediction, prediction.max(), prediction.min()) * 255.0
            prediction = prediction.astype('uint8')
            # if args['crf_refine']:
            #     prediction = crf_refine(np.array(img), prediction)

            gt = np.array(Image.open(os.path.join(gt_root, img_name + '.png')).convert('L'))
            precision, recall, mae = cal_precision_recall_mae(prediction, gt)
            for pidx, pdata in enumerate(zip(precision, recall)):
                p, r = pdata
                precision_record[pidx].update(p)
                recall_record[pidx].update(r)
            mae_record.update(mae)

        fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                [rrecord.avg for rrecord in recall_record])

        # results[name] = {'fmeasure': fmeasure, 'mae': mae_record.avg}
        # print('fmeasure:', fmeasure)
        # top1, top5 = top1 / total, top5 / total
        #
        # top1, top5 = 1 - top1 / 100, 1 - top5 / 100

        print('fmeasure: {:.5f} MAE: {:.5f}'.format(fmeasure, (1 - mae_record.avg)))

        return fmeasure, 1 - mae_record.avg

