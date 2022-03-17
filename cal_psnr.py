import os
import numpy as np
from PIL import Image
from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from utils.utils_mine import load_part_of_model2, MaxMinNormalization, calculate_psnr, calculate_ssim
from matplotlib import pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Bar

ckpt_path = './ckpt/UIEB'
gt_path = '/Users/tangyi/Downloads/Ucolor_final_model_corrected/gt_test'

psnr_record = AvgMeter()
ssim_record = AvgMeter()
results = {}

images = os.listdir(ckpt_path)
images.sort()
image_names = []
psnr_list = []
for name in images:
    img = Image.open(os.path.join(ckpt_path, name)).convert('RGB')
    img = np.array(img)

    gt = Image.open(os.path.join(gt_path, name)).convert('RGB')
    gt = np.array(gt)
    psnr = calculate_psnr(img, gt)
    ssim = calculate_ssim(img, gt)
    psnr_record.update(psnr)
    ssim_record.update(ssim)
    # each = {'name': name, 'psnr': psnr}
    image_names.append(name)
    psnr_list.append(psnr)

results['UIEB'] = {'PSNR': psnr_record.avg, 'SSIM': ssim_record.avg}
# print(psnr_list)
# print(psnr_record)
bar = (
    Bar()
    .add_xaxis(image_names)
    .add_yaxis('psnr', psnr_list)
    .reversal_axis()
    .set_series_opts(label_opts=opts.LabelOpts(position="right"))
    .set_global_opts(title_opts=opts.TitleOpts(title="bar"))
    .set_series_opts(
        label_opts=opts.LabelOpts(is_show=False),
        markline_opts=opts.MarkLineOpts(
            data=[
                opts.MarkLineItem(type_="min", name="min"),
                opts.MarkLineItem(type_="max", name="max"),
                opts.MarkLineItem(type_="average", name="avg"),
            ]
        ),
    )
)
bar.render('psnr.html')

# plt.yticks(rotation=-15)
# plt.show()


