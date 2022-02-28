import os
import numpy as np
from PIL import Image
from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from utils.utils_mine import load_part_of_model2, MaxMinNormalization, calculate_psnr, calculate_ssim
from matplotlib import pyplot as plt


ckpt_path = '/home/ty/code/NewIdeaTest/ckpt/WaterEnhance_2022-02-21 21:36:14/200000/rain'
gt_path = '/home/ty/data/rain/test_gt'

psnr_record = AvgMeter()
ssim_record = AvgMeter()
results = {}

images = os.listdir(ckpt_path)
images.sort()
image_names = []
psnr_list = []
for name in images:
    img = Image.open(os.path.join(ckpt_path, name)).convert('YCbCr')
    img = np.array(img)

    gt = Image.open(os.path.join(gt_path, name)).convert('YCbCr')
    gt = np.array(gt)
    psnr = calculate_psnr(img[:, :, 0], gt[:, :, 0])
    ssim = calculate_ssim(img[:, :, 0], gt[:, :, 0])
    psnr_record.update(psnr)
    ssim_record.update(ssim)
    # each = {'name': name, 'psnr': psnr}
    image_names.append(name)
    psnr_list.append(psnr)

results['rain'] = {'PSNR': psnr_record.avg, 'SSIM': ssim_record.avg}
print(results)
# print(psnr_list)
# print(psnr_record)
# bar = (
#     Bar()
#     .add_xaxis(image_names)
#     .add_yaxis('psnr', psnr_list)
#     .reversal_axis()
#     .set_series_opts(label_opts=opts.LabelOpts(position="right"))
#     .set_global_opts(title_opts=opts.TitleOpts(title="bar"))
#     .set_series_opts(
#         label_opts=opts.LabelOpts(is_show=False),
#         markline_opts=opts.MarkLineOpts(
#             data=[
#                 opts.MarkLineItem(type_="min", name="min"),
#                 opts.MarkLineItem(type_="max", name="max"),
#                 opts.MarkLineItem(type_="average", name="avg"),
#             ]
#         ),
#     )
# )
# bar.render('psnr.html')

# plt.yticks(rotation=-15)
# plt.show()


