# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
import cv2
from PIL import Image, ImageCms
from matplotlib import pyplot as plt
import numpy as np
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def rain_data(path, gt_path):
    images = os.listdir(path)
    images.sort()
    gt_imgs = os.listdir(gt_path)
    gt_imgs.sort()
    for i, j in zip(images, gt_imgs):
        print(i, '----', j)
        os.rename(os.path.join(path, i), os.path.join(path, i.split('-')[1]))
        os.rename(os.path.join(gt_path, j), os.path.join(gt_path, j.split('-')[1]))

def pickup_image(path, dataset):
    if dataset == 'UIEB':
        images = os.listdir(path)
        uieb = []
        for img in images:
            if img.find('deep') > 0:
                continue
            uieb.append(img[:-4])
        return uieb
    else:
        images = os.listdir(path)
        s1000 = []
        for img in images:
            if img.find('deep') > 0:
                s1000.append(img[:-4])
        return s1000

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    import cv2
    from PIL import Image, ImageCms
    from matplotlib import pyplot as plt
    import numpy as np
    # path = '/home/ty/data/uw/input_train'
    # gt_path = '/home/ty/data/uw/gt_train'
    # depth_path = '/home/ty/data/uw/depth_train'
    # save_input_root = '/home/ty/data/uw/input_train_uw'
    # save_gt_root = '/home/ty/data/uw/gt_train_uw'
    # save_depth_root = '/home/ty/data/uw/depth_train_uw'
    # if not os.path.exists(save_input_root):
    #     os.makedirs(save_input_root)
    # if not os.path.exists(save_gt_root):
    #     os.makedirs(save_gt_root)
    # if not os.path.exists(save_depth_root):
    #         os.makedirs(save_depth_root)
    # image_names = pickup_image(path, 'UIEB')
    # for img in image_names:
    #     image = Image.open(os.path.join(path, img + '.png'))
    #     image.save(os.path.join(save_input_root, img + '.png'))
    #
    #     image = Image.open(os.path.join(gt_path, img + '.png'))
    #     image.save(os.path.join(save_gt_root, img + '.png'))
    #
    #     image = Image.open(os.path.join(depth_path, img + '.png_depth_estimate.png'))
    #     image.save(os.path.join(save_depth_root, img + '.png_depth_estimate.png'))

    path = '/home/ty/data/LSUI/test_input'
    gt_path = '/home/ty/data/LSUI/test_gt'

    imgs = os.listdir(path)
    for img in imgs:
        # print(img)
        image = Image.open(os.path.join(path, img)).convert('RGB')
        gt = Image.open(os.path.join(gt_path, img)).convert('RGB')
        w, h = image.size
        if w > 800 or h > 800:
            print(img)
            image = image.resize((int(w/2), int(h/2)), Image.BILINEAR)
            gt = gt.resize((int(w / 2), int(h / 2)), Image.BILINEAR)

        image.save(os.path.join(path, img))
        gt.save(os.path.join(gt_path, img))

    # path = '/home/ty/data/test'
    # images = os.listdir(path)
    # images.sort()
    #
    # path2 = '/home/ty/data/LSUI/backup/input'
    # path2_gt = '/home/ty/data/LSUI/backup/GT'
    #
    # save_test2 = '/home/ty/data/LSUI/test_input'
    # save_test2_gt = '/home/ty/data/LSUI/test_gt'
    #
    # save_train2 = '/home/ty/data/LSUI/train_input'
    # save_train2_gt = '/home/ty/data/LSUI/train_gt'
    #
    # all_images = os.listdir(path2)
    # all_images.sort()
    #
    # for img in all_images:
    #     if img in images:
    #         test_img = Image.open(os.path.join(path2, img)).convert('RGB')
    #         test_gt_img = Image.open(os.path.join(path2_gt, img)).convert('RGB')
    #         test_img.save(os.path.join(save_test2, img))
    #         test_gt_img.save(os.path.join(save_test2_gt, img))
    #
    #     else:
    #         train_img = Image.open(os.path.join(path2, img)).convert('RGB')
    #         train_gt_img = Image.open(os.path.join(path2_gt, img)).convert('RGB')
    #         train_img.save(os.path.join(save_train2, img))
    #         train_gt_img.save(os.path.join(save_train2_gt, img))




