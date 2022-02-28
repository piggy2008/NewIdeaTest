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

    path = '/home/ty/data/rain/search_input'
    gt_path = '/home/ty/data/rain/search_gt'
    rain_data(path, gt_path)


    # plt.style.use('classic')
    # img = Image.open('/Users/tangyi/Downloads/Ucolor_final_model_corrected/input_test/15704.png')
    # img_gt = Image.open('/Users/tangyi/Downloads/Ucolor_final_model_corrected/gt_test/15704.png')
    # srgb_profile = ImageCms.createProfile("sRGB")
    # lab_profile = ImageCms.createProfile("LAB")
    # rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
    # lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, "LAB", "RGB")
    # lab = ImageCms.applyTransform(img, rgb2lab_transform)
    # lab = np.array(lab)
    # gt = ImageCms.applyTransform(img_gt, rgb2lab_transform)
    #
    # gt = np.asarray(gt)
    # gt[:, :, 0] = lab[:, :, 0]
    # print(gt.shape, '---', np.unique(gt))
    # Image.fromarray()
    # print(gt)
    # gt_img = Image.fromarray(gt).convert('RGB')
    # gt_save = ImageCms.applyTransform(gt_img, lab2rgb_transform)
    # gt_save.save('ckpt/15704.png')
    # gt_img_rgb = cv2.imread('/Users/tangyi/Downloads/Ucolor_final_model_corrected/gt_test/603_img_.png')
    # input_img_rgb = cv2.imread('/Users/tangyi/Downloads/Ucolor_final_model_corrected/input_test/603_img_.png')
    # l, a, b = cv2.split(img_lab)
    #
    # img_lab2 = cv2.cvtColor(gt_img_rgb, cv2.COLOR_BGR2LAB)
    # l2, a2, b2 = cv2.split(img_lab2)
    # # print(np.unique(l))
    # l2 = np.full_like(l2, 50)
    # ab = cv2.merge((l2, a2, b2))
    # img_rgb = cv2.cvtColor(ab, cv2.COLOR_LAB2BGR)

    # cv2.imwrite('ckpt/603_img_.png', img_rgb)
    # plt.subplot(1, 3, 1)
    # plt.imshow(lab[:, :, 0])
    # plt.subplot(1, 3, 2)
    # plt.imshow(l)
    # plt.subplot(1, 3, 3)
    # plt.imshow(b)
    # plt.show()

