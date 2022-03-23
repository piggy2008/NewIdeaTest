# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

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
    #     image = Image.open(os.path.join(depth_path, img + '.png'))
    #     image.save(os.path.join(save_depth_root, img + '.png'))
    # path = '/Users/tangyi/Downloads/segment_train_uw'
    # image_name = '3_img_.bmp'
    #
    # fv = cv2.imread(os.path.join(path, 'FV', image_name), 0)
    # hd = cv2.imread(os.path.join(path, 'HD', image_name), 0)
    # ri = cv2.imread(os.path.join(path, 'RI', image_name), 0)
    # ro = cv2.imread(os.path.join(path, 'RO', image_name), 0)
    # wr = cv2.imread(os.path.join(path, 'WR', image_name), 0)
    # con = np.stack((fv, hd, ri ,ro, wr), axis=0)
    # con2 = cv2.resize(fv[:, :, np.newaxis], (112, 112))
    #
    # print(con2.shape, '--', con2.ndim)
    # con2 = np.flip(con, -1)
    # plt.subplot(1, 2, 1)
    # plt.imshow(con2[0, :, :])
    # plt.subplot(1, 2, 2)
    # plt.imshow(fv)
    # plt.show()
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

    path1 = '/Users/tangyi/Downloads/Ucolor_final_model_corrected/input_test2'
    path2 = '/Users/tangyi/Downloads/Ucolor_final_model_corrected/real_90_input'
    img1 = os.listdir(path1)
    img1.sort()
    img2 = os.listdir(path2)
    img2.sort()
    test_list = []
    for img in img1:
        image = cv2.imread(os.path.join(path1, img))
        img_dict = {}
        img_dict['name'] = img
        img_dict['shape'] = image.shape
        img_dict['image'] = image[:25, :25, 0]
        test_list.append(img_dict)

    total_list = []
    for img in img2:
        image = cv2.imread(os.path.join(path2, img))
        img_dict = {}
        img_dict['name'] = img
        img_dict['shape'] = image.shape
        img_dict['image'] = image[:25, :25, 0]
        total_list.append(img_dict)

    for dict in test_list:
        shape = dict['shape']
        for dict_total in total_list:
            total_shape = dict_total['shape']
            if shape == total_shape:
                image = dict['image']
                image_total = dict_total['image']
                if (image_total - image) == 0:
                    print(dict['name'], '-----', dict_total['name'])

