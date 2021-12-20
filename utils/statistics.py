import os
# from infer_water import read_testset
from matplotlib import pyplot as plt
import cv2
image_path = '/Users/tangyi/Downloads/Ucolor_final_model_corrected/input_test'
depth_path = '/Users/tangyi/Downloads/Ucolor_final_model_corrected/depth_test'
gt_path = '/Users/tangyi/Downloads/Ucolor_final_model_corrected/gt_test'
results_path = '../ckpt/48000'
dataset = 'UIEB'

image_name = '210_img_.png'
# image_names = read_testset(dataset, image_path)

enhanced_img = cv2.imread(os.path.join(results_path, dataset, image_name))
original_img = cv2.imread(os.path.join(image_path, image_name))
gt_img = cv2.imread(os.path.join(gt_path, image_name))
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.title('original image')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
plt.title('enhanced image')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
plt.title('GT image')

color = ['blue', 'springgreen', 'red']
plt.subplot(2, 3, 4)
for i in [0, 1, 2]:
    hist = cv2.calcHist([original_img], [i], None, [256], [0.0, 255.0])
    plt.plot(hist, color[i])
    plt.title('original hist')

plt.subplot(2, 3, 5)
for i in [0, 1, 2]:
    hist = cv2.calcHist([enhanced_img], [i], None, [256], [0.0, 255.0])
    plt.plot(hist, color[i])
    plt.title('enhanced hist')

plt.subplot(2, 3, 6)
for i in [0, 1, 2]:
    hist = cv2.calcHist([gt_img], [i], None, [256], [0.0, 255.0])
    plt.plot(hist, color[i])
    plt.title('gt hist')
plt.show()
