import os
import random
from PIL import Image

path = '/home/ty/data/LSUI/backup/input'
gt_path = '/home/ty/data/LSUI/backup/GT'

save_root_path = '/home/ty/data/LSUI'

images = os.listdir(path)

pick_test_num = random.sample(range(0, len(images)), 504)
lsui_test_images = []
for num in pick_test_num:
    lsui_test_images.append(images[num])
    image = Image.open(os.path.join(path, images[num])).convert('RGB')
    save_path = os.path.join(save_root_path, 'test_input')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image.save(os.path.join(save_path, images[num]))

    image = Image.open(os.path.join(gt_path, images[num])).convert('RGB')
    save_gt_path = os.path.join(save_root_path, 'test_gt')
    if not os.path.exists(save_gt_path):
        os.makedirs(save_gt_path)
    image.save(os.path.join(save_gt_path, images[num]))


for img in images:
    if img not in lsui_test_images:
        image = Image.open(os.path.join(path, img)).convert('RGB')
        save_path = os.path.join(save_root_path, 'train_input')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image.save(os.path.join(save_path, img))

        image = Image.open(os.path.join(gt_path, img)).convert('RGB')
        save_gt_path = os.path.join(save_root_path, 'train_gt')
        if not os.path.exists(save_gt_path):
            os.makedirs(save_gt_path)
        image.save(os.path.join(gt_path, img))
