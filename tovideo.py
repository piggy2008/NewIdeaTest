import cv2
import os

image_folder = '/home/ty/Downloads/save'
image_folder2 = '/home/ty/Downloads/save_enhanced'
video_name = 'video3.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images2 = [img for img in os.listdir(image_folder2) if img.endswith(".jpg")]
images.sort(key=lambda x: int(x.split('frame')[1].split('.jpg')[0]))
images2.sort(key=lambda x: int(x.split('frame')[1].split('.jpg')[0]))

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 24, (width * 2,height))

for image in images:
    img1 = cv2.imread(os.path.join(image_folder, image))
    img2 = cv2.imread(os.path.join(image_folder2, image))
    cv2.putText(img1, 'original', (width - 100, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_4)
    cv2.putText(img2, 'enhanced', (width - 130, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_4)
    img3 = cv2.hconcat([img1, img2])

    video.write(img3)
    # if image in images2:
    #     print('same', image)
    # else:
    #     print('no', image)

cv2.destroyAllWindows()
video.release()