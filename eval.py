'''
Metrics for unferwater image quality evaluation.
Author: Xuelei Chen 
Email: chenxuelei@hotmail.com
Usage:
python evaluate.py RESULT_PATH
'''
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import math
import sys
from skimage import io, color, filters
import os
import math
import cv2


def nmetrics(a):
    rgb = a
    lab = color.rgb2lab(a)
    gray = color.rgb2gray(a)
    # UCIQE
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    l = lab[:, :, 0]

    # 1st term
    chroma = (lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2) ** 0.5
    uc = np.mean(chroma)
    sc = (np.mean((chroma - uc) ** 2)) ** 0.5

    # 2nd term
    top = np.int(np.round(0.01 * l.shape[0] * l.shape[1]))
    sl = np.sort(l, axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[::top]) - np.mean(sl[::top])

    # 3rd term
    satur = []
    chroma1 = chroma.flatten()
    l1 = l.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0:
            satur.append(0)
        elif l1[i] == 0:
            satur.append(0)
        else:
            satur.append(chroma1[i] / l1[i])

    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753

    # 1st term UICM
    rg = rgb[:, :, 0] - rgb[:, :, 1]
    yb = (rgb[:, :, 0] + rgb[:, :, 1]) / 2 - rgb[:, :, 2]
    rgl = np.sort(rg, axis=None)
    ybl = np.sort(yb, axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = np.int(al1 * len(rgl))
    T2 = np.int(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr - uyb) ** 2)

    uicm = -0.0268 * np.sqrt(urg ** 2 + uyb ** 2) + 0.1586 * np.sqrt(s2rg + s2yb)

    # 2nd term UISM (k1k2=8x8)
    Rsobel = rgb[:, :, 0] * filters.sobel(rgb[:, :, 0])
    Gsobel = rgb[:, :, 1] * filters.sobel(rgb[:, :, 1])
    Bsobel = rgb[:, :, 2] * filters.sobel(rgb[:, :, 2])

    Rsobel = np.round(Rsobel).astype(np.uint8)
    Gsobel = np.round(Gsobel).astype(np.uint8)
    Bsobel = np.round(Bsobel).astype(np.uint8)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    # 3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm
    return uiqm, uciqe


def eme(ch, blocksize=8):
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)

    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i + 1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j + 1) * blocksize
            else:
                yrb = ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]

            blockmin = np.float(np.min(block))
            blockmax = np.float(np.max(block))

            # # old version
            # if blockmin == 0.0: eme += 0
            # elif blockmax == 0.0: eme += 0
            # else: eme += w * math.log(blockmax / blockmin)

            # new version
            if blockmin == 0: blockmin += 1
            if blockmax == 0: blockmax += 1
            eme += w * math.log(blockmax / blockmin)
    return eme


def plipsum(i, j, gamma=1026):
    return i + j - i * j / gamma


def plipsub(i, j, k=1026):
    return k * (i - j) / (k - j)


def plipmult(c, j, gamma=1026):
    return gamma - gamma * (1 - j / gamma) ** c


def logamee(ch, blocksize=8):
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)

    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i + 1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j + 1) * blocksize
            else:
                yrb = ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]
            blockmin = np.float(np.min(block))
            blockmax = np.float(np.max(block))

            top = plipsub(blockmax, blockmin)
            bottom = plipsum(blockmax, blockmin)

            m = top / bottom
            if m == 0.:
                s += 0
            else:
                s += (m) * np.log(m)

    return plipmult(w, s)

def compute_uiqm(img):
    # img = cv2.imread('MSRCR_retinex.tif')  # 读取图像
    b, r, g = cv2.split(img)
    RG = r - g
    YB = (r + g) / 2 - b
    m, n, o = np.shape(img)  # img为三维 rbg为二维 o并未用到
    K = m * n
    alpha_L = 0.1
    alpha_R = 0.1  # 参数α 可调
    T_alpha_L = math.ceil(alpha_L * K)  # 向上取整 #表示去除区间
    T_alpha_R = math.floor(alpha_R * K)  # 向下取整

    RG_list = RG.flatten()  # 二维数组转一维（方便计算）
    RG_list = sorted(RG_list)  # 排序
    sum_RG = 0  # 计算平均值
    for i in range(T_alpha_L + 1, K - T_alpha_R):
        sum_RG = sum_RG + RG_list[i]
    U_RG = sum_RG / (K - T_alpha_R - T_alpha_L)
    squ_RG = 0  # 计算方差
    for i in range(K):
        squ_RG = squ_RG + np.square(RG_list[i] - U_RG)
    sigma2_RG = squ_RG / K

    # YB和RG计算一样
    YB_list = YB.flatten()
    YB_list = sorted(YB_list)
    sum_YB = 0
    for i in range(T_alpha_L + 1, K - T_alpha_R):
        sum_YB = sum_YB + YB_list[i]
    U_YB = sum_YB / (K - T_alpha_R - T_alpha_L)
    squ_YB = 0
    for i in range(K):
        squ_YB = squ_YB + np.square(YB_list[i] - U_YB)
    sigma2_YB = squ_YB / K

    uicm = -0.0268 * np.sqrt(np.square(U_RG) + np.square(U_YB)) + 0.1586 * np.sqrt(sigma2_RG + sigma2_YB)
    print(uicm)
    return uicm


def main():
    result_path = sys.argv[1]

    result_dirs = os.listdir(result_path)

    sumuiqm, sumuciqe = 0., 0.
    sumuicm = 0.
    N = 0
    for imgdir in result_dirs:
        if '.png' in imgdir:
            # corrected image
            corrected = cv2.imread(os.path.join(result_path, imgdir))

            uiqm, uciqe = nmetrics(corrected)
            uicm = compute_uiqm(corrected)

            sumuiqm += uiqm
            sumuciqe += uciqe
            sumuicm += uicm
            N += 1

            with open(os.path.join(result_path, 'metrics.txt'), 'a') as f:
                f.write('{}: uiqm={} uciqe={}\n'.format(imgdir, uiqm, uciqe))

    muiqm = sumuiqm / N
    muciqe = sumuciqe / N
    muicm = sumuicm / N

    with open(os.path.join(result_path, 'metrics.txt'), 'a') as f:
        f.write('Average: uiqm={} uciqe={}\n'.format(muiqm, muciqe))

    print('Average: uiqm={} uciqe={}, uicm={}\n'.format(muiqm, muciqe, muicm))


if __name__ == '__main__':
    main()