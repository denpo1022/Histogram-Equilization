# Reference website: https://github.com/torywalker/histogram-equalizer/blob/master/HistogramEqualization.ipynb

import numpy as np
import matplotlib.pyplot as plt
import cv2


def cumulative_sum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)


def get_graynum(image, bins):
    num = np.zeros(bins)
    for pixel in image:
        num[pixel] += 1
    return num


def main():
    # read image as gray scale
    gray_lena = cv2.imread('lena.tiff', 0)

    # flatten the np.aaray into 1-dimention
    lena_flat = gray_lena.flatten()
    num = get_graynum(lena_flat, 256)

    # get the accumulated sum
    lena_CS = cumulative_sum(num)

    # numerator & denomenator
    numerator = (lena_CS - lena_CS.min()) * 255
    N = lena_CS.max() - lena_CS.min()

    # re-normalize the cdf
    lena_CS = numerator / N

    # get the value from cumulative sum for every index in flat, and set that as img_new
    lena_new = lena_CS[lena_flat]

    # reshape the new image into 2-dimention
    lena_new = np.reshape(lena_new, gray_lena.shape)

    # set up side-by-side image display
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)

    # display the oringinal image
    fig.add_subplot(2, 2, 1)
    plt.imshow(gray_lena, cmap='gray')

    # display the new image
    fig.add_subplot(2, 2, 2)
    plt.imshow(lena_new, cmap='gray')

    # display the oringinal histogram
    fig.add_subplot(2, 2, 3)
    plt.hist(lena_flat, bins=256)
    plt.title('Grayscale Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Amounts of Pixels')

    # display the histogram after equalization
    fig.add_subplot(2, 2, 4)
    plt.hist(lena_new.flatten(), bins=256)
    plt.title('After Histogram Equalization')
    plt.xlabel('Bins')
    plt.ylabel('Amounts of Pixels')

    plt.show()


if __name__ == "__main__":
    main()
