import cv2
import numpy as np
import math
import argparse


def reduce_gray_levels(image, intensity):
    if intensity < 2 or intensity > 255:
        raise ValueError('Intensity value should be between 2 and 255')
    if math.ceil(math.log2(intensity)) != math.floor(math.log2(intensity)):
        raise ValueError('Intensity value should be an integer power of 2')

    # Converting image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #  Reduce the gray level using this simple formula-
    # I_new=floor(I_old/k)*k
    # where I= pixel value and floor function rounds the value to the nearest smallest integer.
    # k is calculated as  k=256/required_gray_levels
    k = 256 / intensity
    gray_transformed = (np.floor(gray / k) * k).astype('uint8')
    return gray_transformed


def show_image(title, image):
    cv2.imshow(str(title), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(
        description='Evaluate expression'
    )
    p.add_argument('img', type=str, help='image path')
    p.add_argument('intensity', type=int, help='desired number of  intensity levels')
    args = p.parse_args()
    # Reading an image in default mode
    image = cv2.imread(args.img)
    transformed = reduce_gray_levels(image, args.intensity)
    show_image('Original image', image)
    show_image('Transformed image',  transformed)


if __name__ == '__main__':
    main()


