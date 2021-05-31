import numpy as np
import cv2


def reduce_img(image, fac=1):
    height = image.shape[0]  # define the height of the image
    width = image.shape[1]  # define the width of the image

    # define output image
    output = np.zeros((height, width, image.shape[2]), dtype='uint8')
    avg = fac * fac

    for i in range(0, height, fac):
        for j in range(0, width, fac):
            output[i:i + fac, j: j + fac, 0] = np.sum(image[i: i + fac, j: j + fac, 0]) / avg
            output[i:i + fac, j: j + fac, 1] = np.sum(image[i: i + fac, j: j + fac, 1]) / avg
            output[i:i + fac, j: j + fac, 2] = np.sum(image[i: i + fac, j: j + fac, 2]) / avg
    return output


def show_image(title, image):
    cv2.imshow(str(title), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


path = 'cat.jpg'
fac = 10


def main(path, fac):
    image = cv2.imread(path)
    transformed = reduce_img(image, fac)
    show_image('Transformed image', transformed)


if __name__ == '__main__':
    main(path, fac)
