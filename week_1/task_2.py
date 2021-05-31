import numpy as np
import cv2


def spatial_average(image, filter_size):
    """"
    Simple spatial average of image pixels.

    Input:
    ------
    image : array_like image.
    filter_size: tuple of filter size.

    Output:
    -------
    array_like grayscale processed image.
    """
    # Converting image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Image size
    m, n = image.shape
    # Filter size
    p, q = filter_size
    # Padding
    pad_rows = (p - 1) // 2
    pad_columns = (q - 1) // 2
    # Padded matrix
    pad = np.zeros((m + 2 * pad_rows, n + 2 * pad_columns))
    pad[pad_rows:m + pad_rows, pad_columns:n + pad_columns] = image
    # Number of pixels in filter
    avg = p * q

    output_image = np.zeros((m, n), dtype='uint8')

    for k in range(m):
        for l in range(n):
            output_image[k, l] = np.sum(pad[k: k + p, l: l + q]) / avg

    return output_image


def show_image(title, image):
    cv2.imshow(str(title), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


path = 'cat.jpg'


def main(path):
    image = cv2.imread(path)
    transformed = spatial_average(image, (9, 9))
    show_image('Transformed image', transformed)


if __name__ == '__main__':
    main(path)
