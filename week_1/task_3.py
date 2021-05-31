import math
import numpy as np
import cv2


def rotate(image, angle):
    angle = math.radians(angle)  # converting degrees to radians
    cosine = math.cos(angle)
    sine = math.sin(angle)
    height = image.shape[0]  # define the height of the image
    width = image.shape[1]  # define the width of the image

    # define output image
    output = np.zeros((height, width, image.shape[2]), dtype='uint8')

    # Find the centre of the image about which we have to rotate the image
    centre_height = image.shape[0] // 2
    centre_width = image.shape[1] // 2

    for i in range(height):
        for j in range(width):
            # co-ordinates of pixel with respect to the centre of original image
            y = i - centre_height
            x = j - centre_width

            # co-ordinate of pixel with respect to the rotated image
            new_y = round(-x * sine + y * cosine)
            new_x = round(x * cosine + y * sine)

            #  Shift the coordinates after the rotation
            new_y = centre_height + new_y
            new_x = centre_width + new_x

            # adding if check to prevent any errors in the processing
            if 0 <= new_x < width and 0 <= new_y < height:
                output[new_y, new_x, :] = image[i, j, :]

    return output


def show_image(title, image):
    cv2.imshow(str(title), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


path = 'cat.jpg'
angle = -60


def main(path, angle):
    image = cv2.imread(path)
    transformed = rotate(image, angle)
    show_image('Rotated image', transformed)


if __name__ == '__main__':
    main(path, angle)
