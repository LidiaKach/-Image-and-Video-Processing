import numpy as np
import cv2

title = {0: "(-1, 0)",
         1: "(0, 1)",
         2: "(-1, 0), (-1, 1), (0, 1)"}


# Calculate information entropy
def entropy(labels):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log2(norm_counts)).sum()


def prediction_errors(image, flag):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    if flag == 0:  # (-1, 0)
        diff = cv2.absdiff(img[2:h, :], img[1:h - 1, :])

    elif flag == 1:  # (0, 1)
        diff = cv2.absdiff(img[:, 2: w], img[:, 1: w - 1])
    else:  # (-1, 0), (-1, 1), (0, 1)
        diff = np.zeros((h, w))
        for i in range(2, h):
            for j in range(1, w - 1):
                diff[i, j] = (abs(img[i, j] - img[i - 1, j]) + abs(img[i, j] - img[i - 1, j]) + abs(
                    img[i, j] - img[i - 1, j + 1])) / 3
    return diff


def show_image(title, image):
    cv2.imshow(str(title), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    error_entropy = []
    image = cv2.imread("alice.jpg")
    for flag in (0, 1, 2):
        error = prediction_errors(image, flag)
        show_image(title[flag], error)
        error_entropy.append(entropy(error))

    print(error_entropy)
