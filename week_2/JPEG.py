import cv2
import numpy as np


class Jpeg:
    def __init__(self):
        # Initialize the A matrix of the DCT transformation
        self.__dctA = np.zeros(shape=(8, 8))
        for i in range(8):
            c = 0
            if i == 0:
                c = np.sqrt(1 / 8)
            else:
                c = np.sqrt(2 / 8)
            for j in range(8):
                self.__dctA[i, j] = c * np.cos(np.pi * i * (2 * j + 1) / (2 * 8))
        # Luminance quantization matrix
        self.__lq = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                              [12, 12, 14, 19, 26, 58, 60, 55],
                              [14, 13, 16, 24, 40, 57, 69, 56],
                              [14, 17, 22, 29, 51, 87, 80, 62],
                              [18, 22, 37, 56, 68, 109, 103, 77],
                              [24, 35, 55, 64, 81, 104, 113, 92],
                              [49, 64, 78, 87, 103, 121, 120, 101],
                              [72, 92, 95, 98, 112, 100, 103, 99]])
        # Color quantization matrix
        self.__cq = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                              [18, 21, 26, 66, 99, 99, 99, 99],
                              [24, 26, 56, 99, 99, 99, 99, 99],
                              [47, 66, 99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99, 99, 99]])
        # Mark the matrix type, lt is the brightness matrix, ct is the chroma matrix
        self.__lt = 0
        self.__ct = 1


    def __rgb2yuv(self, r, g, b):
        # Get YUV matrix from image
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
        v = 0.5 * r - 0.419 * g - 0.081 * b + 128
        return y, u, v

    def __dct(self, block):
        # DCT transformation
        res = np.dot(self.__dctA, block)
        res = np.dot(res, np.transpose(self.__dctA))
        return res

    def __quantize(self, block, tag):
        res = block
        if tag == self.__lt:
            res = np.round(res / self.__lq)
        elif tag == self.__ct:
            res = np.round(res / self.__cq)
        return res

    def __pad(self, matrix):

        fh, fw = 0, 0
        if self.height % 8 != 0:
            fh = 8 - self.height % 8
        if self.width % 8 != 0:
            fw = 8 - self.width % 8
        res = np.pad(matrix, ((0, fh), (0, fw)), 'constant',
                     constant_values=(0, 0))
        return res

    def __encode(self, matrix, tag):
        # Pad the matrix first
        matrix = self.__pad(matrix)
        # Cut the image matrix into 8*8 pieces
        height, width = matrix.shape
        res = matrix
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                dct = self.__dct(matrix[i: i + 8, j: j + 8])
                res[i: i + 8, j: j + 8] = self.__quantize(dct, tag)
        return res

    def compress(self, filename):
        # Read the picture according to the path image_path and store it as an RGB matrix
        image = cv2.imread(filename)
        # Get image width and height
        self.height, self.width, channels = image.shape
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        # Convert image RGB to YUV
        y, u, v = self.__rgb2yuv(r, g, b)
        # Downsample:
        #u = 2 * np.round(u / 2)
        #v = 2 * np.round(v / 2)
        # Encode the image matrix
        y_blocks = self.__encode(y - 128, self.__lt)
        u_blocks = self.__encode(u - 128, self.__ct)
        v_blocks = self.__encode(v - 128, self.__ct)
        return np.dstack((y_blocks, u_blocks, v_blocks))

    def __idct(self, block):
        # IDCT transformation
        res = np.dot(np.transpose(self.__dctA), block)
        res = np.dot(res, self.__dctA)
        return res

    def __iquantize(self, block, tag):
        res = block
        if tag == self.__lt:
            res *= self.__lq
        elif tag == self.__ct:
            res *= self.__cq
        return res

    def __decode(self, matrix, tag):
        height, width = matrix.shape
        res = matrix
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                res[i: i + 8, j: j + 8] = self.__idct(self.__iquantize(matrix[i: i + 8, j: j + 8], tag))
        return res

    def decompress(self, matrix):
        y_blocks = matrix[:, :, 0]
        u_blocks = matrix[:, :, 1]
        v_blocks = matrix[:, :, 2]
        y = self.__decode(y_blocks, self.__lt) + 128
        u = self.__decode(u_blocks, self.__ct) + 128
        v = self.__decode(v_blocks, self.__ct) + 128
        r = (y + 1.402 * (v - 128))
        g = (y - 0.34414 * (u - 128) - 0.71414 * (v - 128))
        b = (y + 1.772 * (u - 128))
        return np.dstack((r, g, b)).astype('uint8')



if __name__ == '__main__':
    kjpeg = Jpeg()
    comp = kjpeg.compress("sample.bmp")
    img = kjpeg.decompress(comp)
    cv2.imwrite("sample.jpg", img)
    cv2.imshow("compressed", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

