import cv2
import numpy as np
from ImgIO import ImgIO

class ImgProcessing:

    def __init__(self):
        self.imgIO = ImgIO()
        self.ddepth = cv2.CV_64F

    # Pass image to bandpass filter, and produce LoG result that is ranged from 0 to 255
    def bandpass_filtering(self, img, blur_size, kernel_sizes):
        results = []
        for kernel_size in kernel_sizes:
            src = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
            log_img = cv2.Laplacian(src, self.ddepth, ksize=kernel_size)
            
            log_img += abs(log_img.min())
            log_img *= (255 / log_img.max())
            results.append(log_img)

        return results

    # Return magnitude and greadient direction result (dradient direction range from 0 to 360 degree)
    def extract_gradient_directions(self, img):
        # Compute gradients along the x and y axis, respectively
        gX = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        gY = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        # Compute the gradient magnitude and orientation
        mag = np.sqrt((gX ** 2) + (gY ** 2))
        # Convert to degree
        # Arctan2 return -pi to pi, now change to 0 to 180, and 180 to 360 when degree is negative, means -50 is 310 or -1 is 359
        orient = np.arctan2(gY, gX) * (180 / np.pi)
        orient[orient < 0] += 360
        return mag, orient

    def convert_gradient(self, grad):
        # change range from 0 to 360 to 0 to 255
        grad = grad / 360 * 255
        # apply color map for gradient
        grad = grad.astype(np.uint8)
        grad = cv2.applyColorMap(grad, cv2.COLORMAP_HSV)
        return grad

    def weighted_add(self, pathOne, pathTwo, path):
        img1, g_img1 = self.imgIO.read_image(pathOne)
        img2, g_img2 =  self.imgIO.read_image(pathTwo)
        dst = cv2.addWeighted(img1, 1, img2, 0.25, 0)
        self.imgIO.write_image(path, "overlapped.jpg", dst)

    # Apply dialation effect to img to reduce noise, pass in binarilized img
    def dialation(self, img, d_kernel):
        d_kernel = np.array(d_kernel)
        d_kernel = d_kernel.astype(np.uint8)
        img_dilation = cv2.dilate(img, d_kernel, iterations=1)
        return img_dilation

    def erosion(self, img, e_kernel):
        e_kernel = np.array(e_kernel)
        e_kernel = e_kernel.astype(np.uint8)
        img_erosion = cv2.erode(img, e_kernel, iterations=1)
        return img_erosion

    # Use connected component method to count up the connected area, eliminate area with smaller count than threshold
    def connected_component(self, img, threshold):
        print("Delete conponent by: ", threshold)
        analysis = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S) 
        (totalLabels, label_ids, values, centroid) = analysis

        output = np.zeros(img.shape, dtype="uint8")

        # Loop through each component
        for i in range(1, totalLabels):
            area = values[i, cv2.CC_STAT_AREA] 
            if (area > threshold): # only keep the component that greater than a certain value
                componentMask = (label_ids == i).astype("uint8") * 255
                output = cv2.bitwise_or(output, componentMask)

        return output

    # To counter the lighting effect on the img, apply the Otsu's method to generate a threshold value
    # also generate a thresholded mask
    def otsu_threshold(self, img, k):
        img = img.astype(np.uint8)
        blurred = cv2.GaussianBlur(img, (k, k), 0)
        threshold, t_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return t_img, threshold
