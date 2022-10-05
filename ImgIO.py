import cv2
import os
import numpy as np
import shutil

class ImgIO:

    def read_image(self, img_path):
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # convert img to float
        img = img.astype(np.float32)
        gray_img = gray_img.astype(np.float32)

        return img, gray_img

    def write_image(self, path, name, src):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path + "/" + name, src)

    def write_data(self, path, name, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path + "/" + name, data)

    def clean_folder(self, dir):
        for files in os.listdir(dir):
            path = os.path.join(dir, files)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)

    # Pass in the acc array
    def save_curl_heat_map(self, acc, path, type):
        # Change acc values to 0 to 255
        acc *= 255
        # Apply color map for acc
        acc = acc.astype(np.uint8)
        cv2.imwrite(path + type + "-raw_curl_acc_map.jpg", acc)
        acc = cv2.applyColorMap(acc, cv2.COLORMAP_JET)

        cv2.imwrite(path + type + "-curl_acc_map.jpg", acc)

    def draw_curl(self, locations, original_img, color=None):
        for point in locations:
            if point is not None:
                cv2.circle(original_img, (point[1], point[0]), 3, color, 1)

        return original_img