from distutils import log
import sys
import cv2
import os
import numpy as np
import ImgIO
import ImgProcessing
import IPUtil
import copy

class Runner:

    ## Command
    # /Main.py process 000129.jpg
    # /Main.py detect 000123.jpg C  for generate type information detect img typetest
    # /Main.py draw 000129.jpg
    # /Main.py runsingle 000129.jpg
     
    def __init__(self) -> None:
        self.parameters = {
            "LoG_kernel_sizes": [5,7,9,11,21,31],
            "blur_before_LoG": 13,
            "blur_before_gradient": 7,
            "dilation_kernel": [[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
            "erosion_kernel": [[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0],[0,0,1,0,0],[0,0,1,0,0]],
            "otsu_blur_kernel_size": 5,
            "low_threshold": 0.86,
            "sample_radius": 4,
            "sample_offset": 3,
            "conn_threshold": 90,
            "min_max_radius_mutiplier": 3,
            "image_names": ["000012.jpg", "000037.jpg", "000049.jpg", "000129.jpg", "181-1.jpg"]
        }
        self.imgIO = ImgIO.ImgIO()
        self.ImgPro = ImgProcessing.ImgProcessing()
        self.IPU = IPUtil.IPUtil()
        self.log_kernel_sizes = self.parameters["LoG_kernel_sizes"]

    def img_preprocessing(self, name):
        blur_before = self.parameters["blur_before_LoG"]
        blur_after = self.parameters["blur_before_gradient"]
        d_kernel = self.parameters["dilation_kernel"]
        e_kernel = self.parameters["erosion_kernel"]
        otsu_blur_kernel_size = self.parameters["otsu_blur_kernel_size"]
        # define path
        source_path = os.path.join(os.getcwd(), 'source_img', name)
        print(source_path)
        # read in img as both original and gray
        img, g_img = self.imgIO.read_image(source_path)
        R = img.shape[0]
        C = img.shape[1]
        # conn_threshold = R * C * 0.0007
        # if conn_threshold < 90:
        #     conn_threshold = 90
        conn_threshold = self.parameters["conn_threshold"]
        # perform contrast enhencement
        contrast_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        topHat = cv2.morphologyEx(g_img, cv2.MORPH_TOPHAT, contrast_kernel)
        blackHat = cv2.morphologyEx(g_img, cv2.MORPH_BLACKHAT, contrast_kernel)
        c_img = g_img + topHat - blackHat

        log_imgs = self.ImgPro.bandpass_filtering(g_img, blur_before, self.log_kernel_sizes)
        # go through the log imgs
        for log_img, k in zip(log_imgs, self.log_kernel_sizes):
            write_path = os.path.join(os.getcwd(), 'results', name, 'log_' + str(k) + '_img/')
            print("Processing Laplacian kernel size: " + str(k))
            # write gray image and Laplacian of Gaussian result
            self.imgIO.write_image(write_path, "gray.jpg", g_img)
            self.imgIO.write_image(write_path, "contrast.jpg", c_img)
            self.imgIO.write_image(write_path, "1log.jpg", log_img)
            # produce the threshold result of log_img
            log_img_copy = copy.deepcopy(log_img)
            t_img, otsu_t = self.ImgPro.otsu_threshold(log_img_copy, otsu_blur_kernel_size)
            print("Otsu threshold value: ", otsu_t)
            self.imgIO.write_image(write_path, "2tlog.jpg", t_img)
            # use erosion first to isolated out the noise from the main
            e_img = self.ImgPro.erosion(t_img, e_kernel)
            self.imgIO.write_image(write_path, "3elog.jpg", e_img)
            # use connected component to eliminate small area
            conn_t_img = self.ImgPro.connected_component(e_img, conn_threshold)
            self.imgIO.write_image(write_path, "4ctlog.jpg", conn_t_img)
            # apply the dialation method
            d_img = self.ImgPro.dialation(conn_t_img, d_kernel)
            self.imgIO.write_image(write_path, "5dlog.jpg", d_img)
            # mask back to the log_img
            log_img = log_img.astype(np.uint8)
            mask_log_img = cv2.bitwise_and(log_img, d_img)
            self.imgIO.write_image(write_path, "6mlog.jpg", mask_log_img)
            # make a final blur
            d_img = d_img.astype(np.float32)
            mask_log_img = mask_log_img.astype(np.float32)
            b_d_img = cv2.GaussianBlur(d_img, (blur_after, blur_after), 0)
            b_mask_log_img = cv2.GaussianBlur(mask_log_img, (blur_after, blur_after), 0)
            self.imgIO.write_image(write_path, "7blog.jpg", b_d_img)
            self.imgIO.write_image(write_path, "8bmlog.jpg", b_mask_log_img)
            # calculate the gradient direction and save the gradient result
            mag, grad = self.ImgPro.extract_gradient_directions(b_d_img)
            self.imgIO.write_data(write_path, "mag", mag)
            m_mag, m_grad = self.ImgPro.extract_gradient_directions(b_mask_log_img)

            self.imgIO.write_image(write_path, "9mag.jpg", mag)
            self.imgIO.write_image(write_path, "10mmag.jpg", m_mag)

            # Save the grad data
            self.imgIO.write_data(write_path, "grad", grad)
            self.imgIO.write_data(write_path, "mgrad", m_grad)
            self.imgIO.write_image(write_path, "11grad.jpg", self.ImgPro.convert_gradient(grad))
            self.imgIO.write_image(write_path, "12mgrad.jpg", self.ImgPro.convert_gradient(m_grad))
            self.imgIO.write_image(write_path, "13ggrad.jpg", (grad/360)*255)

    def run_detect(self, img_name, command_type, r, off_set):
        for k in self.log_kernel_sizes:
            n = 9
            if command_type == "typetest":
                path = os.path.join(os.getcwd(), 'results')
                self.IPU.IP_detector(np.array([[]]), np.array([[]]), path, r, off_set, n, command_type)
            path = os.path.join(os.getcwd(), 'results', img_name, 'log_' + str(k) + '_img/')
            g_img = np.load(path + "grad.npy")
            m_img = np.load(path + "mag.npy")
            acc, t_result, acc_au, acc_ad, acc_bu, acc_bd = self.IPU.IP_detector(g_img, m_img, path, r, off_set, n, command_type)
            self.imgIO.write_data(path, "acc", acc)
            self.imgIO.write_data(path, "acc_au", acc_au)
            self.imgIO.write_data(path, "acc_ad", acc_ad)
            self.imgIO.write_data(path, "acc_bu", acc_bu)
            self.imgIO.write_data(path, "acc_bd", acc_bd)
            self.imgIO.write_data(path, "t_result", t_result)
            self.imgIO.save_curl_heat_map(acc_au, path, "au")
            self.imgIO.save_curl_heat_map(acc_ad, path, "ad")
            self.imgIO.save_curl_heat_map(acc_bu, path, "bu")
            self.imgIO.save_curl_heat_map(acc_bd, path, "bd")

    def filter_locations(self, acc, thresholds, r, min_max_r):
        locations = self.IPU.get_curl_locations(acc, thresholds, r, one=True)
        locations = self.IPU.non_min_max_suppression(locations, min_max_r, acc)

        return locations

    def run_draw(self, r, img_name):
        for k in self.log_kernel_sizes:
            min_max_r = r * self.parameters["min_max_radius_mutiplier"]
            path = os.path.join(os.getcwd(), 'results', img_name, 'log_' + str(k) + '_img/')
            src_path = os.path.join(os.getcwd(), 'source_img', img_name)
            img, g_img = self.imgIO.read_image(src_path)
            acc_au = np.load(path + "acc_au.npy")
            acc_bu = np.load(path + "acc_bu.npy")
            acc_ad = np.load(path + "acc_ad.npy")
            acc_bd = np.load(path + "acc_bd.npy")
            self.IPU.R = img.shape[0]
            self.IPU.C = img.shape[1]

            low_t = self.parameters["low_threshold"]

            # YELLOW AU
            print("AU yellow, ", end="")
            thresholds = [low_t]
            au_locations = self.filter_locations(acc_au, thresholds, r, min_max_r)

            # GREEN BU
            print("BU green, ", end="")
            thresholds = [low_t + 0.01]
            print(thresholds)
            bu_locations = self.filter_locations(acc_bu, thresholds, r, min_max_r)

            # RED AD
            print("AD red, ", end="")
            thresholds = [low_t + 0.01]
            ad_locations = self.filter_locations(acc_ad, thresholds, r, min_max_r)

            # BLUE BD
            print("BD blue, ", end="")
            thresholds = [low_t]
            bd_locations = self.filter_locations(acc_bd, thresholds, r, min_max_r)

            # supress too closed different types
            au_locations, bd_locations = self.IPU.type_supression(au_locations, bd_locations, r*2)
            bu_locations, ad_locations = self.IPU.type_supression(bu_locations, ad_locations, r*2)

            self.imgIO.draw_curl(au_locations, img, (0, 239, 255))
            self.imgIO.draw_curl(bu_locations, img, (0, 255, 0))
            self.imgIO.draw_curl(ad_locations, img, (0, 0, 255))
            self.imgIO.draw_curl(bd_locations, img, (255, 255, 0))

            self.imgIO.write_image(path, "detetcion_result.jpg", img)


    def run_single(self, img_name, command_type, r, off_set):
        self.img_preprocessing(img_name)
        self.run_detect(img_name, command_type, r, off_set)
        self.run_draw(r, img_name)

    def run_all(self, r, off_set):
        img_names = self.parameters["image_names"]
        for img_name in img_names:
            print("Run img: ", img_name)
            self.run_single(img_name, 'C', r, off_set)

    def run(self, argv):
        r = self.parameters["sample_radius"]
        off_set = self.parameters["sample_offset"]
        if argv[1] == "process":
            self.img_preprocessing(argv[2])
        elif argv[1] == "clean":
            print("Cleaning folders")
            self.imgIO.clean_folder(os.path.join(os.getcwd(), 'results'))
        elif argv[1] == "combine":
            print("Overlapping")
        elif argv[1] == "detect":
            self.run_detect(argv[2], argv[3], r, off_set)
        elif argv[1] == "draw":
            self.run_draw(r, argv[2])
        elif argv[1] == "runsingle":
            # default to type C
            self.run_single(argv[2], 'C', r, off_set)
        elif argv[1] == "runall":
            self.run_all(r, off_set)