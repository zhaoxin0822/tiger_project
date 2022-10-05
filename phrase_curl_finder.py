import sys
import cv2
import os
import numpy as np
import shutil
import math
import copy
import itertools

ddepth = cv2.CV_64F

def read_image(img_path, gray):
    img = cv2.imread(img_path)
    # convert img to float
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("gray_test.jpg", img)
    img = img.astype(np.float32)
    print(img.dtype)
    return img

def write_image(name, bfr, orient, mag):
    cv2.imwrite("bandpass_filter_results/" + name, bfr)
    cv2.imwrite("gradient_directions/" + name, orient)
    cv2.imwrite("gradient_magnitude/" + name, mag)

# Use the bandpass filter is essentially to supress the nosie in gradient extraction, low pass filter to get rid of the noise (high frequency filter)
# sigma_low: low bound for sigma value
# sigma_high: high bound for sigma value
# blur size: the Gaussian kernel to blur
def bandpass_filtering(img, sigma_low, sigma_high, thresh_hold, init_blur_size, blur_size, dof_lock):
    # Laplacian of Gaussian filter gradient direction and magnitude result
    for kernel_size in [5, 7, 9, 11, 13]:
        src = cv2.GaussianBlur(img, (init_blur_size, init_blur_size), 0)
        log_img = cv2.Laplacian(src, ddepth, ksize=kernel_size)
        
        # !!! Probably need to convert the negative value to 0. and above to 0 to 255, should I use the negative values to calculate the gradient direction or the postive value?
        # need ask DR, Farrell about it
        log_img += abs(log_img.min())
        log_img *= (255 / log_img.max())
        mag, orient = extract_gradient_directions(log_img)
        write_image(str(kernel_size) + "-log.jpg", log_img, orient, mag)

        log_img[log_img < thresh_hold] = 0

        log_img_blur = cv2.GaussianBlur(log_img, (blur_size, blur_size), 0)
        log_img_copy = copy.deepcopy(log_img)
        mag, orient = extract_gradient_directions(log_img)
        write_image(str(kernel_size) + "-slog.jpg", log_img_copy, orient, mag)
        mag, orient = extract_gradient_directions(log_img_blur, "gradient_directions/" + str(kernel_size) + "-blog.npy", True)
        write_image(str(kernel_size) + "-blog.jpg", log_img_blur, orient, mag)

    if dof_lock:
        print(img.dtype)
        # Differnece of Gaussian with different sigma value (normal distribution)
        for sigma in np.arange(sigma_low, sigma_high, 0.2):
            print("sigma: ", sigma)
            # Calculate the DoG by subtracting
            for l_k, h_k in [(5, 5)]:
                for m in [1.5, 2, 3, 4, 5, 6, 7, 8, 9]:
                    low_sigma = cv2.GaussianBlur(img, (l_k,l_k), sigma)
                    high_sigma = cv2.GaussianBlur(img, (h_k, h_k), sigma * m)
                    np.set_printoptions(threshold=sys.maxsize)
                    dog_img = low_sigma - high_sigma
                    # dog_img[dog_img <= 240] = 0
                    # smooth before extract gradient
                    # dog_img = cv2.GaussianBlur(dog_img, (blur_size, blur_size), 0)
                    # dog_img += abs(dog_img.min())
                    # dog_img *= (255 / dog_img.max())
                    # print(dog_img.min(), dog_img.max())
                    mag, orient = extract_gradient_directions(dog_img)

                    name  = str(l_k) + "-" + str(h_k) + "-" + str(sigma)[0:3] + "-" + str(sigma * m)[0:3] + "-dog.jpg"
                    write_image(name, dog_img, orient, mag)

# Calculate gradient direction
def extract_gradient_directions(img, path_name=None, save=False):
    # Compute gradients along the x and y axis, respectively
    gX = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    gY = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    # gX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 7)
    # gY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 7)
    # Compute the gradient magnitude and orientation
    mag = np.sqrt((gX ** 2) + (gY ** 2))
    # Convert to degree
    # Arctan2 return -pi to pi, now change to 0 to 180, and 180 to 360 when degree is negative, means -50 is 310 or -1 is 359
    orient = np.arctan2(gY, gX) * (180 / np.pi)
    orient[orient < 0] += 360
    # save gradient direction result
    if save:
        np.save(path_name, orient)
    # change range from 0 to 360 to 0 to 255
    orient = orient / 360 * 255
    # apply color map for gradient
    orient = orient.astype(np.uint8)
    orient = cv2.applyColorMap(orient, cv2.COLORMAP_HSV)

    return mag, orient


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def weighted_add():
    img1 = read_image(os.path.join(os.getcwd(), 'image_preprocessing', 'test_crop_small.jpg'), False)
    img2 = read_image(os.path.join(os.getcwd(), 'gradient_directions', '7-blog.jpg'), False)
    dst = cv2.addWeighted(img1, 1, img2, 0.25, 0)
    cv2.imwrite("overlap.jpg", dst)

# all points for a circle with radius specified
def get_neighborhood_of_r(radius):
    switch = 3 - (2 * radius)
    points = set()
    x = 0
    y = radius
    while x <= y:
        points.add((x,-y))
        points.add((y,-x))
        points.add((y,x))
        points.add((x,y))
        points.add((-x,y))        
        points.add((-y,x))
        points.add((-y,-x))
        points.add((-x,-y))
        if switch < 0:
            switch = switch + (4 * x) + 6
        else:
            switch = switch + (4 * (x - y)) + 10
            y = y - 1
        x = x + 1

    # Adding these four points so that bresenham can grap all the points around it
    if radius == 1:
        points.add((1, 1))
        points.add((-1, -1))
        points.add((-1, 1))
        points.add((1, -1))

    return points

def calculate_bin_vals(neighbors, n, r, c, img, R, C):
    bins = {}
    count = 0
    for i in range(1, n+1):
        bins[i] = 0

    for n_g in neighbors:
        # make the center translation and grab the gradient in radius
        n_r = n_g[0] + r
        n_c = n_g[1] + c 
        if (n_r < 0 or n_r >= R) or (n_c < 0 or n_c >= C):
            continue
        n_g = img[n_r, n_c] * (math.pi / 180)
        # print(img[n_r, n_c], n_g)
        # Calculate which bin does the gradient belong
        val = int(1 + math.floor((n * n_g) / (2 * math.pi)))
        if val <= n:
            bins[val] += 1
            count += 1

    mean_bin = count / n
    return bins, mean_bin, count

def calc_bin_residual_from_mean(bins, mean_bin):
    bin_residual_sum = 0
    for bin_val in bins.values():
        bin_residual = (bin_val - mean_bin) ** 2
        bin_residual_sum += bin_residual
    return math.sqrt(bin_residual_sum)

# Counting phase directions in disc-shaped neighborhoods of radii 1 to lambda_x/2
# img is the source image
# x, y is the location
# slow function
def calc_a_x(img, r, c, m, n, circles, R, C):
    M = 2 / (m**2 + m)
    l2_dists = {}
    # go through each histogram
    for j in range(1, m+1):
        ns = circles[j]
        j_bins, j_mean_bin, j_bin_count = calculate_bin_vals(ns, n, r, c, img, R, C)
        
        N = j_bin_count * math.sqrt((n-1)/(n))
        importance_weight = (1 + m - j) / N
        j_dist = calc_bin_residual_from_mean(j_bins, j_mean_bin)
        l2_dists[j] = importance_weight * j_dist
    
    a_x = 1 - M * sum(l2_dists.values())
    return a_x 

# lambda_x is the diameter of the smaple circle
# n is the number of bins, by default this value is 8
def curl_detection(img, lambda_x, n=8):
    m = int(lambda_x/2) # m histograms
    r = img.shape[0] # row count
    c = img.shape[1] # column count
    acc = np.zeros(img.shape)
    # pregenerate all the circles
    circles = {}
    for j in range(1, m+1):
        circles[j] = get_neighborhood_of_r(j)
    # go through the image pixels
    for r_i in range(r):
        for c_i in range(c):
            a_x = calc_a_x(img, r_i, c_i, m, n, circles, r, c)
            acc[r_i, c_i] = a_x
    return acc

def process(input_list, threshold=(10,10)):
    combos = itertools.combinations(input_list, 2)
    points_to_remove = [point2 for point1, point2 in combos if abs(point1[0]-point2[0])<=threshold[0] and abs(point1[1]-point2[1])<=threshold[1]]
    points_to_keep = [point for point in input_list if point not in points_to_remove]
    return points_to_keep

def get_curl_locations(acc, curl_threshold, path_name):

    locationsOne = np.vstack(np.where(acc >= curl_threshold)).T
    locationsTwo = []
    # locationsTwo = np.vstack(np.where((acc >= 0.8) & (acc < 0.9))).T
    locationsThree = []
    # locationsThree = np.vstack(np.where((acc >= 0.61) & (acc < 0.615))).T
    # locationsFour = []
    locationsFour = np.vstack(np.where((acc >= 0.6) & (acc < 0.63))).T

    print("location one: ", len(locationsOne))
    print("location two: ", len(locationsTwo))
    print("location three: ", len(locationsThree))
    print("location Four: ", len(locationsFour))

    np.save(path_name + "-locationsOne.npy", locationsOne)
    np.save(path_name + "-locationsTwo.npy", locationsTwo)
    np.save(path_name + "-locationsThree.npy", locationsThree)
    np.save(path_name + "-locationsFour.npy", locationsFour)

    return [locationsOne, locationsTwo, locationsThree, locationsFour]

def save_curl_heat_map(acc, path):
    # amp acc to 0 to 255 to see the color map
    acc *= 255
    # apply color map for acc
    acc = acc.astype(np.uint8)
    acc = cv2.applyColorMap(acc, cv2.COLORMAP_JET)

    cv2.imwrite(path + "-curl_acc_map.jpg", acc)

def draw_curl(lcoations, path_name):
    absolution_path = os.path.join(os.getcwd(), 'image_preprocessing', 'test_crop_small.jpg')
    img = read_image(absolution_path, False)

    for i, location in enumerate(lcoations):
        for point in location:
            color = None
            if i == 0:
                color = (0, 0, 0)
            if i == 1:
                color = (255, 0, 0)
            if i == 2:
                color = (0, 255, 0)
            if i == 3:
                color = (0, 0, 255)

            cv2.circle(img, (point[1], point[0]), 3, color, 1)

    cv2.imwrite(path_name + "-detetcion_result.jpg", img)    


def main():
    absolution_path = os.path.join(os.getcwd(), 'image_preprocessing', 'test_crop_small.jpg')
    img = read_image(absolution_path, True)
    # mag, orient = extract_gradient_directions(blur_img)
    # write_image("origin.jpg", blur_img, orient, mag)
    
    # sigma value low to high, threshold, blur size, blur size after band pass filter
    bandpass_filtering(img, 0.5, 5, 162, 13, 7, False)

def print_matrix(matrix):
    for r in matrix:
        print(r)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == "clean":
        print("Clean folders")
        clean_folder(os.path.join(os.getcwd(), 'bandpass_filter_results'))
        clean_folder(os.path.join(os.getcwd(), 'gradient_directions'))
        clean_folder(os.path.join(os.getcwd(), 'gradient_magnitude'))
    elif sys.argv[1] == "combine":
        weighted_add()
    elif sys.argv[1] == "detect":
        # test = np.array(range(1, 101)).reshape((10, 10))
        # load gradient direction data
        for name in ["5-blog", "7-blog", "9-blog", "11-blog", "13-blog"]:
            print("processing " + name)
            if not os.path.exists("curl_detection_results/" + name):
                os.mkdir("curl_detection_results/" + name)
            g_img = np.load("gradient_directions/" + name + ".npy")
            # print(g_img)
            curl_threshold = 0.9
            n = 8
            for lambda_x in [8, 10, 12]:
                result_path = "curl_detection_results/" + name + "/" + str(lambda_x) + "-" + str(n)
                print("processing: ", lambda_x)
                acc = curl_detection(g_img, lambda_x, n)
                np.save(result_path + "-acc.npy", acc)
                locations = get_curl_locations(acc, curl_threshold, result_path)
                draw_curl(locations, result_path)
                save_curl_heat_map(acc, result_path)
