from mmap import ACCESS_READ
import numpy as np
import math
from CircleSampler import CircleSampler
from ImgIO import ImgIO
from ImgProcessing import ImgProcessing
from scipy.optimize import linear_sum_assignment
import copy
import os


class IPUtil:
    def __init__(self):
        self.R = 0
        self.C = 0
        self.m = 0
        self.n = 0
        self.total = 0
        self.ImgIO = ImgIO()
        self.ImgP = ImgProcessing()
        self.cirSample = CircleSampler()

    # Method B Implementation
    # Given all the histogram and its corresponding gradient, calculate a cost matrix
    # If we use the shape context method, we will inherite the roatation invariance attribute of it, thus we will only need A or B type
    # n is the bin number in each histogram row
    def calculate_cost_matrix(self, gradients, n):
        # print(len(gradients), n)
        # Defin empty costmatrix
        c_m = np.zeros((len(gradients), n + 1)) # add an extra bin for 360 degree
        # Each circle (a histogram) has its corresponding gradient values
        step = 360 / n
        for i, gradient in enumerate(gradients):
            for g in gradient:
                bin_i = math.floor(g/step)
                # If a gradient falls in a bin just increment it
                c_m[i, bin_i] += 1 
        # normalize the matrix
        # c_m = c_m / c_m.max()
        return c_m

        # Calculate the a_x value for each pixel, there are two a_x values corresponding to two types
    def calc_cost(self, n, circles, sample_cms, r, c, src):

        gradients = []
        # Get the cost matrix for this pixel
        for circle in circles:
            # location translation
            circle = np.add(circle, np.array([r, c]))
            # filter out locations that are out of bound
            circle = circle[circle[:,0] >= 0]
            circle = circle[circle[:,0] < self.R]
            circle = circle[circle[:,1] >= 0]
            circle = circle[circle[:,1] < self.C]
            # get all the gradient values from cordinates, need to checkout bounding
            row_grads = []
            for loc in circle:
                row_grads.append(src[loc[0], loc[1]])
            # print(row_grads)
            gradients.append(row_grads)

        cm = self.calculate_cost_matrix(gradients, n)
        
        A_val = self.cost_matrix_similar_score(cm, sample_cms[0])
        B_val = self.cost_matrix_similar_score(cm, sample_cms[1])

        if A_val > B_val:
            return A_val, 0
        else:
            return B_val, 1

    # cmOne is each pixels cost matrix, cmTwo is the smaples
    def cost_matrix_similar_score(self, cmOne, cmTwo):
        # for each histogram (row)
        sum_cost = 0
        for r in range(self.m):
            r_one = cmOne[r]
            r_two = cmTwo[r]
            # the cost value of each row range from 0 - 1, 0 means no difference, qhi square test
            r_diff =  r_one - r_two
            # print(r_diff.max(), r_diff.min())
            r_sqr = np.square(r_diff)
            r_sum = np.add(r_one, r_two)
            r_division = np.divide(r_sqr, r_sum, out=np.zeros_like(r_sqr), where=r_sum!=0)
            # r_sum_sqrt = math.sqrt(sum(r_sqr))
            # this value will range between 0 and 1, biggest difference is like 00000000 and 11111111
            c = sum(r_division) / self.n
            # importance weights calculation, the first histogram have more weight
            # w_c = (1 + self.m - (r + 1)) / (sum(r_one) * math.sqrt(((self.n-1)/self.n)))
            w_c = (1 + self.m - (r + 1)) / self.total
            sum_cost += w_c * c

        # balancer = 2 / (self.m**2 + self.m)
        # print(sum_cost)
        result = 1 - sum_cost
        # print(result)

        return result

    # Method A Implementation
    # Calculate a min cost matrix
    def build_gradient_cost_matrix(self, circles, sample_grads, r, c, src, mag, avg):
        # check magnitude, if gradient magnitude is 0 we just skip it
        points = {}
        l = 0
        for circle, grad in zip(circles, sample_grads):
            for cir, s_g in zip(circle, grad):
                # convert cordinates, and make translation
                row = -cir[1] + r
                col = cir[0] + c
                g = None
                # make a check, the out of bound are should still take into consideration, the out pixel should just be 180, a middle gradient which has small impact
                # maybe when it takes value from mag under threshold make a punbishment as well
                if row < 0 or row >= self.R:
                    g = avg
                if col < 0 or col >= self.C:
                    g = avg
                if g is None:
                    # if mag[row, col] <= 122:
                    #     g = 180
                    # else:
                    g = src[row, col]
                points[(row, col)] = [s_g, g]
                l += 1

        # can be changed in a way that the shape does nto have to be a square
        cost_matrix = np.zeros((l, l))
        for i, p_r in enumerate(points.keys()):
            for j, p_c in enumerate(points.keys()):
                s_g = points[p_r][0]
                g = points[p_c][1]
                diff = abs(s_g - g)
                if diff > 180:
                    diff = 360 - diff
                cost_matrix[i, j] = diff

        # nomalized the difference since the max difference is 360
        cost_matrix = cost_matrix / 180
        # calculate the min cost path and add the costs together, in ideal situation the cost is 0 (Hungarian method)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        result = cost_matrix[row_ind, col_ind].sum()
        return result

    # Method C, the order is AU AD BU BD
    def calc_direct_mask_score(self, src, circles, samples, r, c):
        totals = [0, 0, 0, 0]
        # ignore the edge where it can't be taken as a full sample, for not taking as a full sample may increase the score of the points around edge
        # sum all the gradient diff score (remember the diff for each spot can't bigger than 180)
        for t in range(4):
            circle = circles[t]
            sample = samples[t]
            for h_circle, gs in zip(circle, sample):
                for cir, g in zip(h_circle, gs):
                    row = -cir[1] + r
                    col = cir[0] + c

                    diff = abs(src[row, col] - g)
                    if diff > 180:
                        diff = 360 - 180
                    totals[t] += diff

        # for each point concolve with the four defined samples around this point
        # will get a large value in the end but it is okay as large as 45720 when the sample radius is 9, the value range from 0 to 45720

        # return four diff sum values out
        return totals

    def edge_acc_filter(self, acc, edge_rows, edge_cols, ignored_rows, ignored_cols):
        acc_max = acc.max()
        acc[ignored_rows, ignored_cols] = acc_max
        acc[edge_rows, edge_cols] = acc_max
        acc = acc / acc_max
        acc = np.negative(acc)
        acc += 1

        return acc

    # Main method to return the acc array, r means the radius or how many historgrams, n means how mmany bins in a histogram
    def IP_detector(self, src, mag, path, r=4, off_set=0, n=8, method="A"):
        # Define dimensions 
        print("Src img shape: ", src.shape, r, off_set)
        self.R = src.shape[0]
        self.C = src.shape[1]
        acc = np.zeros(src.shape)
        acc_au = np.zeros(src.shape)
        acc_ad = np.zeros(src.shape)
        acc_bu = np.zeros(src.shape)
        acc_bd = np.zeros(src.shape)
        t_reults = np.zeros(src.shape)
        self.m = r
        self.n = n
        self.total = sum(range(1, self.m + 1))

        # Pregenerate all the circles, and each circle has two corresponding ideal samples (cost matrixs)
        circles = []
        AU_samples = []
        AD_samples = []
        BU_samples = []
        BD_samples = []
        AU_sum = 0
        AD_sum = 0
        BU_sum = 0
        BD_sum = 0
        count = 0
        for j in range(0, r+1):
            circle = self.cirSample.get_circles_of_r(j)
            # need to add a cut off here
            count += len(circle)
            samples, avgs = self.cirSample.generate_type_smaples(circle)
            circles.append(circle)
            AU_sum += avgs[0]
            AU_samples.append(samples[0])
            AD_sum += avgs[1]
            AD_samples.append(samples[1])
            BU_sum += avgs[2]
            BU_samples.append(samples[2])
            BD_sum += avgs[3]
            BD_samples.append(samples[3])

        circles, AU_samples, AD_samples, BU_samples, BD_samples = self.cirSample.circle_cut(circles, r, off_set, AU_samples, AD_samples, BU_samples, BD_samples)
        print("Finish cutting")

        if method == "A":
            for r_i in range(self.R):
                print("finshied row: ", r_i)
                for c_i in range(self.C):
                    if mag[r_i, c_i] <= 122:
                        continue
                    c_AU  = self.build_gradient_cost_matrix(circles, AU_samples, r_i, c_i, src, mag, AU_sum/count)
                    c_BU  = self.build_gradient_cost_matrix(circles, BU_samples, r_i, c_i, src, mag, BU_sum/count)
                    acc_au[r_i, c_i] = c_AU
                    acc_bu[r_i, c_i] = c_BU

            ignored_points = np.argwhere(mag <= 122)
            ignored_rows, ignored_cols = zip(*ignored_points)

            acc_au = acc_au / acc_au.max()
            acc_au = np.negative(acc_au)
            acc_au += 1 # flip the result, 1 menas a perfect score
            acc_au[ignored_rows, ignored_cols] = 0

            acc_bu = acc_bu / acc_bu.max()
            acc_bu = np.negative(acc_bu)
            acc_bu += 1 # flip the result, 1 menas a perfect score
            acc_bu[ignored_rows, ignored_cols] = 0

        elif method == "B":
            ideal_AU_cm = self.calculate_cost_matrix(AU_samples, n)
            ideal_BU_cm = self.calculate_cost_matrix(BU_samples, n)

            # Go through the image pixel by pixel
            for r_i in range(self.R):
                for c_i in range(self.C):
                    a_x, a_t = self.calc_cost(n, circles, [ideal_AU_cm, ideal_BU_cm], r_i, c_i, src)
                    acc[r_i, c_i] = a_x
                    t_reults[r_i, c_i] = a_t

        elif method == "C":
            edge_rows = []
            edge_cols = []
            for r_i in range(self.R):
                for c_i in range(self.C):

                    if mag[r_i, c_i] <= 122:
                        continue
                    if (c_i <= r or c_i >= (self.C - r)) or (r_i <= r or r_i >= (self.R - r)):
                        edge_rows.append(r_i)
                        edge_cols.append(c_i)
                        continue

                    scores = self.calc_direct_mask_score(src, circles, [AU_samples, AD_samples, BU_samples, BD_samples], r_i, c_i)
                    acc_au[r_i, c_i] = scores[0]
                    acc_ad[r_i, c_i] = scores[1]
                    acc_bu[r_i, c_i] = scores[2]
                    acc_bd[r_i, c_i] = scores[3]

            ignored_points = np.argwhere(mag <= 122)
            ignored_rows, ignored_cols = zip(*ignored_points)

            acc_au = self.edge_acc_filter(acc_au, edge_rows, edge_cols, ignored_rows, ignored_cols)
            acc_ad = self.edge_acc_filter(acc_ad, edge_rows, edge_cols, ignored_rows, ignored_cols)
            acc_bu = self.edge_acc_filter(acc_bu, edge_rows, edge_cols, ignored_rows, ignored_cols)
            acc_bd = self.edge_acc_filter(acc_bd, edge_rows, edge_cols, ignored_rows, ignored_cols)

        elif method == "typetest":
            print("Generating")
            self.cirSample.sample_test(circles[0], AU_samples, path, "AU_type.jpg")
            self.cirSample.sample_test(circles[1], AD_samples, path, "AD_type.jpg")
            self.cirSample.sample_test(circles[2], BU_samples, path, "BU_type.jpg")
            self.cirSample.sample_test(circles[3], BD_samples, path, "BD_type.jpg")
            print("Type smaple generation finished")
            exit()
        else:
            exit()

        return acc, t_reults, acc_au, acc_ad, acc_bu, acc_bd

    ### Code below deal with find the IP location
    def filter_out_of_bound(self, circle, r, c, radius):
        if len(circle) == 0:
            return []
        circle = np.array(circle)
        circle = np.add(circle, np.array([r, c]))
        circle = circle[circle[:,0] >= radius]
        circle = circle[circle[:,1] >= radius]
        circle = circle[circle[:,0] < self.R - radius]
        circle = circle[circle[:,1] < self.C - radius]
        circle = list(circle)
        return circle

    def get_curl_locations(self, acc, curl_thresholds, r, one=False, two=False, three=False, four=False):

        locationsOne = []
        locationsTwo = []
        locationsThree = []
        locationsFour = []

        if one:
            locationsOne = np.vstack(np.where(acc >= curl_thresholds[0])).T
        if two:
            locationsTwo = np.vstack(np.where((acc >= curl_thresholds[1]) & (acc < curl_thresholds[2]))).T
        if three:
            locationsThree = np.vstack(np.where((acc >= curl_thresholds[3]) & (acc < curl_thresholds[4]))).T
        if four:
            locationsFour = np.vstack(np.where((acc >= curl_thresholds[5]) & (acc < curl_thresholds[6]))).T


        locationsOne = self.filter_out_of_bound(locationsOne, 0, 0, int(r/2))
        # locationsTwo = self.filter_out_of_bound(locationsTwo, 0, 0, int(r/2))
        # locationsThree = self.filter_out_of_bound(locationsThree, 0, 0, int(r/2))
        # locationsFour = self.filter_out_of_bound(locationsFour, 0, 0, int(r/2))

        combine = [locationsOne, locationsTwo, locationsThree, locationsFour]

        # print("location one: ", len(locationsOne))
        # print("location two: ", len(locationsTwo))
        # print("location three: ", len(locationsThree))
        # print("location Four: ", len(locationsFour))

        print("Find: ", len(locationsOne))

        return combine

    def non_min_max_suppression(self, locations, r, acc):
        points = sum(locations, [])
        result = copy.deepcopy(points)
        for i in range(len(points)):
            for j in range(len(points)):
                p = points[i]
                pc = points[j]
                if p[0] == pc[0] and p[1] == pc[1]:
                    continue
                else:
                    # if too close
                    dist = math.sqrt((p[0]-pc[0])**2 + (p[1]-pc[1])**2)
                    if dist < r:
                        if acc[p[0], p[1]] >= acc[pc[0], pc[1]]:
                            result[j] = None
                        else:
                            result[i] = None
        return result

    # This is used for supress the case where there exist two points red and green or blue and yellow are too close to each other
    # When these types are too close means there is a dot or a small hole which we do not want to have
    def type_supression(self, locationsOne, lcoationsTwo, r):
        resultOne = copy.deepcopy(locationsOne)
        resultTwo = copy.deepcopy(lcoationsTwo)
        for i in range(len(locationsOne)):
            for j in range(len(lcoationsTwo)):
                p = locationsOne[i]
                pc = lcoationsTwo[j]
                if p is None or pc is None:
                    continue
                dist = math.sqrt((p[0]-pc[0])**2 + (p[1]-pc[1])**2)
                if dist < r:
                    # Delete both points
                    resultOne[i] = None
                    resultTwo[j] = None

        return resultOne, resultTwo