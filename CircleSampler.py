import numpy as np
import math

from ImgIO import ImgIO
from ImgProcessing import ImgProcessing

class CircleSampler:

    def __init__(self):
        self.ImgIO = ImgIO()
        self.ImgP = ImgProcessing()
        self.circle_enhence_points = {4:[(-2,3),(-3,2),(2,3),(3,2),(-3,-2),(-2,-3),(2,-3),(3,-2)], 
                                      6:[(-4,4),(4, 4),(-4,-4),(4,-4)], 
                                      9:[(-3,8),(3,8),(-3,-8),(3,-8),(-5,7),(5,7),(-5,-7),(5,-7),(-6,6),(6,6),(-6,-6),(6,-6),(-7,5),(-7,-5),(7,5),(7,-5)]}

    # Sample points as disc like circles via Bresenham method 
    def mirror_points_8(self, x, y):
        return [( x,  y),
                ( y,  x),
                (-x,  y),
                (-y,  x),
                ( x, -y),
                ( y, -x),
                (-x, -y),
                (-y, -x)]
                
    def get_circles_of_r_naive(self, r):
        points = []
        for x in range(r + 1):
            y = math.isqrt((r * r) - (x * x))
            if x > y:
                break
            points.extend(self.mirror_points_8(x, y))

        points = np.array(list(set(points)))
        return points


    def get_circles_of_r(self, r):
        points = []
        x = 0
        y = -r
        F_M = 1 - r
        d_e = 3
        d_ne = -(r << 1) + 5
        points.extend(self.mirror_points_8(x, y))
        while x < -y:
            if F_M <= 0:
                F_M += d_e
            else:
                F_M += d_ne
                d_ne += 2
                y += 1
            d_e += 2
            d_ne += 2
            x += 1
            points.extend(self.mirror_points_8(x, y))

        if r >= 4:
            for p in self.circle_enhence_points[4]:
                points.append(p)
        if r >= 6:
            for p in self.circle_enhence_points[6]:
                points.append(p)
        if r >= 9:
            for p in self.circle_enhence_points[9]:
                points.append(p)

        points = np.array(list(set(points)))

        return points

    def generate_type_smaples(self, circle):
        # define the four types
        A_U = []
        A_D = []
        B_U = []
        B_D = []

        for i, p in enumerate(circle):
            # calculate the degree for each point to its center
            d = math.atan2(p[1],p[0]) / math.pi*180

            if p[1] > 0: # if y value > 0
                # for type U
                A_U.append(180 - d)
                B_U.append(360 - d)
                # for type D
                if p[0] < 0:
                    A_D.append(180)
                    B_D.append(360)
                elif p[0] > 0:
                    A_D.append(0)
                    B_D.append(180)
                else:
                    A_D.append(90)
                    B_D.append(270)
            else:
                # for type U
                if p[0] < 0:
                    A_U.append(0)
                    B_U.append(180)
                elif p[0] > 0:
                    A_U.append(180)
                    B_U.append(360)
                else:
                    A_U.append(90)
                    B_U.append(270)
                # for type D
                A_D.append(abs(d))
                B_D.append(abs(d) + 180)

        A_U = np.array(A_U)
        A_D = np.array(A_D)
        B_U = np.array(B_U)
        B_D = np.array(B_D)

        return [A_U, A_D, B_U, B_D], [sum(A_U), sum(A_D), sum(B_U), sum(B_D)]

    def sample_test(self, circles, sample, path, name):
        test = np.zeros((42, 42))
        for grads, circle in zip(sample, circles):
            circle = np.add(circle, np.array([21, 21]))
            for loc, grad in zip(circle, grads):
                test[-loc[1], loc[0]] = grad # convert cordinate
        self.ImgIO.write_image(path, "raw-" + name, (test / 360 * 255).astype(np.uint8))
        self.ImgIO.write_image(path, name, self.ImgP.convert_gradient(test))

    def circle_cut(self, circles, r, offset, AU_samples, AD_samples, BU_samples, BD_samples):
        au_circles = []
        ad_circles = []
        bu_circles = []
        bd_circles = []
        au_grads = []
        ad_grads = []
        bu_grads = []
        bd_grads = []
        for circle, gs_au, gs_ad, gs_bu, gs_bd in zip(circles, AU_samples, AD_samples, BU_samples, BD_samples):
            au_circle = []
            bu_circle = []
            ad_circle = []
            bd_circle = []
            au_grad = []
            ad_grad = []
            bu_grad = []
            bd_grad = []
            for cir, g_au, g_ad, g_bu, g_bd in zip(circle, gs_au, gs_ad, gs_bu, gs_bd):
                # now is x y cordinates
                cir = [cir[0], cir[1]]
                if cir[1] < 0:
                    if abs(cir[1]) <= (r - offset): # only keep the within part
                        au_circle.append(cir)
                        bu_circle.append(cir)
                        au_grad.append(g_au)
                        bu_grad.append(g_bu)
                    ad_circle.append(cir)
                    bd_circle.append(cir)
                    ad_grad.append(g_ad)
                    bd_grad.append(g_bd)
                elif cir[1] > 0:
                    au_circle.append(cir)
                    bu_circle.append(cir)
                    au_grad.append(g_au)
                    bu_grad.append(g_bu)
                    if cir[1] <= (r - offset):
                        ad_circle.append(cir)
                        bd_circle.append(cir)
                        ad_grad.append(g_ad)
                        bd_grad.append(g_bd)
                else:
                    au_circle.append(cir)
                    bu_circle.append(cir)
                    au_grad.append(g_au)
                    bu_grad.append(g_bu)
                    ad_circle.append(cir)
                    bd_circle.append(cir)
                    ad_grad.append(g_ad)
                    bd_grad.append(g_bd)
            au_circles.append(au_circle)
            ad_circles.append(ad_circle)
            bu_circles.append(bu_circle)
            bd_circles.append(bd_circle)
            au_grads.append(au_grad)
            ad_grads.append(ad_grad)
            bu_grads.append(bu_grad)
            bd_grads.append(bd_grad)

        # print(au_circles)
        # print(au_grads)

        return [au_circles, ad_circles, bu_circles, bd_circles], au_grads, ad_grads, bu_grads, bd_grads

    # Rotate a sample by n degree, basiclly add n toi each grad and if it is over 360, we wrap around it
    def rotate_sample(self):
        pass