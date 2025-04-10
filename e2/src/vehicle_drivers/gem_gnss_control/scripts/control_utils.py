import numpy as np
from math import sin, cos, tan, sqrt
import matplotlib.pyplot as plt

class CarModel():

    def __init__(self, carLength, steerSpeed, carAccel, carDecel, carDamp, steerLimits, throttleLimits, brakeLimits, throttleRateLimits, brakeRateLimits, approx_steerTau):
        self.L = carLength
        self.deltaROC = steerSpeed  # (rad/s) rate of change for steering angle
        self.vAccel = carAccel      # (m/s^2) car acceleration for throttle = 1
        self.vDecel = carDecel      # (m/s^2) car deceleration for brake = 1
        self.vDamp = carDamp        # damping to slow the car down (a = -v_damp*v)
        self.steerTau = approx_steerTau     # approx. time constant (1st-order) of the steering angle (must be slower than actual)
        self.delta_min, self.delta_max = steerLimits[0], steerLimits[1]
        self.throttle_min, self.throttle_max = throttleLimits[0], throttleLimits[1]
        self.throttleROC_min, self.throttleROC_max = throttleRateLimits[0], throttleRateLimits[1]
        self.brake_min, self.brake_max = brakeLimits[0], brakeLimits[1]
        self.brakeROC_min, self.brakeROC_max = brakeRateLimits[0], brakeRateLimits[1]
        self.n = 5  # model dimensions
        self.m = 3
    
    def dx(self, x, u):
        '''
            x = [x; y; theta; delta (steering angle); v (car speed)]
            u = [delta_des; throttle; brake]
        '''
        dx = np.array([ x[4]*cos(x[2]),
                        x[4]*sin(x[2]),
                        x[4]/self.L*tan(x[3]),
                        np.sign(u[0] - x[3])*self.deltaROC,
                        -x[4]*self.vDamp + u[1]*self.vAccel + u[2]*self.vDecel ])
        return dx

    def linearize(self, xs):
        A = np.array([  [0, 0, -xs[4]*sin(xs[2]), 0, cos(xs[2])],
                        [0, 0, xs[4]*cos(xs[2]), 0, sin(xs[2])],
                        [0, 0, 0, xs[4]/(self.L*cos(xs[3])**2), tan(xs[3])/self.L],
                        [0, 0, 0, -1/self.steerTau, 0],
                        [0, 0, 0, 0, -self.vDamp]    ])
        B = np.array([  [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [1/self.steerTau, 0, 0],
                        [0, self.vAccel, self.vDecel] ])
        return A,B

    def saturateControl(self, u, u0, dt):
        # Limit rate
        u[1] = u0[1] + self.saturate((u[1] - u0[1])/dt, self.throttleROC_min, self.throttleROC_max)*dt
        u[2] = u0[2] + self.saturate((u[2] - u0[2])/dt, self.brakeROC_min, self.brakeROC_max)*dt
        
        # Limit value
        u[0] = self.saturate(u[0], self.delta_min, self.delta_max)
        u[1] = self.saturate(u[1], self.throttle_min, self.throttle_max)
        u[2] = self.saturate(u[2], self.brake_min, self.brake_max)
        
        return u

    def saturate(self, val, lower_bound, upper_bound):
        return max(lower_bound, min(upper_bound, val))


class LQR():    # continuous-time only

    def __init__(self, n, m):
        self.type = type
        self.n = n
        self.m = m
    
    def setWeight(self, Q, R):
        self.Q = Q
        self.R = R
        self.R_inv = np.linalg.inv(R)

    def setModel(self, A, B):
        self.A = A
        self.B = B

    def calculateGain(self):
        H = np.vstack((np.hstack((self.A, -self.B.dot(self.R_inv.dot(self.B.T)))),
                        np.hstack((-self.Q, -self.A.T))))
        eigVal, eigVec = np.linalg.eig(H)
        U = eigVec[:, np.argsort(eigVal)]
        U = U[:, :self.n]       # 'stable' subspace - n smallest eigenvalues
        U11 = U[:self.n, :]
        U21 = U[self.n:, :]
        try:
            P = U21.dot(np.linalg.inv(U11))
        except:
            # raise Exception('U11 is not invertible!')
            P = U21.dot(np.linalg.pinv(U11))
        K = self.R_inv.dot(self.B.T.dot(P.real))
        self.K = K
        # return K


class Geometry():

    def __init__(self):
        pass

    def point_point_distance(self, p1, p2):
        '''
            Inputs must be numpy arrays (and not python lists)
        '''
        return np.linalg.norm(p1 - p2)

    def point_line_distance(self, p, s):  # perpendicular distance from point to line
        """Perpendicular distance from point to line.

            Args:
                p: A tuple (x, y) of the coordinates of the point.
                s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating 2 points on the line.

            Return:
                Euclidean distance from the point to the line.
        """
        # AB = self.vector(s[0], s[1])
        # AE = self.vector(s[0], p)
        AB = s[1] - s[0]
        AE = p - s[0]
        # return abs(self.cross(AB,AE)) / self.norm(AB)
        return np.cross(AB,AE) / np.linalg.norm(AB)
        
    def fit_line(self, p):
        """Least square fit line y = a0 + a1*x.

            Args:
                p = [[x1,y1], [x2,y2], ...]

            Return:
                2 points on the fitted line.
        """
        n = len(p)
        x = [p[i][0] for i in range(n)]
        y = [p[i][1] for i in range(n)]
        a0 = (sum(y)*sum(np.power(x,2)) - sum(x)*sum(np.multiply(x,y))) / (n*sum(np.power(x,2)) - pow(sum(x),2))
        a1 = (n*sum(np.multiply(x,y)) - sum(x)*sum(y)) / (n*sum(np.power(x,2)) - pow(sum(x),2))
        return [[p[i][0], a0 + a1*p[i][0]] for i in range(2)]

    def calculateVref(self, curr_pos, future_unreached_waypoints, lookahead, defaultV):
        '''
            defaulV: [Vcurve, Vstraight]
        '''
        target_velocity = defaultV[0]    # should work safely for curves
        fiterror = .1    # max allowed error for line fit
        for i in range(3, len(future_unreached_waypoints)):     # start with the first 3 waypoints
            wp = future_unreached_waypoints[:i]
            line_p = self.fit_line(wp)
            e = [abs(self.point_line_distance(p,line_p)) for p in wp]
            emax = max(e)
            l_d = np.linalg.norm(wp[-1] - curr_pos) # lookahead distance to the furthest point

            if emax <= fiterror and l_d >= lookahead:  # straight path ahead
                # print('Straight path ahead! vel=' + str(curr_vel))
                target_velocity = defaultV[1]
                break
            
            if emax > fiterror or l_d >= lookahead:     # either failed to fit or looked far enough ahead
                # print('Could not fit a straight line! vel=' + str(curr_vel))
                break

        ####################### TODO: Your TASK 2 code ends Here #######################
        return target_velocity

