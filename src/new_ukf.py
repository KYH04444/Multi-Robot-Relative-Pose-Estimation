#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, cholesky

v_i = 0.1
w_i = 0.063
v_j = 0.139
w_j = 0.09


class UKFRangePlusBearingWithoutCommunication:
    def __init__(self):
        self.m_vecX = np.array([1.0, 1.0, 1.5, 0.1, 0.1], dtype=np.float32)
        self.m_matP = np.zeros((5, 5), dtype=np.float32)+1e-6*np.eye(5)
        self.m_vecZ = np.zeros(2, dtype=np.float32)
        self.m_matQ = 0.0001 * np.eye(5, dtype=np.float32)
        self.m_matR = np.array([[0.0001, 0], [0, 0.001]], dtype=np.float32)
    
        self.H = np.zeros((2,11), dtype=np.float32)
        self.kappa = 1
    def getVecX(self):
        return self.m_vecX

    def getMatP(self):
        return self.m_matP

    def getvecZ(self):
        return self.m_vecZ   

    def sigma_points(self, vecx, p, kappa):
        n = len(vecx)
        Xi = np.zeros((n, 2*n+1))
        W = np.zeros(2*n+1)

        Xi[:, 0] = vecx
        W[0] = kappa/(n+kappa)

        U = cholesky((n+kappa)*p) # U.transepose()@U = (n+kappa)*p임  
                                  # 대각행렬 콜레스키법 걍 루트 씌우는거랑 같다
                                  # 여기서는 우리가 5개의 데이터[Xji, Yji, Thetaji, Vj, Wj]를 사용하니까
                                  # 2*5+1 = 11개의 샘플링 데이터를 임의로 생성하여 이를 사용
        for i in range(n):
            Xi[:, i+1] = vecx+ U[: , i]
            Xi[:, n+i+1] = vecx - U[: , i]
            W[i+1] = 1/(2*(n+kappa))
            W[n+i+1] = W[i+1]
        return Xi, W
    
    def fx(self, Xi, delta_t):
        # 새롭게 구한 샘플링 데이터 Xi, W로 다시 모션모델에 대한 예측 값을 구함
        # EKF에서는 자코비안
        _, kmax = Xi.shape
        Xji,Yji,Tji,Vj,Wj = Xi
        xdot = np.zeros((5, kmax))
        xdot[0,:] = Vj*np.cos(Tji)+w_i*Yji - v_i
        xdot[1,:] = Vj*np.sin(Tji)-w_i*Xji 
        xdot[2,:] = Wj-w_i
        xdot[3,:] = 0
        xdot[4,:] = 0
        x_pred = Xi + delta_t * xdot
        return x_pred
    
    def hx(self, Xi):
        _x,_y,_1,_2,_3 = Xi
        self.H[0, :] = np.sqrt(_x ** 2 + _y ** 2)
        self.H[1, :] = np.arctan2(_y, _x)
        # UT전 h(Xi)를 구해주자
        return self.H

    def UT(self,func_sampled_Xi,W, noiseCov):
       # reshape(-1,n)의 의미 n개의 열을 생성할때 필요한 행의 개수를 알아서 선택하라
       # 지금은 mean이 5x1이므로 굳이 필요없긴함 reshape한번 활용해보는 것도 좋음
       # reshape(-1,1)을 mean으로 사용해도 무방
       mean = np.sum(W*func_sampled_Xi, axis = 1) #가중치와f(Xi)의 합은 평균
       # mean = func_sampled_Xi[:,0] 이렇게 둬도 동일
       # np.sum에서 axis 값에대한 설명-어떻게 더할지에 대한 표시
       #    row, cloumn ,depth
       #    행 , 열, 깊이 순으로 axis 값 0~2
       #    0이면 행끼리 다 더함, 1이면 열끼리 다 더해서 출력, 2도 마찬가지
       #    지금은 1이므로 열 끼리 더해서 mean은 5x1
       cov = W*(func_sampled_Xi - mean.reshape(-1,1))@(func_sampled_Xi - mean.reshape(-1,1)).T
    #    cov = W(func_sampled_Xi - mean)@(func_sampled_Xi - mean)).T
    #    self.m_vecX = mean
    #    self.m_matP = cov+noiseCov
       return mean, cov + noiseCov
        # 위에서 구한 f(Xi), h(Xi)를 사용하여 추정값, 추정값 오차공분산 그리고 측정값, 측정값 오차공분산을 구해주자
        # xk, Pk = ~~~
        # return xk, pk
    

    def unscented_kalman_filter(self, m_vecZ, x_esti, P, delta_t):

        Xi, W = self.sigma_points(x_esti, P, self.kappa)

        fXi = self.fx(Xi,delta_t)
        x_pred, P_x = self.UT(fXi, W, self.m_matQ)

        hXi = self.hx(fXi)
        z_pred, P_z = self.UT(hXi, W, self.m_matR)

        Pxz = W * (fXi - x_pred.reshape(-1, 1)) @ (hXi - z_pred.reshape(-1, 1)).T
        K = Pxz @ inv(P_z)

        self.m_vecX = x_pred + K @ (m_vecZ - z_pred)
        self.m_matP = P_x - K @ P_z @ K.T

        # return self.m_vecX, P
    
class UKFNode:
    def __init__(self):
        self.ukf_range_plus_bearing_without_comm = UKFRangePlusBearingWithoutCommunication()
        self.rho = []
        self.bearing = []
        self.theta_ji = []

        self.rms_x = []
        self.rms_y = []
        self.rms_t = []

        self.gt_xji = []
        self.gt_yji = []
        self.gt_theta = []
        
        self.esti = []
        self.esti_x = []
        self.esti_y = []
        self.esti_theta = []
        self.esti_v = []
        self.esti_w = []
        self.time = []

        with open('new_rho_sc2_1.txt', 'r') as file:
            for_ukf = file.readlines()

        with open('gt_sc2_xji.txt', 'r') as file:
            gt = file.readlines()

        for line in for_ukf:
            x, y, z = map(float, line.strip().split("\t"))
            self.rho.append(x)
            self.bearing.append(y)
            self.theta_ji.append(z)

        for line in gt:
            x, y, z = map(float, line.strip().split("\t"))
            self.gt_xji.append(x)
            self.gt_yji.append(y)
            self.gt_theta.append(z)

        for i in range(len(self.rho)):
            delta_t = 0.05
            self.ukf_range_plus_bearing_without_comm.getvecZ()[0] = self.rho[i]
            self.ukf_range_plus_bearing_without_comm.getvecZ()[1] = self.bearing[i] * np.pi / 180

            self.ukf_range_plus_bearing_without_comm.unscented_kalman_filter(self.ukf_range_plus_bearing_without_comm.getvecZ(),
                                                                              self.ukf_range_plus_bearing_without_comm.getVecX(),
                                                                              self.ukf_range_plus_bearing_without_comm.getMatP(),
                                                                                delta_t)
            self.time.append(i)
            self.esti_x.append(self.ukf_range_plus_bearing_without_comm.getVecX()[0])
            self.esti_y.append(self.ukf_range_plus_bearing_without_comm.getVecX()[1])
            self.esti_theta.append(self.ukf_range_plus_bearing_without_comm.getVecX()[2])
            self.esti_v.append(self.ukf_range_plus_bearing_without_comm.getVecX()[3])
            self.esti_w.append(self.ukf_range_plus_bearing_without_comm.getVecX()[4])
            self.rms_x.append((self.esti_x[i] - self.gt_xji[i]) ** 2)
            self.rms_y.append((self.esti_y[i] - self.gt_yji[i]) ** 2)
            self.rms_t.append((self.esti_theta[i] - self.gt_theta[i]) ** 2)

        print(np.sqrt((np.mean(self.rms_x))))
        print(np.sqrt((np.mean(self.rms_y))))
        print(np.sqrt((np.mean(self.rms_t))))
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        axs[0].set_ylabel('X ji [m]')
        axs[0].plot(self.gt_xji, linestyle='-', color='blue', label='true x')
        axs[0].plot(self.esti_x, linestyle='-', color='red', label='ukf x')
        axs[0].grid(True)

        axs[1].set_ylabel('Y ji [m]')
        axs[1].plot(self.gt_yji, linestyle='-', color='blue', label='true y')
        axs[1].plot(self.esti_y, linestyle='-', color='red', label='ukf y')
        axs[1].grid(True)

        for ax in axs:
            ax.legend()
        plt.show()

        with open("new_ukf_xji_yji.txt", 'w') as file:
            for x, y, z, a, b in zip(self.esti_x, self.esti_y, self.esti_theta, self.esti_v, self.esti_w):
                file.write(f"{x}\t {y}\t{z}\t{a}\t{b}\n")


if __name__ == "__main__":
    UKFNode()
