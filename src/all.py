import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, cholesky
from scipy.stats import norm

v_i = 0.1
w_i = 0.063
v_j = 0.139
w_j = 0.09


class EKFRangePlusBearingWithoutCommunication:
    def __init__(self):
        self.prev_m_vecX_2 = 0
        self.m_vecX = np.array([1.0, 1.0, 1.5, 0.1, 0.1], dtype=np.float32)
        self.m_matP = np.zeros((5, 5), dtype=np.float32)
        self.m_matQ = 0.0001 * np.eye(5, dtype=np.float32)
        self.m_jacobian_matF = np.zeros((5, 5), dtype=np.float32)
        self.m_vecZ = np.zeros(2, dtype=np.float32)
        self.m_vech = np.zeros(2, dtype=np.float32)
        self.m_matR = np.array([[0.0001, 0], [0, 0.001]], dtype=np.float32)
        self.m_jacobian_matH = np.zeros((2, 5), dtype=np.float32)

    def getVecX(self):
        return self.m_vecX

    def getMatP(self):
        return self.m_matP

    def getvecZ(self):
        return self.m_vecZ

    def motionModelJacobian(self, vec, delta_t):
        self.m_jacobian_matF = np.array([
            [0, w_i, -vec[3] * np.sin(vec[2]), np.cos(vec[2]), 0],
            [-w_i, 0, vec[3] * np.cos(vec[2]), np.sin(vec[2]), 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.float32)
        self.m_jacobian_matF = np.eye(5) + delta_t * self.m_jacobian_matF

    def motionModel(self, vec, delta_t):
        tmp_vec = np.array([
            vec[3] * np.cos(vec[2]) + w_i * vec[1] - v_i,
            vec[3] * np.sin(vec[2]) - w_i * vec[0],
            vec[4] - w_i,
            0,
            0], dtype=np.float32)

        self.m_vecX += delta_t * tmp_vec

    def prediction(self, delta_t):
        self.motionModelJacobian(self.m_vecX, delta_t)
        self.motionModel(self.m_vecX, delta_t)
        self.m_matP = np.dot(np.dot(self.m_jacobian_matF, self.m_matP), self.m_jacobian_matF.T) + self.m_matQ

    def measurementModel(self, vec):
        self.m_vech[0] = np.sqrt(vec[0] ** 2 + vec[1] ** 2)
        self.m_vech[1] = np.arctan2(vec[1], vec[0])

    def measurementModelJacobian(self, vec):
        self.m_jacobian_matH = np.array([
            [vec[0] / np.sqrt(vec[0] ** 2 + vec[1] ** 2), vec[1] / np.sqrt(vec[0] ** 2 + vec[1] ** 2), 0, 0, 0],
            [-vec[1] / (vec[0] ** 2 + vec[1] ** 2), vec[0] / (vec[0] ** 2 + vec[1] ** 2), 0, 0, 0]], dtype=np.float32)

    def correction(self):
        self.measurementModel(self.m_vecX)
        self.measurementModelJacobian(self.m_vecX)

        residual = self.m_vecZ - self.m_vech

        residual_cov = np.dot(np.dot(self.m_jacobian_matH, self.m_matP), self.m_jacobian_matH.T) + self.m_matR


        Kk = np.dot(np.dot(self.m_matP, self.m_jacobian_matH.T), np.linalg.inv(residual_cov))

        self.m_vecX += np.dot(Kk, residual)
        self.m_matP = np.dot((np.eye(5) - np.dot(Kk, self.m_jacobian_matH)), self.m_matP)

class UKFRangePlusBearingWithoutCommunication:
    def __init__(self):
        self.m_vecX = np.array([1.0, 1.0, 1.5, 0.1, 0.1], dtype=np.float32)
        self.m_matP = np.zeros((5, 5), dtype=np.float32)+1e-6*np.eye(5)
        self.m_vecZ = np.zeros(2, dtype=np.float32)
        self.m_matQ = 0.0001 * np.eye(5, dtype=np.float32)
        self.m_matR = np.array([[0.0001, 0], [0, 0.001]], dtype=np.float32)
    
        self.H = np.zeros((2,11), dtype=np.float32)
        self.kappa = 0
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
        # vec = np.zeros((5,1), dtype= np.float32)
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
    #    cov = W(func_sampled_Xi - mean@(func_sampled_Xi - mean)).T
       self.m_vecX = mean
       self.m_matP = cov+noiseCov
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

class PFRangePlusBearingWithoutCommunication:
    def __init__(self):
        self.m_vecX = np.array([1.0, 1.0, 1.5, 0.1, 0.1], dtype=np.float32)
        self.m_vecZ = np.zeros(2, dtype=np.float32)        
        self.wt = np.zeros((1,10000), dtype=np.float32)
        self.H = np.zeros((2,10000), dtype=np.float32)

    def getVecX(self):
        return self.m_vecX

    def init_pt_wt(self):
        Npt = 10000
        self.pt = np.zeros((5,Npt),dtype=np.float32)
        self.pt = self.getVecX().reshape(-1, 1)+0.001*self.getVecX().reshape(-1, 1)*np.random.randn(1,Npt) #1xNpt행렬 무작위 곱함
        self.wt = np.ones((1,Npt))/Npt
        
    def getvecZ(self):
        return self.m_vecZ  

    def fx(self, pt, delta_t):
        xdot = np.zeros((5, pt.shape[1]), dtype=np.float32)
        xdot[0, :] = pt[3, :] * np.cos(pt[2, :]) + w_i * pt[1, :] - v_i
        xdot[1, :] = pt[3, :] * np.sin(pt[2, :]) - w_i * pt[0, :]
        xdot[2, :] = pt[4, :] - w_i
        xdot[3, :] = 0
        xdot[4, :] = 0
        x_pred = pt + delta_t * xdot
        return x_pred

    
    def hx(self, pt):
        _x,_y,_,_,_ = pt
        self.H[0,:] = np.sqrt(_x ** 2 + _y ** 2)
        self.H[1,:] = np.arctan2(_y, _x)
        return self.H
    
    def particle_filter(self, delta_t):
        self.pt = self.fx(self.pt,delta_t) + np.random.randn(*self.pt.shape)
        self.wt = self.wt*(norm.pdf(self.getvecZ()[0], self.hx(self.pt)[0], 0.7099)+norm.pdf(self.getvecZ()[1], self.hx(self.pt)[1], 0.999)) 
        #평균이 getvecZ이고 표준편차가 0.7099, 0.999인 정규분포를 생성하여 가중치에 곱함 -> 샘플중 getvexZ근처에 있는 값이 크므로 얘들이 더 큰 가중치를 가짐
        self.wt = self.wt / np.sum(self.wt) # 가중치 다 더하면 1
        self.m_vecX = self.pt @ self.wt.T # 샘플마다 가중치 합, 가중치 큰 값이 영향 많이 줌
        Npt = self.pt.shape[1]
        inds = np.random.choice(Npt, Npt, p=self.wt[0], replace=True)  #pt배열에서 pt개수만큼 다시 choice 근데 wt의 가중치가 높은 값이 선택될 확률 더 커짐
        # np.random.choice(배열, 몇개 선택, 각 요소마다 선택될 확률, 중복 허용)
        self.pt = self.pt[:, inds] #예를들어 1~100까지 수 중에 50의 가중치가 제일 커서 inds안에 50이 100개중 20개나 차지 그럼 그 50에 대한 열벡터 값들이 pt에 20개나 저장됨
        self.wt = np.ones((1, Npt)) / Npt  # 가중치 초기화
        
class ESEKF:
    def __init__(self):
        self.prev_m_vecX_2 = 0
        self.m_vecX = np.array([1.0, 1.0, 1.5, 0.1, 0.1], dtype=np.float32)
        self.m_matP = np.zeros((5, 5), dtype=np.float32)
        self.m_matQ = 0.0001 * np.eye(5, dtype=np.float32)
        self.m_jacobian_matF = np.zeros((5, 5), dtype=np.float32)
        self.m_vecZ = np.zeros(2, dtype=np.float32)
        self.m_vech = np.zeros(2, dtype=np.float32)
        self.m_matR = np.array([[0.0001, 0], [0, 0.001]], dtype=np.float32)
        self.m_jacobian_matH = np.zeros((2, 5), dtype=np.float32)
        self.errorstate_jacobian_matH = np.zeros((5, 5), dtype=np.float32)
        self.m_matG = np.zeros((5, 5), dtype=np.float32)
        
    def getVecX(self):
        return self.m_vecX

    def getMatP(self):
        return self.m_matP

    def getvecZ(self):
        return self.m_vecZ

    def motionModelErrorStateJacobian(self, delta_t):
        vec = np.array([0,0,0,0,0],dtype=np.float32) # errorVec_x = 0
        self.m_jacobian_matF = np.array([
            [0, w_i, -vec[3] * np.sin(vec[2]), np.cos(vec[2]), 0],
            [-w_i, 0, vec[3] * np.cos(vec[2]), np.sin(vec[2]), 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.float32)
        self.m_jacobian_matF = np.eye(5) + delta_t * self.m_jacobian_matF

    def motioinModel(self, vec, delta_t):
        tmp_vec = np.array([
            vec[3] * np.cos(vec[2]) + w_i * vec[1] - v_i,
            vec[3] * np.sin(vec[2]) - w_i * vec[0],
            vec[4] - w_i,
            0,
            0], dtype=np.float32)

        self.m_vecX += delta_t * tmp_vec        

    def prediction(self, delta_t):
        self.motionModelErrorStateJacobian(delta_t)
        self.motioinModel(self.m_vecX, delta_t)
        self.m_matP = np.dot(np.dot(self.m_jacobian_matF, self.m_matP), self.m_jacobian_matF.T) + self.m_matQ

    def measurementModel(self, vec):
        self.m_vech[0] = np.sqrt(vec[0] ** 2 + vec[1] ** 2)
        self.m_vech[1] = np.arctan2(vec[1], vec[0])

    def measurementModelJacobian(self, vec):
        self.m_jacobian_matH = np.array([
            [vec[0] / np.sqrt(vec[0] ** 2 + vec[1] ** 2), vec[1] / np.sqrt(vec[0] ** 2 + vec[1] ** 2), 0, 0, 0],
            [-vec[1] / (vec[0] ** 2 + vec[1] ** 2), vec[0] / (vec[0] ** 2 + vec[1] ** 2), 0, 0, 0]], dtype=np.float32)
        
    def measurementModelErrorStateJacobian(self):
        vec = np.array([0,0,0,0,0],dtype=np.float32) # errorVec_x = 0
        tmp = np.eye(5, dtype=np.float32)
        self.errorstate_jacobian_matH = np.dot(self.m_jacobian_matH, tmp)

    def resetG(self, vec):
        self.m_matG = np.array([
            [0, w_i, -vec[3] * np.sin(vec[2]), np.cos(vec[2]), 0],
            [-w_i, 0, vec[3] * np.cos(vec[2]), np.sin(vec[2]), 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.float32)


    def correction(self):
        self.measurementModel(self.m_vecX)
        self.measurementModelJacobian(self.m_vecX)
        self.measurementModelErrorStateJacobian()
        residual = self.m_vecZ - self.m_vech
        residual_cov = np.dot(np.dot(self.errorstate_jacobian_matH, self.m_matP), self.errorstate_jacobian_matH.T) + self.m_matR
        Kk = np.dot(np.dot(self.m_matP, self.errorstate_jacobian_matH.T), np.linalg.inv(residual_cov))
        self.m_vecX += np.dot(Kk, residual)
        self.m_matP = np.dot((np.eye(5) - np.dot(Kk, self.errorstate_jacobian_matH)), self.m_matP)
        self.resetG(self.m_vecX)
        self.m_matP = np.dot(np.dot(self.m_matG,self.m_matP),self.m_matG.T)

class EKF_UKF_PF_ESEKF_Node:
    def __init__(self):
        self.ekf_range_plus_bearing_without_comm = EKFRangePlusBearingWithoutCommunication()
        self.ukf_range_plus_bearing_without_comm = UKFRangePlusBearingWithoutCommunication()
        self.Pf_range_plus_bearing_without_comm = PFRangePlusBearingWithoutCommunication()
        self.esekf = ESEKF()
        self.Pf_range_plus_bearing_without_comm.init_pt_wt()
        self.rho= []
        self.bearing= []
        self.theta_ji = []

        self.ekf_rms_x = []
        self.ekf_rms_y = []
        self.ekf_rms_t = []

        self.ukf_rms_x = []
        self.ukf_rms_y = []
        self.ukf_rms_t = []

        self.pf_rms_x = []
        self.pf_rms_y = []
        self.pf_rms_t = []

        self.esekf_rms_x = []
        self.esekf_rms_y = []
        self.esekf_rms_t = []

        self.gt_xji = []
        self.gt_yji = []
        self.gt_theta = []
        self.gt_v = []
        self.gt_w = []

        self.ceres_xji = []
        self.ceres_yji = []
        
        self.ekf_esti_x = []
        self.ekf_esti_y = []
        self.ekf_esti_theta = []
        self.ekf_esti_v = []
        self.ekf_esti_w = []

        self.pf_esti_x = []
        self.pf_esti_y = []
        self.pf_esti_theta = []
        self.pf_esti_v = []
        self.pf_esti_w = []

        self.ukf_esti_x = []
        self.ukf_esti_y = []
        self.ukf_esti_theta = []
        self.ukf_esti_v = []
        self.ukf_esti_w = []
        
        self.esekf_esti_x = []
        self.esekf_esti_y = []
        self.esekf_esti_theta = []
        self.esekf_esti_v = []
        self.esekf_esti_w = []

        self.time = []
        with open('new_rho_sc2_1.txt', 'r') as file:
            for_ekf = file.readlines()
        
        with open('gt_sc2_xji.txt', 'r') as file:
            gt = file.readlines()
        
        with open('ceres_sc1_xji.txt', 'r') as file:
            ceres = file.readlines()
        
        for line in for_ekf:
            x, y, z =map(float, line.strip().split("\t"))
            self.rho.append(x)
            self.bearing.append(y)
            self.theta_ji.append(z)

        for line in gt:
            x, y, z, a, b =map(float, line.strip().split("\t"))
            self.gt_xji.append(x)
            self.gt_yji.append(y)
            self.gt_theta.append(z)
            self.gt_v.append(a)
            self.gt_w.append(b)

        for line in ceres:
            x, y =map(float, line.strip().split("\t"))
            self.ceres_xji.append(x)
            self.ceres_yji.append(y)


        for i in range(len(self.rho)):
            
            ekf_delta_t = 0.05
            ukf_delta_t = 0.06
            pf_delta_t = 0.005
            esekf_delta_t = 0.025
            self.ekf_range_plus_bearing_without_comm.getvecZ()[0] = self.rho[i]
            self.ekf_range_plus_bearing_without_comm.getvecZ()[1] = self.bearing[i] * np.pi / 180
            

            self.ekf_range_plus_bearing_without_comm.prediction(ekf_delta_t)
            self.ekf_range_plus_bearing_without_comm.correction()
            self.time.append(i)
            self.ekf_esti_x.append(self.ekf_range_plus_bearing_without_comm.getVecX()[0])
            self.ekf_esti_y.append(self.ekf_range_plus_bearing_without_comm.getVecX()[1])
            self.ekf_esti_theta.append(self.ekf_range_plus_bearing_without_comm.getVecX()[2])
            self.ekf_esti_v.append(self.ekf_range_plus_bearing_without_comm.getVecX()[3])
            self.ekf_esti_w.append(self.ekf_range_plus_bearing_without_comm.getVecX()[4])
            self.ekf_rms_x.append((self.ekf_esti_x[i]-self.gt_xji[i])**2)
            self.ekf_rms_y.append((self.ekf_esti_y[i]-self.gt_yji[i])**2)
            self.ekf_rms_t.append((self.ekf_esti_theta[i]-self.gt_theta[i])**2)


            self.ukf_range_plus_bearing_without_comm.getvecZ()[0] = self.rho[i]
            self.ukf_range_plus_bearing_without_comm.getvecZ()[1] = self.bearing[i] * np.pi / 180

            self.ukf_range_plus_bearing_without_comm.unscented_kalman_filter(self.ukf_range_plus_bearing_without_comm.getvecZ(),
                                                                              self.ukf_range_plus_bearing_without_comm.getVecX(),
                                                                              self.ukf_range_plus_bearing_without_comm.getMatP(),
                                                                                ukf_delta_t)
            # self.time.append(i)
            self.ukf_esti_x.append(self.ukf_range_plus_bearing_without_comm.getVecX()[0])
            self.ukf_esti_y.append(self.ukf_range_plus_bearing_without_comm.getVecX()[1])
            self.ukf_esti_theta.append(self.ukf_range_plus_bearing_without_comm.getVecX()[2])
            self.ukf_esti_v.append(self.ukf_range_plus_bearing_without_comm.getVecX()[3])
            self.ukf_esti_w.append(self.ukf_range_plus_bearing_without_comm.getVecX()[4])
            self.ukf_rms_x.append((self.ukf_esti_x[i] - self.gt_xji[i]) ** 2)
            self.ukf_rms_y.append((self.ukf_esti_y[i] - self.gt_yji[i]) ** 2)
            self.ukf_rms_t.append((self.ukf_esti_theta[i] - self.gt_theta[i]) ** 2)

            self.Pf_range_plus_bearing_without_comm.getvecZ()[0] = self.rho[i]
            self.Pf_range_plus_bearing_without_comm.getvecZ()[1] = self.bearing[i] * np.pi / 180
            self.Pf_range_plus_bearing_without_comm.particle_filter(pf_delta_t)
            # self.time.append(i)
            self.pf_esti_x.append(self.Pf_range_plus_bearing_without_comm.getVecX()[0])
            self.pf_esti_y.append(self.Pf_range_plus_bearing_without_comm.getVecX()[1])
            self.pf_esti_theta.append(self.Pf_range_plus_bearing_without_comm.getVecX()[2])
            self.pf_esti_v.append(self.Pf_range_plus_bearing_without_comm.getVecX()[3])
            self.pf_esti_w.append(self.Pf_range_plus_bearing_without_comm.getVecX()[4])
            self.pf_rms_x.append((self.pf_esti_x[i] - self.gt_xji[i]) ** 2)
            self.pf_rms_y.append((self.pf_esti_y[i] - self.gt_yji[i]) ** 2)
            self.pf_rms_t.append((self.pf_esti_theta[i] - self.gt_theta[i]) ** 2)

            self.esekf.getvecZ()[0] = self.rho[i]
            self.esekf.getvecZ()[1] = self.bearing[i] * np.pi / 180
            self.esekf.prediction(esekf_delta_t)
            self.esekf.correction()
            self.esekf_esti_x.append(self.esekf.getVecX()[0])
            self.esekf_esti_y.append(self.esekf.getVecX()[1])
            self.esekf_esti_theta.append(self.esekf.getVecX()[2])
            self.esekf_esti_v.append(self.esekf.getVecX()[3])
            self.esekf_esti_w.append(self.esekf.getVecX()[4])
            self.esekf_rms_x.append((self.esekf_esti_x[i]-self.gt_xji[i])**2)
            self.esekf_rms_y.append((self.esekf_esti_y[i]-self.gt_yji[i])**2)
            self.esekf_rms_t.append((self.esekf_esti_theta[i]-self.gt_theta[i])**2)

        print("EKF Xji RMS = ",np.sqrt((np.mean(self.ekf_rms_x))))
        print("EKF Yji RMS = ",np.sqrt((np.mean(self.ekf_rms_y))))
        print("EKF Thetaji RMS = ",np.sqrt((np.mean(self.ekf_rms_t))),"\n")

        print("UKF Xji RMS = ",np.sqrt((np.mean(self.ukf_rms_x))))
        print("UKF Yji RMS = ",np.sqrt((np.mean(self.ukf_rms_y))))
        print("UKF Thetaji RMS = ",np.sqrt((np.mean(self.ukf_rms_t))),"\n")

        print("PF Xji RMS = ",np.sqrt((np.mean(self.pf_rms_x))))
        print("PF Yji RMS = ",np.sqrt((np.mean(self.pf_rms_y))))
        print("PF Thetaji RMS = ",np.sqrt((np.mean(self.pf_rms_t))),"\n")
        
        print("ESEKF Xji RMS = ",np.sqrt((np.mean(self.esekf_rms_x))))
        print("ESEKF Yji RMS = ",np.sqrt((np.mean(self.esekf_rms_y))))
        print("ESEKF Thetaji RMS = ",np.sqrt((np.mean(self.esekf_rms_t))))

        fig, axs  =plt.subplots(5,1, figsize=(10,12))

        axs[0].set_ylabel('X ji [m]')
        axs[0].plot(self.gt_xji,linestyle='-', color='black',label = 'true x')
        axs[0].plot(self.ekf_esti_x, linestyle='-',color='red',label = 'ekf x')
        axs[0].plot(self.ukf_esti_x, linestyle='--',color='green',label = 'ukf x')
        axs[0].plot(self.pf_esti_x, linestyle='-',color='blue',label = 'pf x')
        axs[0].plot(self.esekf_esti_x, linestyle='-',color='purple',label = 'esekf x')
        axs[0].grid(True)

        axs[1].set_ylabel('Y ji [m]')
        axs[1].plot( self.gt_yji,linestyle='-', color='black',label = 'true y')
        axs[1].plot( self.ekf_esti_y,linestyle='-', color='red',label = 'ekf y')
        axs[1].plot( self.ukf_esti_y,linestyle='--', color='green',label = 'ukf y')
        axs[1].plot( self.pf_esti_y,linestyle='-', color='blue',label = 'pkf y')
        axs[1].plot(self.esekf_esti_y, linestyle='-',color='purple',label = 'esekf y')
        axs[1].grid(True)

        axs[2].set_ylabel('Tetha ji [rad]')
        axs[2].plot( self.gt_theta,linestyle='-', color='black',label = 'true theta')
        axs[2].plot( self.ekf_esti_theta,linestyle='-', color='red',label = 'ekf theta')
        axs[2].plot( self.ukf_esti_theta,linestyle='--', color='green',label = 'ukf theta')
        # axs[2].plot( self.pf_esti_theta,linestyle='-', color='blue',label = 'pf theta')        
        axs[2].plot(self.esekf_esti_theta, linestyle='-',color='purple',label = 'esekf theta')
        axs[2].grid(True)
        
        axs[3].set_ylabel('V j [m/s]')
        axs[3].plot( self.gt_v,linestyle='-', color='black',label = 'true v')
        axs[3].plot( self.ekf_esti_v,linestyle='-', color='red',label = 'ekf v')
        axs[3].plot( self.ukf_esti_v,linestyle='--', color='green',label = 'ukf v')
        # axs[3].plot( self.pf_esti_v,linestyle='-', color='blue',label = 'pf v')    
        axs[3].plot(self.esekf_esti_v, linestyle='--',color='purple',label = 'esekf v')    
        axs[3].grid(True)

        axs[4].set_ylabel('W j [m]')
        axs[4].plot( self.gt_w,linestyle='-', color='black',label = 'true w')
        axs[4].plot( self.ekf_esti_w,linestyle='-', color='red',label = 'ekf w')
        axs[4].plot( self.ukf_esti_w,linestyle='--', color='green',label = 'ukf w')
        # axs[4].plot( self.pf_esti_w,linestyle='-', color='blue',label = 'pf w')     
        axs[4].plot(self.esekf_esti_w, linestyle='--',color='purple',label = 'esekf w')   
        axs[4].grid(True)

        for ax in axs:
            ax.legend()
        plt.show()
        # with open("new_ekf_xji_yji.txt", 'w') as file:
        #     for x,y,z,a,b  in zip(self.esti_x, self.esti_y, self.esti_theta, self.esti_v, self.esti_w):
        #         file.write(f"{x}\t {y}\t{z}\t{a}\t{b}\n")
                


if __name__ =="__main__":
    EKF_UKF_PF_ESEKF_Node()
