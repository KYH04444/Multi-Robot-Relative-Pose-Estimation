import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import norm

v_i = 0.1
w_i = 0.063
v_j = 0.139
w_j = 0.09

class PFRangePlusBearingWithoutCommunication:
    def __init__(self):
        self.m_vecX = np.array([1.0, 1.0, 1.5, 0.1, 0.1], dtype=np.float32)
        self.m_matP = np.zeros((5, 5), dtype=np.float32)+1e-6*np.eye(5)
        self.m_vecZ = np.zeros(2, dtype=np.float32)
        self.m_matQ = 0.0001 * np.eye(5, dtype=np.float32)
        self.m_matR = np.array([[0.0001, 0], [0, 0.001]], dtype=np.float32)
        
        self.wt = np.zeros((1,1000), dtype=np.float32)
        self.H = np.zeros((2,1000), dtype=np.float32)

    def getVecX(self):
        return self.m_vecX

    def init_pt_wt(self):
        Npt = 1000
        self.pt = np.zeros((5,Npt),dtype=np.float32)
        self.pt = self.getVecX().reshape(-1, 1)+0.01*self.getVecX().reshape(-1, 1)*np.random.randn(1,Npt) #1xNpt행렬 무작위 곱함
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
        self.wt = self.wt*(norm.pdf(self.getvecZ()[0], self.hx(self.pt)[0], 0.7099)+norm.pdf(self.getvecZ()[1], self.hx(self.pt)[1], 0.999)) # vecZ값을 평균이 hx(pt)이고 표준편차가 10인 정규분포 pdf를 만들어서 곱함
        self.wt = self.wt / np.sum(self.wt)
        self.m_vecX = self.pt @ self.wt.T
        Npt = self.pt.shape[1]
        inds = np.random.choice(Npt, Npt, p=self.wt[0], replace=True)  #pt배열에서 pt개수만큼 다시 choice 근데 wt의 가중치가 높은 값이 선택될 확률 더 커짐
        # print(inds)
        self.pt = self.pt[:, inds] #예를들어 1~100중 50의 가중치가 제일 커서 inds안에 50이 100개중 20개나 차지 그럼 그 50에 대한 열벡터 값들이 pt에 20개나 저장됨
        self.wt = np.ones((1, Npt)) / Npt  # 가중치 초기화

        
        

class PFNode:
    def __init__(self):
        self.Pf_range_plus_bearing_without_comm = PFRangePlusBearingWithoutCommunication()
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
        self.Pf_range_plus_bearing_without_comm.init_pt_wt()

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
            delta_t = 0.005
            self.Pf_range_plus_bearing_without_comm.getvecZ()[0] = self.rho[i]
            self.Pf_range_plus_bearing_without_comm.getvecZ()[1] = self.bearing[i] * np.pi / 180
            self.Pf_range_plus_bearing_without_comm.particle_filter(delta_t)
            self.time.append(i)
            self.esti_x.append(self.Pf_range_plus_bearing_without_comm.getVecX()[0])
            self.esti_y.append(self.Pf_range_plus_bearing_without_comm.getVecX()[1])
            self.esti_theta.append(self.Pf_range_plus_bearing_without_comm.getVecX()[2])
            self.esti_v.append(self.Pf_range_plus_bearing_without_comm.getVecX()[3])
            self.esti_w.append(self.Pf_range_plus_bearing_without_comm.getVecX()[4])
            self.rms_x.append((self.esti_x[i] - self.gt_xji[i]) ** 2)
            self.rms_y.append((self.esti_y[i] - self.gt_yji[i]) ** 2)
            self.rms_t.append((self.esti_theta[i] - self.gt_theta[i]) ** 2)

        print(np.sqrt((np.mean(self.rms_x))))
        print(np.sqrt((np.mean(self.rms_y))))
        print(np.sqrt((np.mean(self.rms_t))))
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        axs[0].set_ylabel('X ji [m]')
        axs[0].plot(self.gt_xji, linestyle='-', color='blue', label='true x')
        axs[0].plot(self.esti_x, linestyle='-', color='red', label='pf x')
        axs[0].grid(True)

        axs[1].set_ylabel('Y ji [m]')
        axs[1].plot(self.gt_yji, linestyle='-', color='blue', label='true y')
        axs[1].plot(self.esti_y, linestyle='-', color='red', label='pf y')
        axs[1].grid(True)

        for ax in axs:
            ax.legend()
        plt.show()

if __name__ == "__main__":
    PFNode()
