import numpy as np
import matplotlib.pyplot as plt

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



class EKFNode:
    def __init__(self):
        self.ekf_range_plus_bearing_without_comm = EKFRangePlusBearingWithoutCommunication()
        self.rho= []
        self.bearing= []
        self.theta_ji = []

        self.rms_x = []
        self.rms_y = []
        self.rms_t = []

        self.gt_xji = []
        self.gt_yji = []
        self.gt_theta = []

        self.ceres_xji = []
        self.ceres_yji = []
        
        self.esti_x = []
        self.esti_y = []
        self.esti_theta = []
        self.esti_v = []
        self.esti_w = []
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
            x, y, z =map(float, line.strip().split("\t"))
            self.gt_xji.append(x)
            self.gt_yji.append(y)
            self.gt_theta.append(z)

        for line in ceres:
            x, y =map(float, line.strip().split("\t"))
            self.ceres_xji.append(x)
            self.ceres_yji.append(y)


        for i in range(len(self.rho)):
            
            delta_t = 0.08
            self.ekf_range_plus_bearing_without_comm.getvecZ()[0] = self.rho[i]
            self.ekf_range_plus_bearing_without_comm.getvecZ()[1] = self.bearing[i] * np.pi / 180
    
            self.ekf_range_plus_bearing_without_comm.prediction(delta_t)
            self.ekf_range_plus_bearing_without_comm.correction()
            self.time.append(i)
            self.esti_x.append(self.ekf_range_plus_bearing_without_comm.getVecX()[0])
            self.esti_y.append(self.ekf_range_plus_bearing_without_comm.getVecX()[1])
            self.esti_theta.append(self.ekf_range_plus_bearing_without_comm.getVecX()[2])
            self.esti_v.append(self.ekf_range_plus_bearing_without_comm.getVecX()[3])
            self.esti_w.append(self.ekf_range_plus_bearing_without_comm.getVecX()[4])
            self.rms_x.append((self.esti_x[i]-self.gt_xji[i])**2)
            self.rms_y.append((self.esti_y[i]-self.gt_yji[i])**2)
            self.rms_t.append((self.esti_theta[i]-self.gt_theta[i])**2)
        
        print(np.sqrt((np.mean(self.rms_x))))
        print(np.sqrt((np.mean(self.rms_y))))
        print(np.sqrt((np.mean(self.rms_t))))
        fig, axs  =plt.subplots(2,1, figsize=(10,12))

        axs[0].set_ylabel('X ji [m]')
        axs[0].plot(self.gt_xji,linestyle='-', color='blue',label = 'true x')
        axs[0].plot(self.esti_x, linestyle='-',color='red',label = 'ekf x')
        axs[0].grid(True)

        axs[1].set_ylabel('Y ji [m]')
        axs[1].plot( self.gt_yji,linestyle='-', color='blue',label = 'true y')
        axs[1].plot( self.esti_y,linestyle='-', color='red',label = 'ekf y')
        axs[1].grid(True)
        for ax in axs:
            ax.legend()
        plt.show()
        with open("new_ekf_xji_yji.txt", 'w') as file:
            for x,y,z,a,b  in zip(self.esti_x, self.esti_y, self.esti_theta, self.esti_v, self.esti_w):
                file.write(f"{x}\t {y}\t{z}\t{a}\t{b}\n")
                


if __name__ =="__main__":
    EKFNode()