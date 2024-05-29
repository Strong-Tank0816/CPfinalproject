import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit, prange, set_num_threads
@njit(parallel=True)                                                                      
def calculate_PE_kernel(nparticles,mass,position,G=0.1,rsoft=0.01):
    pe = np.zeros(nparticles)
    for i in prange(nparticles):
        for j in prange(i+1,nparticles):            #避免double counting，亦可算出全部粒子的位能再除以2
                rij = position[i,:] - position[j,:]
                r = np.sqrt(np.sum(rij**2) + rsoft**2)
                pe[i] = pe[i] - G * mass[i] * mass[j] / (r)
    return pe         

class Particles:
    #定義粒子的各屬性,tag為第幾顆粒子，ke及pe分別為動能及位能
    def __init__(self,N):                 
        self.nparticles = N
        self.mass = np.zeros((N,1))
        self.position = np.zeros((N,3))
        self.velocity = np.zeros((N,3))
        self.acceleration = np.zeros((N,3))
        self.tag = np.arange(N)
        self.time = 0
        self.ke = np.zeros(N)
        self.pe = np.zeros(N)
        return

    #給粒子設定(初始)狀態
    def setting_particles(self, pos, vel, acc):    
        self.position = pos
        self.velocity = vel
        self.acceleration = acc

        return
    
    #將新的粒子加進系統中
    def adding_particles(self, mass, pos, vel, acc):   
        self.nparticles = self.nparticles + len(mass)
        self.mass = np.vstack((self.mass, mass))
        self.position = np.vstack((self.position, pos))
        self.velocity = np.vstack((self.velocity, vel))
        self.acceleration = np.vstack((self.acceleration, acc))
        return

    #計算動能，並直接帶入self.ke
    def calculate_KE(self):                                 
        self.ke = np.zeros(self.nparticles)
        for i in range(self.nparticles):
            self.ke[i] = 0.5*self.mass[i]*(np.sum(self.velocity[i]**2)) 

    #計算位能，並直接帶入self.pe 
    def calculate_PE(self,G=0.1,rsoft=0.01):                
        self.pe = np.zeros(self.nparticles)
        nparticles = self.nparticles
        mass = self.mass
        position = self.position
        self.pe = calculate_PE_kernel(nparticles,mass,position)

    
    #將疊代的結果數據output出來
    def output(self, filename):                              
        mass = self.mass
        pos = self.position
        vel = self.velocity
        acc = self.acceleration
        tag = self.tag
        time = self.time
        header = """
                ----------------------------------------------------
                Data from a 3D direct N-body simulation. 

                rows are i-particle; 
                coumns are :mass, tag, x ,y, z, vx, vy, vz, ax, ay, az, ke, pe
                The data is for Homework 2.
                ----------------------------------------------------
                """
        header += "Time = {}".format(time)
        np.savetxt(filename,(tag[:],mass[:],pos[:,0],pos[:,1],pos[:,2],
                            vel[:,0],vel[:,1],vel[:,2],
                            acc[:,0],acc[:,1],acc[:,2]),header=header)
        return 
    
    #將疊代出來的particle顯示在塗上ˊ
    def draw(self, dim=2):                                          
        fig = plt.figure()

        if dim == 2:
            ax = fig.add_subplot(111)
            ax.scatter(self.position[:,0], self.position[:,1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
        elif dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.position[:,0], self.position[:,1], self.position[:,2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        else:
            print("Invalid dimension!")
            return

        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()
        return fig, ax
    
    


