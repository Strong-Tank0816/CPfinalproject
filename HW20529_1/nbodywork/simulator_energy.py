import numpy as np
import matplotlib as plt
from pathlib import Path
from .particles import Particles
from numba import jit, njit, prange, set_num_threads


#This file is just for me to test result, I didn't use this file to compute any particle



@njit(parallel=True)
def calculate_Kinetic_energy_kernel(nparticles, mass, velocity):      #用numba算動能
    KE = 0         #定義動能
    for i in prange(nparticles):
        KE = KE + 0.5*mass[i,0]*(np.linalg(velocity[i,:]))**2       #np.linalg為求向量v的長度(也就是速度大小)
    
    return KE

@njit(parallel=True)
def calculate_Potential_energy_kernel(nparticles, mass, position, acceleration, G, rsoft):      
    PE = 0
    for i in prange(nparticles):
        for j in prange(nparticles):
            if (j>i): 
                rij = position[i,:] - position[j,:]
                r = np.sqrt(np.sum(rij**2) + rsoft**2)
                PE = PE - G * mass[i,0] * mass[j,0] / r**2            #假使有3顆粒子，那這樣未能就會算2+1+0=3次(12,13,23)，剛好是我們要的值
                

    return PE

def calculate_Kinetic_energy(self, nparticles, mass, velocity):
    KE = 0
    KE = calculate_Kinetic_energy_kernel(nparticles, mass, velocity)
    return KE

def calculate_Potential_energy(self, nparticles, mass, position):
    KE = 0
    KE = calculate_Kinetic_energy_kernel(nparticles, mass, position)
    return KE

if __name__ == "__main__":

    pass

