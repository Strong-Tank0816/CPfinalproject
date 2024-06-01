import numpy as np
import matplotlib as plt
from pathlib import Path
from .particles import Particles
from numba import jit, njit, prange, set_num_threads

#用kernel去跑加速度增加效率
@njit(parallel=True)                                                                      
def calculate_accleration_kernel(nparticles, mass, position, acceleration, G, rsoft):
    for i in prange(nparticles):
        for j in prange(nparticles):
            if (j>i): 
                rij = position[i,:] - position[j,:]
                r = np.sqrt(np.sum(rij**2) + rsoft**2)
                force = - G * mass[i] * mass[j] * rij / r**3
                acceleration[i,:] += force[:] / mass[i]
                acceleration[j,:] -= force[:] / mass[j]

    return acceleration

class NBodySimulator:
    #定義例子的各屬性，並input particles.py的數據來進行疊代
    def __init__(self, particles: Particles):
        self.particles = particles
        self.time = particles.time                
        self.setup()
        return
    
    #設置好(並可以透過函數input改變)必要的參數，並準備好寫output函數所需參數
    def setup(self, G=0.1, rsoft=0.01, method="RK4", outfreq=10,                            
              outheader="nbody", outscreen=True, visualiztion=False):    
        self.G = G                                   #G:重力常數
        self.rsoft = rsoft                           #rosft:避免出現極端值(r->0)而加的緩衝值           
        self.method = method                         #method:決定要用哪一種疊代方法   
        self.outfreq = outfreq                       #outfreq:多久Output一次數據
        self.outheader = outheader
        self.outscreen = outscreen
        self.visualiztion = visualiztion
        return
    
    #準備疊代，疊代方法由setup輸入的值決定，粒子的狀態來自particles.py
    def evolve(self, dt:float, tmax:float):           

        self.dt = dt
        self.tmax = tmax
        nsteps = int(np.ceil(tmax/dt))        #np.ceil功能為無條件進入
        time = self.time
        method = self.method
        particles = self.particles

        #這邊我是用教授的RK2跟RK4 method，我亦有自己寫RK2跟RK4(用泰勒展開的方式)，不過算過跟老師的方法結果會差一點(有時比較準有時比較不準)。
        if method.lower() == "euler":
            simulation_method = self.simulation_Euler
        elif method.lower() == "rk2":
            simulation_method = self._advance_particles_RK2          
        elif method.lower() == "rk4":
            simulation_method = self._advance_particles_RK4
        elif method.lower() == "frog":
            simulation_method = self.simulation_Leap_frog
        else:
            raise ValueError("Unknown method")
        
        #這裡是真正疊代的地方，下面的method只是計算一次dt，這裡才是對時間做疊代
        outfolder = "data_"+self.outheader
        Path(outfolder).mkdir(parents=True, exist_ok=True)
        for n in range(nsteps):
            if (dt > tmax-time):
                dt = tmax - time           #避免dt太大
            
            particles = simulation_method(dt, particles)

            if (n % self.outfreq == 0):        #也就是在一定間隔內output一次數據 ex. outfreq=200-->200*dt做一次疊代
                if self.outscreen:
                    print("n", n, "Time", time, "dt:", dt )
                fn = self.outheader + "_" + str(n).zfill(7)+".dat"
                fn = outfolder+"/"+fn
                particles.output(fn)

            time+=dt
        
        print("Simulation finished.")
        return
    
    #這裡計算加速度(先用kernel計算完再丟回來)，再用這個函數回傳的結果帶入我們的Euler,RK2,RK4疊代
    def calculate_accleration(self, nparticles, mass, position):         
        accleration = np.zeros_like(position)
        G = self.G
        rsoft = self.rsoft

        accleration = calculate_accleration_kernel(nparticles, mass, position, accleration, G, rsoft)

        return accleration


    #Euler method(1st order)
    def simulation_Euler(self, dt, particles):              
        nparticles = particles.nparticles                                       
        mass = particles.mass
        position = particles.position
        velocity = particles.velocity
        accleration = self.calculate_accleration(nparticles, mass, position)

        #把更新後的值output
        position = position + velocity*dt
        velocity = velocity + accleration*dt
        accleration = self.calculate_accleration(nparticles, mass, position)
        particles.setting_particles(position,velocity,accleration)
        return particles
    
    #RK2 method(2nd order)(我自己寫的RK2疊代方法)
    def simulation_RK2(self, dt, particles):                                    
        nparticles = particles.nparticles                                       #input資料進去疊代
        mass = particles.mass
        position = particles.position
        velocity = particles.velocity
        accleration = self.calculate_accleration(nparticles, mass, position)    #RK2其實就是泰勒展開到第2項，所以就position而言，新的position = position + velocity*dt + accleration*(dt)^2/2
        pos = position                                                          #而就velocity而言，新的velocity = velocity + accleration(input position)*dt + accleration(input position的一次微分,也就是velocity)*(dt)^2/2          
        posd1 = velocity  #position的一次微分
        posd2 = accleration
        vel = velocity
        veld1 = accleration #velocity的一次微分
        veld2 = self.calculate_accleration(nparticles, mass, posd1)

        #所以我得到新的position跟velocity
        position = pos + posd1*dt + posd2*(dt**2)/2
        velocity = vel + veld1*dt + veld2*(dt**2)/2
        accleration = self.calculate_accleration(nparticles, mass, position)
        particles.setting_particles(position, velocity, accleration)
        return particles
    
    #RK2 method(2nd order)
    def _advance_particles_RK2(self, dt, particles):

        nparticles = particles.nparticles
        mass = particles.mass
        pos = particles.position
        vel = particles.velocity
        acc = self.calculate_accleration(nparticles, mass, pos)


        pos2 = pos + vel*dt 
        vel2 = vel + acc*dt
        acc2 = self.calculate_accleration(nparticles, mass, pos2) 

        pos2 = pos2 + vel2*dt
        vel2 = vel2 + acc2*dt


        pos = 0.5*(pos + pos2)
        vel = 0.5*(vel + vel2)
        acc = self.calculate_accleration(nparticles, mass, pos)

        particles.setting_particles(pos, vel, acc)

        return particles
    def _advance_particles_RK2_dm(self, dt, particles,d):

        nparticles = particles.nparticles
        mass = particles.mass
        particles.mass[0]-=d*mass[0]*dt
        pos = particles.position
        vel = particles.velocity
        acc = self.calculate_accleration(nparticles, mass, pos)


        pos2 = pos + vel*dt 
        vel2 = vel + acc*dt
        acc2 = self.calculate_accleration(nparticles, mass, pos2) 

        pos2 = pos2 + vel2*dt
        vel2 = vel2 + acc2*dt


        pos = 0.5*(pos + pos2)
        vel = 0.5*(vel + vel2)
        acc = self.calculate_accleration(nparticles, mass, pos)

        particles.setting_particles(pos, vel, acc)

        return particles  
    



    def _advance_particles_RK4_dm(self, dt, particles,d):
        
        nparticles = particles.nparticles
        particles.mass[0]-=d*mass[0]*dt
        mass = particles.mass
        
        # y0
        pos = particles.position
        vel = particles.velocity # k1
        acc = self.calculate_accleration(nparticles, mass, pos) # k1

        dt2 = dt/2
        # y1
        pos1 = pos + vel*dt2
        vel1 = vel + acc*dt2 # k2
        acc1 = self.calculate_accleration(nparticles, mass, pos1) # k2
        
        # y2
        pos2 = pos + vel1*dt2
        vel2 = vel + acc1*dt2 # k3
        acc2 = self.calculate_accleration(nparticles, mass, pos2) # k3

        # y3
        pos3 = pos + vel2*dt
        vel3 = vel + acc2*dt # k4
        acc3 = self.calculate_accleration(nparticles, mass, pos3) # k4

        # rk4
        pos = pos + (vel + 2*vel1 + 2*vel2 + vel3)*dt/6
        vel = vel + (acc + 2*acc1 + 2*acc2 + acc3)*dt/6
        acc = self.calculate_accleration(nparticles, mass, pos)

        # update the particles
        particles.setting_particles(pos, vel, acc)

        return particles






    #RK4 method(4th order)(我自己寫的RK4)
    def simulation_RK4(self, dt, particles):
        nparticles = particles.nparticles                                       #input資料進去疊代
        mass = particles.mass
        position = particles.position
        velocity = particles.velocity
        accleration = self.calculate_accleration(nparticles, mass, position)
        #同上，RK4其實就是泰勒展開到第四項    
        pos = position
        posd1 = velocity  #position的一次微分
        posd2 = accleration
        posd3 = self.calculate_accleration(nparticles, mass, posd1)
        posd4 = self.calculate_accleration(nparticles, mass, posd2)
        
        vel = velocity
        veld1 = accleration #velocity的一次微分
        veld2 = self.calculate_accleration(nparticles, mass, posd1)
        veld3 = self.calculate_accleration(nparticles, mass, posd2)
        veld4 = self.calculate_accleration(nparticles, mass, posd3)

        position = pos + posd1*dt + posd2*(dt**2)/2 + posd3*(dt**3)/6 + posd4*(dt**4)/24
        velocity = vel + veld1*dt + veld2*(dt**2)/2 + veld3*(dt**3)/6 + veld4*(dt**4)/24
        accleration = self.calculate_accleration(nparticles, mass, position)

        particles.setting_particles(position, velocity, accleration) #更新數值
        
        return particles
    
    #RK4 method(4th order)
    def _advance_particles_RK4(self, dt, particles):
        
        nparticles = particles.nparticles
        mass = particles.mass

        # y0
        pos = particles.position
        vel = particles.velocity # k1
        acc = self.calculate_accleration(nparticles, mass, pos) # k1

        dt2 = dt/2
        # y1
        pos1 = pos + vel*dt2
        vel1 = vel + acc*dt2 # k2
        acc1 = self.calculate_accleration(nparticles, mass, pos1) # k2
        
        # y2
        pos2 = pos + vel1*dt2
        vel2 = vel + acc1*dt2 # k3
        acc2 = self.calculate_accleration(nparticles, mass, pos2) # k3

        # y3
        pos3 = pos + vel2*dt
        vel3 = vel + acc2*dt # k4
        acc3 = self.calculate_accleration(nparticles, mass, pos3) # k4

        # rk4
        pos = pos + (vel + 2*vel1 + 2*vel2 + vel3)*dt/6
        vel = vel + (acc + 2*acc1 + 2*acc2 + acc3)*dt/6
        acc = self.calculate_accleration(nparticles, mass, pos)

        # update the particles
        particles.setting_particles(pos, vel, acc)

        return particles
    
    #Leap frog method 
    def simulation_Leap_frog(self, dt, particles):
        nparticles = particles.nparticles                                       #input資料進去疊代
        mass = particles.mass
        position = particles.position
        velocity = particles.velocity
        accleration = self.calculate_accleration(nparticles, mass, position)

        velocity = velocity + accleration*(dt/2)
        position = position + velocity*dt
        accleration = self.calculate_accleration(nparticles, mass, position) #先用新的position得出新的accleration
        velocity = velocity + accleration*(dt/2)                             #再帶回去疊代velocity

        particles.setting_particles(position, velocity, accleration)

        return particles
    def evolve_sun_mass_decreace(self, dt:float, tmax:float,d):  #d:每經過dt，太陽掉多少質量         

        self.dt = dt
        self.tmax = tmax
        nsteps = int(np.ceil(tmax/dt))        #np.ceil功能為無條件進入
        time = self.time
        method = self.method
        particles = self.particles

        #這邊我是用教授的RK2跟RK4 method，我亦有自己寫RK2跟RK4(用泰勒展開的方式)，不過算過跟老師的方法結果會差一點(有時比較準有時比較不準)。
       
        if method.lower() == "rk2":
            simulation_method = self._advance_particles_RK2_dm          
        elif method.lower() == "rk4":
            simulation_method = self._advance_particles_RK4_dm
        else:
            raise ValueError("Unknown method")
        
        #這裡是真正疊代的地方，下面的method只是計算一次dt，這裡才是對時間做疊代
        outfolder = "data_"+self.outheader
        Path(outfolder).mkdir(parents=True, exist_ok=True)
        for n in range(nsteps):
            if (dt > tmax-time):
                dt = tmax - time           #避免dt太大
            
            particles = simulation_method(dt, particles,d)

            if (n % self.outfreq == 0):        #也就是在一定間隔內output一次數據 ex. outfreq=200-->200*dt做一次疊代
                if self.outscreen:
                    print("n", n, "Time", time, "dt:", dt )
                fn = self.outheader + "_" + str(n).zfill(7)+".dat"
                fn = outfolder+"/"+fn
                particles.output(fn)

            time+=dt
        
        print("Simulation finished.")
        return

if __name__ == "__main__":

    pass


        