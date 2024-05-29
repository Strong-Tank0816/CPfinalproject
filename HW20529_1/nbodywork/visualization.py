import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

AU = 149597870895

def load_files(header,pattern='[0-9][0-9][0-9][0-9][0-9][0-9]'):
    fns = 'data_' + header+'/'+header +'_'+ pattern+'.dat'
    fns = glob.glob(fns)
    fns.sort()
    return fns


def save_movie(fns, lengthscale=1.0, filename='movie.mp4',fps=30):

    plt.style.use('dark_background')
    scale = lengthscale

    fig, ax = plt.subplots()
    fig.set_linewidth(5)
    fig.set_size_inches(10, 10, forward=True)
    fig.set_dpi(72)
    
    planets = [ax.plot([], [], '.', color='w', markersize=2)[0] for i in range(9)]
    
    def init():
  
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_aspect('equal')
        ax.set_xlabel('X [AU]', fontsize=18)
        ax.set_ylabel('Y [AU]', fontsize=18)
        return planets

    def update(frame):
        fn = fns[frame]
        m,t,x,y,z,vx,vy,vz,ax,ay,az = np.loadtxt(fn)
        xAU = np.zeros_like(x)
        yAU = np.zeros_like(y)
        xAU = x/AU
        yAU = y/AU
        for i, planet in enumerate(planets):
            planet.set_data(xAU[i], yAU[i])
        plt.title(f"day {3*frame}", size=18)
        plt.grid(color = 'w')
        return planets
    
    ani = animation.FuncAnimation(fig, update, frames=len(fns), init_func=init, blit=True)
    ani.save(filename, writer='ffmpeg', fps=fps)
    return





def save_movie_inner(fns, lengthscale=1.0, filename='movie.mp4', fps=30):

    plt.style.use('dark_background')
    scale = lengthscale

    fig, ax = plt.subplots()
    fig.set_linewidth(5)
    fig.set_size_inches(10, 10, forward=True)
    fig.set_dpi(72)
    
    # 假設有三顆行星，分別給他們不同的顏色
    colors = ['w', 'g', 'y','b','r']
    name = ['Sun','Mercury','Venus','Earth','Mars']
    
    markersize = [100,10,20,25,13]
    planets = [ax.plot([], [], '.',label = name[i] , color=colors[i], markersize=markersize[i])[0] for i in range(5)]

    def init():
  
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_aspect('equal')
        ax.set_xlabel('X [AU]', fontsize=18)
        ax.set_ylabel('Y [AU]', fontsize=18)
        return planets

    def update(frame):
        fn = fns[frame]
        m,t,x,y,z,vx,vy,vz,ax,ay,az = np.loadtxt(fn)
        xAU = np.zeros_like(x)
        yAU = np.zeros_like(y)
        xAU = x/AU
        yAU = y/AU
        for i, planet in enumerate(planets):
            planet.set_data(xAU[i], yAU[i])
  
        plt.legend(ncols = 1,frameon = True,markerscale = 0.4,fontsize = 15)
        plt.title(f"day {3*frame}", size=18)
        plt.grid(color = 'w')
        return planets

    ani = animation.FuncAnimation(fig, update, frames=len(fns), init_func=init, blit=True)
    ani.save(filename, writer='ffmpeg', fps=fps)
    return

def save_movie_outer(fns, lengthscale=1.0, filename='movie.mp4',fps=30):

    plt.style.use('dark_background')
    scale = lengthscale

    fig, ax = plt.subplots()
    fig.set_linewidth(5)
    fig.set_size_inches(10, 10, forward=True)
    fig.set_dpi(72)
    colors = ['w', 'g', 'y','b','r']
    markersize = [100,50,40,30,30]
    name = ['Sun','Jupyter','Saturn','Uranus','Neptune']
    planets = [ax.plot([], [], '.',label = name[i] , color=colors[i], markersize=markersize[i])[0] for i in range(5)]

    def init():
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_aspect('equal')
        ax.set_xlabel('X [AU]', fontsize=18)
        ax.set_ylabel('Y [AU]', fontsize=18)

        return planets 

    def update(frame):
        fn = fns[frame]
        m,t,x,y,z,vx,vy,vz,ax,ay,az = np.loadtxt(fn)
        xAU = np.zeros_like(x)
        yAU = np.zeros_like(y)
        xAU = x/AU
        yAU = y/AU
        for i, planet in enumerate(planets):
            planet.set_data(xAU[i], yAU[i])

        plt.legend(ncols = 1,frameon = True,markerscale = 0.4,fontsize = 15)
        plt.title(f"day {3*frame}", size=18)
        plt.grid(color = 'w')
        return planets

    ani = animation.FuncAnimation(fig, update, frames=len(fns), init_func=init, blit=True)
    ani.save(filename, writer='ffmpeg', fps=fps)
    return