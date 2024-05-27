import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from matplotlib import cm
import numpy.linalg

def E(x,y,a,N,gamma0,gamma1,gamma2,gamma3,gamma4,Delta_1,Delta_2,delta):  # me estoy olvidando del 4*pi^2/N
    sigma = (3*a/2)*(-x+1j*y)
    sigma_conj = (3*a/2)*(-x-1j*y)
    delta = delta*10**(-3) # pasamos de meV a eV
    H = np.array([[delta, gamma2/2, 0, gamma0*sigma, gamma4*sigma, 0, 0, gamma3*sigma_conj, 0, 0],
                     [gamma2/2, 0, gamma1, 0, gamma3*sigma, gamma4*sigma, 0, gamma4*sigma_conj, gamma0*sigma_conj, 0],
                     [0, gamma1, 0, 0, 0, gamma0*sigma, gamma4*sigma, 0, gamma4*sigma_conj, gamma3*sigma_conj],
                     [gamma0*sigma_conj, 0, 0, 0, gamma1, 0, 0, gamma4*sigma, 0, 0],
                     [gamma4*sigma_conj, gamma3*sigma_conj, 0, gamma1, 0, gamma2/2, 0, gamma0*sigma, gamma4*sigma, 0],
                     [0, gamma4*sigma_conj, gamma0*sigma_conj, 0, gamma2/2, 0, gamma1, 0, gamma3*sigma, gamma4*sigma],
                     [0, 0, gamma4*sigma_conj, 0, 0, gamma1, 0, 0, 0, gamma0*sigma],
                     [gamma3*sigma, gamma4*sigma, 0, gamma4*sigma_conj, gamma0*sigma_conj, 0, 0, 0, gamma1, 0],
                     [0, gamma0*sigma, gamma4*sigma, 0, gamma4*sigma_conj, gamma3*sigma_conj, 0, gamma1, 0, gamma2/2],
                     [0, 0, gamma3*sigma, 0, 0, gamma4*sigma_conj, gamma0*sigma_conj, 0, gamma2/2, -delta]])
    return np.sort(np.linalg.eigvals(H).real) #Los autovalores son reales, pero como se calculan de forma numérica hay partes imaginarias despreciables, con .real nos quedamos solo con la parte real    

def graf_sim(gamma0,gamma1,gamma2,gamma3,gamma4,Delta_1,Delta_2,delta,a,N,width): #con width modificamos el ancho de la gráfica, está en porcentaje sobre la recta del camino en cuestión
    #notemos que estamos considerando un camino con el punto K en el centro
    #recta de (-4*pi/3*sqrt(3)*a,0) a (0,0)
    x1 = np.linspace((width/100)*(-4*np.pi/(3*np.sqrt(3)*a)),0,1000)
    y1 = np.zeros(1000)
    z1 = np.array([E(x[0],x[1],a,N,gamma0,gamma1,gamma2,gamma3,gamma4,Delta_1,Delta_2,delta) for x in zip(x1,y1)])

    #recta de (0,0) a (-2*pi/3*sqrt(3)*a,2*pi/3a)
    x2 = np.linspace(0,0+(width/100)*(-2*np.pi/(3*np.sqrt(3)*a)),1000)
    y2 = np.linspace(0,(width/100)*(2*np.pi/(3*a)),1000)
    theta = 0
    x2_r = np.cos(theta)*x2-np.sin(theta)*y2
    y2_r = np.sin(theta)*x2+np.cos(theta)*y2
    z2 = np.array([E(x[0],x[1],a,N,gamma0,gamma1,gamma2,gamma3,gamma4,Delta_1,Delta_2,delta) for x in zip(x2_r,y2_r)])

    #las unimos
    z = np.concatenate((z1,z2))
    x = np.concatenate((-np.sqrt(x1**2+y1**2),np.sqrt(x2**2+y2**2)))
    x = x*a
    #x = np.linspace(0,2,len(z[:,0]))
    return z,x

##Parámetros antiguos
#gamma0 = 3.1
#gamma1 = 0.54
#gamma2 = -0.015
#gamma3 = -0.29
#gamma4 = -0.141

##Parámetros nuevos
gamma0 = 3.16
gamma1 = 0.502
gamma2 = -0.0171
gamma3 = -0.377
gamma4 = -0.099

Delta_1 = 0
Delta_2 = 0
delta = 0 # en meV
a = 2.46*10**(-10)/np.sqrt(3)
N = 1000
z,x = graf_sim(gamma0,gamma1,gamma2,gamma3,gamma4,Delta_1,Delta_2,delta,a,N,6)
z = z*1000 #pasamos a meV
for i in range(len(z[0,:])):
    plt.plot(x, z[:,i], color='black')

plt.title("Pentacapa $\Gamma$ - K - M con $\Delta$V = {} meV".format(2*delta))
plt.axvline(x=1, ymin=0, ymax=1, color='black', linewidth=0.7)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plt.xlabel("distancia a K (1/a)")
plt.ylabel("E (meV)")
plt.ylim(-7,7)
plt.xlim(min(x)/2,max(x)/2)
    
plt.show()
