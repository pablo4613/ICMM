import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from matplotlib import cm
import numpy.linalg


#ESTA PRIMERA SECCIÓN ESTABLECE LOS VALORES DE LOS PARÁMETROS Y LA FUNCIÓN DE LA ENERGÍA

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


#ESTA SECCIÓN ES PARA CARGAR LAS REGIONES QUE NOS INTERESAN EN EL ESPACIO RECÍPROCO Y SUS ENERGÍAS

xy = []   #vamos a guardar aquí puntos en un hexágono (preservamos así la simetría) en un entorno cercano al K
d = 6*10**8 #distancia que queremos considerar entre el centro del hexágono (punto K) y el vértice en el eje x del hexágono
grid = 800
for i in np.arange(-d, -d/2, d/grid):
    for j in np.arange(-np.sqrt(3)*(i+d), np.sqrt(3)*(i+d), d/grid):
        xy.append([i,j])
for i in np.arange(-d/2, d/2, d/grid):
    for j in np.arange(-np.sqrt(3)*d/2, np.sqrt(3)*d/2, d/grid):
        xy.append([i,j])
for i in np.arange(d/2, d, d/grid):
    for j in np.arange(np.sqrt(3)*(i-d), -np.sqrt(3)*(i-d), d/grid):
        xy.append([i,j])
deltax_xy = d/grid

puntos_y_energias = {}   #guardaremos aquí los puntos anteriores (como llaves) y sus energías (como valores)
for point in xy:
    puntos_y_energias[str(point)] = E(point[0],point[1],a,N,gamma0,gamma1,gamma2,gamma3,gamma4,Delta_1,Delta_2,delta)

Energias = []
for energia in puntos_y_energias.values():
    Energias.append(energia)
Energias = np.sort(np.concatenate(Energias))
print('Numero energias {}'.format(len(Energias)))
print('Mínimo y máximo de energias {} {}'.format(min(Energias),max(Energias)))

Energias_antiguas = Energias
plt.plot(range(len(Energias)),Energias)
Energias = []
for energia in Energias_antiguas:
    if abs(energia) < 0.2:
        Energias.append(energia)
Energias = np.sort(Energias)
print('Numero energias acotadas {}'.format(len(Energias)))
print('Mínimo y máximo de las energias acotadas {} {}'.format(min(Energias),max(Energias)))

BZ = []   #vamos a guardar aquí todos los puntos en la PZB . IMPORTANTE: AHORA EL (0,0) SÍ COINCIDE CON EL (0,0), NO CON K.
d = 4*np.pi/(3*np.sqrt(3)*a) #distancia que queremos considerar entre el centro del hexágono (punto K) y el vértice en el eje x del hexágono
for i in np.arange(-d, -d/2, d/grid):
    for j in np.arange(-np.sqrt(3)*(i+d), np.sqrt(3)*(i+d), d/grid):
        BZ.append([i,j])
for i in np.arange(-d/2, d/2, d/grid):
    for j in np.arange(-np.sqrt(3)*d/2, np.sqrt(3)*d/2, d/grid):
        BZ.append([i,j])
for i in np.arange(d/2, d, d/grid):
    for j in np.arange(np.sqrt(3)*(i-d), -np.sqrt(3)*(i-d), d/grid):
        BZ.append([i,j])
deltax_PBZ = d/grid
lado_hexagono = d

puntos_y_energias = {}   #guardaremos aquí los puntos anteriores (como llaves) y sus energías (como valores)
for point in BZ:
    puntos_y_energias[str(point)] = E_total(point[0],point[1],a,N,gamma0,gamma1,gamma2,gamma3,gamma4,Delta_1,Delta_2,delta)

Energias_BZ = []
for energia in puntos_y_energias.values():
    Energias_BZ.append(energia)
Energias_BZ = np.sort(np.concatenate(Energias_BZ))
print('Numero energias BZ {}'.format(len(Energias_BZ)))
print('Mínimo y máximo de energias BZ {} {}'.format(min(Energias_BZ),max(Energias_BZ)))


#AQUÍ FINALMENTE CALCULAMOS Y REPRESENTAMOS LA DENSIDAD DE ESTADOS PARA LA ENERGÍA DE FERMI DADO UN LLENADO

def calculo_Efermi(Energias,lado_hexagono,deltax,den,num):  #den y num son el denominador y numerador del llenado
    num_pixeles = (3*np.sqrt(3)*lado_hexagono**2)/(2*deltax**2)
    indice = int((num/(2*den))*(1/2)*10*num_pixeles+indice_llenadomitad)
    Efermi = Energias[indice] #hay que sumar num+den porque ese es el numerador real (recordemos que el 0 corresponde a 1e/at
    return Efermi
    
def plot_densidad_frente_llenado(Energia_llenadomitad,Energias,a,N,gamma0,gamma1,gamma2,gamma3,gamma4,Delta_1,Delta_2,delta,x,N_Energias,PZB,xy):
    print('Energias en el entorno de K {}'.format(Energias))
    Energias_fermi = list(map(lambda point:calculo_Efermi(Energias,lado_hexagono,deltax_xy,x['den'],point), x['num']))
    print('Energias de Fermi mínima y máxima: {} ; {}'.format(min(Energias_fermi),max(Energias_fermi)))
    print('Número energías de Fermi: {}'.format(len(Energias_fermi)))
    dE = (max(Energias_fermi)-min(Energias_fermi))/(2*(N_Energias-1))
    Intervalos_de_energia = [min(Energias_fermi)+n*2*dE for n in range(0,N_Energias,1)]

    Densidad = [0]*N_Energias
    for energia in Energias_fermi:
        for i in range(len(Intervalos_de_energia)):
            if energia>Intervalos_de_energia[i]-dE and energia<Intervalos_de_energia[i]+dE:
                Densidad[i]+=1
    print('densidad en el 0 {}'.format(Densidad[len(Densidad)//2]))

    Densidad_cada_energia = [0]*len(Energias_fermi)
    for i in range(len(Energias_fermi)):
        for j in range(len(Intervalos_de_energia)):
            if Energias_fermi[i]>Intervalos_de_energia[j]-dE and Energias_fermi[i]<Intervalos_de_energia[j]+dE:
                Densidad_cada_energia[i] = Densidad[j]
    
    ax = fig.add_subplot(1,1,1)  
    x = [num/x['den'] for num in x['num']]
    x_cm = []
    for i in x:
        x_cm.append(5*i*4/(3*np.sqrt(3)*(a*100)**2)) # si ponemos un factor 3/5 da igual que de la forma anterior
    print('Mínimo y máximo de x_cm: {} ; {}'.format(min(x_cm),max(x_cm)))

    maximo = 0
    for i in range(len(Densidad_cada_energia)):
        if Densidad_cada_energia[i] > maximo:
            maximo = Densidad_cada_energia[i]
            index = i
    maximo_2 = 0
    for i in range(len(Densidad_cada_energia)):
        if Densidad_cada_energia[i] > maximo_2 and abs(x_cm[i]-x_cm[index])>0.5*10**12:
            maximo_2 = Densidad_cada_energia[i]
            index_2 = i

    maximo_3 = 0
    for i in range(len(Densidad_cada_energia)):
        if Densidad_cada_energia[i] > maximo_3 and x_cm[i]>-4*10**12 and x_cm[i]<-3*10**12:
            maximo_3 = Densidad_cada_energia[i]
            index_3 = i

    Van_Hove = x_cm[index]
    print('El Van Hove se da a llenado {} e- / cm^2'.format(Van_Hove))
    Van_Hove_2 = x_cm[index_2]
    print('El Van Hove secundario se da a llenado {} e- / cm^2'.format(Van_Hove_2))
    Van_Hove_3 = x_cm[index_3]
    print('El Van Hove terciario se da a llenado {} e- / cm^2'.format(Van_Hove_3))

    minimo = 10**5
    for i in range(len(Densidad_cada_energia)):
        if Densidad_cada_energia[i] < minimo and x_cm[i]>Van_Hove and x_cm[i]<Van_Hove_2:
            minimo = Densidad_cada_energia[i]
            index_4 = i

    #Minimo = x_cm[index_4]
    #print('El minimo entre Van Hoves se da a llenado {} e- / cm^2'.format(Minimo))
    
    ax.plot(x_cm,Densidad_cada_energia)
    plt.xlim(min(x_cm),max(x_cm))
    plt.xticks(np.arange(-3, 4, 1)*10**12)
    for point in np.arange(-2, 1.5, 0.1)*10**12:
        plt.axvline(x=point, ymin=0, ymax=1, color='black', linewidth=0.7)
    plt.title('DOS para $\Delta$V = {} meV'.format(2*delta))
    return Van_Hove
    
N = 500
N_Energias = 3000
x = {'den':10**8, 'num':list(range(-2*10**4,2*10**4))}  #llenado medido en e- por átomo con el 0 en el llenado mitad, es decir, el 0 corresponde a 1e- por át.
# hemos cogido estos números para den y num ya que 2*10^(12) e-/cm^2 (límites en el artículo) se corresponden con 3.5*10^(-4) e-/at

fig = plt.figure(figsize=(12, 5))

plot_densidad_frente_llenado(Energia_llenadomitad,Energias,a,N,gamma0,gamma1,gamma2,gamma3,gamma4,Delta_1,Delta_2,delta,x,N_Energias,BZ,xy)

axs = fig.axes
for ax in axs:
    ax.set(xlabel='e- / cm^2', ylabel='Densidad')
    ax.label_outer()

plt.show()
