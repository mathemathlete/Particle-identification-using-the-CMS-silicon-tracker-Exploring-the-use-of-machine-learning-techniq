# import Extraction_tab
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.constants as cst

# data = pd.DataFrame()
# Extraction_tab.import_data(data)

# Constants
I=1. # mean exitation energy 
n=1. # electron density
epsilon_0 = cst.epsilon_0
e = cst.e
m_e = 511e3 # electron mass in eV
c = cst.c 
z = 1. # charge of the particle
pi= np.pi
K= 0.307075 # MeV mol^-1 cm^2
delta=0. # density correction
C=0 # Correction term (C/Z)
R=0.5 # ratio of the atomic number to the atomic mass number (Z/A) (Silicium Z=14, A=28)

Cst_1=K*z**2*R
Cst_2=2*m_e/I**2

m_p = 938e-3# proton mass in GeV
m_deut = 1875e-3 # deuteron mass in GeV
m_pion = 139e-3 # pion mass in GeV
m_kaon = 493e-3 # kaon mass in GeV

# Bethe-Bloch formula
def bethe_bloch(mass, momentum):
    # Calculate the bethe bloch for proton, pion, kaon and Deuteron
    # We consider that we have p & m , thus we deduce Beta and gamma for each particle

    # Non relativistic version here
    #Wmax= 2*m_e*beta**2*gamma**2/((1+2*gamma*m_e/M)+(m_e/M)**2) # maximum energy transfer elastic collision
    #return (Cst_1/beta**2)*(0.5*np.log(Cst_2*beta**2*gamma**2*Wmax)-beta**2-C-delta/2)
    gamma = (1+(momentum/(mass))**2)**0.5
    beta = momentum/(mass*gamma)
    print(gamma)
    return  (Cst_1/beta**2)*(np.log(2*m_e*beta**2*gamma**2/I)-beta**2-C-delta/2) # PID M>>2gamma*m_e (heavy_part_2 slide.47)
    

pp = np.linspace(0.01, 100, 100000)  # Momentum in GeV
bethe_bloch_proton_ref = bethe_bloch(m_p, pp)  # Convert momentum to GeV/c
bethe_bloch_pion_ref = bethe_bloch(m_pion, pp)
bethe_bloch_kaon_ref = bethe_bloch(m_kaon, pp)
bethe_bloch_deuteron_ref = bethe_bloch(m_deut, pp)


plt.figure()
plt.plot(pp,bethe_bloch_pion_ref, 'r',label='dedx_Pion')
plt.plot(pp,bethe_bloch_proton_ref, 'k', label='dedx_Proton')
plt.plot(pp,bethe_bloch_kaon_ref,'g', label='dedx_Kaon')
plt.plot(pp,bethe_bloch_deuteron_ref,'b', label='dedx_Deuteron')
plt.xlabel('Momentum (GeV/c)')
plt.ylim(0,100)
plt.xscale('log')
plt.ylabel('dE/dx (MeV cm^2/g)')
plt.title('Bethe-Bloch Formula')
plt.legend()
plt.grid(True)
plt.show()  