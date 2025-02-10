import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt

# Constants
I=173e-9 # mean exitation energy (GeV)
n=1e23 # electron density (e-/cm^-3)
epsilon_0 = cst.epsilon_0
e = cst.e * 1e-9 # GeV
m_e = 511e3 # electron mass in eV
c = cst.c * 1e2 # cm/s
z = 1. # charge of the particle
pi= np.pi
K= 0.307075*1e-3 # MeV mol^-1 cm^2 => GeV mol^-1 cm^2
delta=0. # density correction
C=0 # Correction term (C/Z)
R=0.5 # ratio of the atomic number to the atomic mass number (Z/A) (Silicium Z=14, A=28)

Cst_1=K*z**2*R 
Cst_2=2*m_e/I**2

# Bethe-Bloch formula
def bethe_bloch(beta, gamma):
    # Version non relativiste
    #Wmax= 2*m_e*beta**2*gamma**2/((1+2*gamma*m_e/M)+(m_e/M)**2) # maximum energy transfer elastic collision
    #return (Cst_1/beta**2)*(0.5*np.log(Cst_2*beta**2*gamma**2*Wmax)-beta**2-C-delta/2)
    return  (Cst_1/beta**2)*(np.log(2*m_e*beta**2*gamma**2/I)-beta**2-C-delta/2) # PID M>>2gamma*m_e (heavy_part_2 slide.47)

# Range of beta values
# beta = np.linspace(0.1, 0.9999999, 10000)
# gamma = 1 / np.sqrt(1 - beta**2)
# beta_gamma = beta * gamma
# # Calculate dE/dx
# Bethe_Bloch = bethe_bloch(beta,gamma)

# # Plot
# plt.plot(beta_gamma, Bethe_Bloch)
# plt.xlabel(r'$\beta$$\gamma$')
# plt.ylabel(r'$-\frac{dE}{dx}$ (MeV cm$^2$/g)')
# plt.title('Bethe-Bloch Formula')
# plt.xscale('log')
# plt.yscale('log')
# plt.grid(True)
# plt.show()
