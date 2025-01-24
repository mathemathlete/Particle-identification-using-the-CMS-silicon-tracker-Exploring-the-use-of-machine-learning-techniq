# import Extraction_tab
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.constants as cst
from sympy import symbols, solve

# Les Seuils sont définis de manière arbitraire , à revoir éventuellement

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

seuil_acceptance_proton=200e-3
seuil_acceptance_deut=300e-3
seuil_acceptance_pion=40e-3
seuil_acceptance_kaon=100e-3

# Bethe-Bloch formula
def bethe_bloch(mass, momentum):
    # Calculate the bethe bloch for proton, pion, kaon and Deuteron
    # We consider that we have p & m , thus we deduce Beta and gamma for each particle

    # Non relativistic version here
    #Wmax= 2*m_e*beta**2*gamma**2/((1+2*gamma*m_e/M)+(m_e/M)**2) # maximum energy transfer elastic collision
    #return (Cst_1/beta**2)*(0.5*np.log(Cst_2*beta**2*gamma**2*Wmax)-beta**2-C-delta/2)
    gamma = (1+(momentum/(mass))**2)**0.5
    beta = momentum/(mass*gamma)
    return  (Cst_1/beta**2)*(np.log(2*m_e*beta**2*gamma**2/I)-beta**2-C-delta/2) # PID M>>2gamma*m_e (heavy_part_2 slide.47)
    

def identification_part(p_dedx):
    # Identification of the particle
    # We consider that we have the dedx & p of the track
    p, dedx = p_dedx
    # Identification of a pion
    if (bethe_bloch(m_pion-seuil_acceptance_pion,p)-dedx) <0 and (bethe_bloch(m_pion+seuil_acceptance_pion,p)-dedx) >0:
        return m_pion
        #print("Pion")
    # Identification of a proton
    elif (bethe_bloch(m_p-seuil_acceptance_proton,p)-dedx) <0 and (bethe_bloch(m_p+seuil_acceptance_proton,p)-dedx) >0:
        return m_p
        #print("Proton")
    # Identification of a kaon
    elif (bethe_bloch(m_kaon-seuil_acceptance_kaon,p)-dedx) <0 and (bethe_bloch(m_kaon+seuil_acceptance_kaon,p)-dedx) >0:
        return m_kaon
        #print("Kaon")
    # Identification of a deuteron
    elif (bethe_bloch(m_deut-seuil_acceptance_deut,p)-dedx) <0 and (bethe_bloch(m_deut+seuil_acceptance_deut,p)-dedx) >0:
        return m_deut
        #print("Deuteron")
    else:
        print("Particle not identified")


def affichage_Bethe_Bloch_borneinfsup():
    ## Partie affichage / Verif
    pp = np.linspace(0.01, 100, 20000)  # Momentum in GeV
    bethe_bloch_proton_ref = bethe_bloch(m_p, pp)  # Convert momentum to GeV/c
    bethe_bloch_pion_ref = bethe_bloch(m_pion, pp)
    bethe_bloch_kaon_ref = bethe_bloch(m_kaon, pp)
    bethe_bloch_deuteron_ref = bethe_bloch(m_deut, pp)


    #Identifications avec moyenne entre les 2
    # bethe_bloch_lim_deuteron_proton = (bethe_bloch_deuteron_ref + bethe_bloch_proton_ref) /2
    # bethe_bloch_lim_proton_kaon = (bethe_bloch_proton_ref + bethe_bloch_kaon_ref) /2
    # bethe_bloch_lim_pion_kaon = (bethe_bloch_pion_ref + bethe_bloch_kaon_ref) /2

    # Partie pour identifier les particules avec acceptances
    # Si on veut identifier les particules en termes de précision , (m_pion +- acceptance)
    proton_inf = bethe_bloch(m_p-seuil_acceptance_proton, pp)  
    proton_sup = bethe_bloch(m_p+seuil_acceptance_proton, pp)  

    pion_inf = bethe_bloch(m_pion-seuil_acceptance_pion, pp)
    pion_sup = bethe_bloch(m_pion+seuil_acceptance_pion, pp)

    kaon_inf = bethe_bloch(m_kaon-seuil_acceptance_kaon, pp)
    kaon_sup = bethe_bloch(m_kaon+seuil_acceptance_kaon, pp)

    deuteron_inf = bethe_bloch(m_deut-seuil_acceptance_deut, pp)
    deuteron_sup = bethe_bloch(m_deut+seuil_acceptance_deut, pp)

    plt.figure()

    # plt.plot(pp,bethe_bloch_lim_deuteron_proton, 'r--',label='dedx_Deuteron+Proton/2')
    # plt.plot(pp,bethe_bloch_lim_proton_kaon, 'g--',label='dedx_Proton+Kaon/2')
    # plt.plot(pp,bethe_bloch_lim_pion_kaon, 'b--',label='dedx_Pion+Kaon/2')

    plt.plot(pp,bethe_bloch_proton_ref, 'k', label='dedx_Proton')
    plt.plot(pp,bethe_bloch_pion_ref, 'r',label='dedx_Pion')
    plt.plot(pp,bethe_bloch_kaon_ref,'g', label='dedx_Kaon')
    plt.plot(pp,bethe_bloch_deuteron_ref,'b', label='dedx_Deuteron')

    plt.plot(pp,proton_inf, 'k:', label='Proton_inf')
    plt.plot(pp,proton_sup, 'r:', label='Proton_sup')
    plt.plot(pp,pion_inf, 'k:', label='Pion_inf')
    plt.plot(pp,pion_sup, 'r:', label='Pion_sup')
    plt.plot(pp,kaon_inf, 'k:', label='Kaon_inf')
    plt.plot(pp,kaon_sup, 'r:', label='Kaon_sup')
    plt.plot(pp,deuteron_inf, 'k:', label='Deuteron_inf')
    plt.plot(pp,deuteron_sup, 'r:', label='Deuteron_sup')
    plt.xlabel('Momentum (GeV/c)')
    plt.ylim(0,100)
    plt.xscale('log')
    plt.ylabel('dE/dx (MeV cm^2/g)')
    plt.title('Bethe-Bloch Formula')
    plt.legend()
    plt.grid(True)
    plt.show()  

## Partie Verification des données
# affichage_Bethe_Bloch_borneinfsup()

Particle_id_test = [[0.018,60],[0.019,96.9],[0.06,68.4],[0.18,17],[0.11,88.2],[0.43,12.2],[0.21,95.1],[0.71,16.2]]

for i in Particle_id_test:
    identification_part(i)
