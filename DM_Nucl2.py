# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:00:27 2025

@author: Kamil
"""

import numpy as np
from scipy.stats import landau
import matplotlib.pyplot as plt

# Définir les paramètres de localisation et d'échelle
loc = 10  # Position
scale = 10  # Échelle

# Générer des valeurs x
x = np.linspace(-10,100, 10000)

# Calculer la fonction de densité de probabilité (PDF)
pdf_values = landau.pdf(x, loc=loc, scale=scale)



# Tracer la PDF
plt.plot(x, pdf_values, 'r-', lw=2, label='landau pdf')
plt.title('Fonction de densité de probabilité de la distribution de Landau')
plt.xlabel('x')
plt.ylabel('Densité de probabilité')
plt.legend()
plt.show()
