# Fichier pour les tests à la con
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import landau


bins = np.arange(446, 43956800,43956)
courbe_est = landau.pdf(bins, loc=5, scale=5)
plt.plot(bins, courbe_est, 'k--', lw=2, label='Courbe estimée')
plt.show()
print(bins)
print(len(bins))