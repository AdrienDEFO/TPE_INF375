

import numpy as np
import scipy.optimize as opt
import scipy.linalg as la
import scipy.integrate as integrate

# 1. DICHOTOMIE
def dichotomie(fonction, borne_inf, borne_sup, seuil=1e-2):
    return opt.bisect(fonction, borne_inf, borne_sup, xtol=seuil)

# 2. BALAYAGE -

def balayage(fonction, borne_inf, borne_sup, n_points=1000):
   
    # Crée des points régulièrement espacés dans l'intervalle
    x_points = np.linspace(borne_inf, borne_sup, n_points)
    # Évalue la fonction sur tous les points
    y_points = fonction(x_points)
    # Trouve l'indice où la fonction change de signe
    idx_changement = np.where(np.diff(np.sign(y_points)))[0][0]
    # Retourne le milieu du segment où le changement se produit
    return (x_points[idx_changement] + x_points[idx_changement + 1]) / 2

# 3. LAGRANGE 

def lagrange(fonction, borne_inf, borne_sup, seuil=1e-6):
   
    return opt.brentq(fonction, borne_inf, borne_sup, xtol=seuil)

# 4. NEWTON-RAPHSON 

def newton_raphson(fonction, derivee, point_initial, seuil=1e-6):
    
    return opt.newton(fonction, point_initial, fprime=derivee, tol=seuil)

# 5. PIVOT DE GAUSS 

def pivot_gauss(matrice, vecteur):
   
    return np.linalg.solve(matrice, vecteur)

# 6. GAUSS-JORDAN 
def gauss_jordan(matrice, vecteur):
   
    return la.solve(matrice, vecteur)

# 7. CROUT 
def methode_crout(matrice, vecteur):
    
    # Factorise la matrice A en matrices L et U
    decomposition_lu = la.lu_factor(matrice)
    # Résout le système en utilisant la décomposition LU
    return la.lu_solve(decomposition_lu, vecteur)



