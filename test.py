

import numpy as np
from numerical import (
    balayage, dichotomie, lagrange, newton_raphson, newton_cote,
    pivot_gauss, gauss_jordan, methode_crout
)
#decommenter le bloc ci-dessous pour implémenter les methodes existantes
"""from Existante import (
    balayage, dichotomie, lagrange, newton_raphson, newton_cote,
    pivot_gauss, gauss_jordan, methode_crout
)""" 
def main():
    """Fonction principale """
    
    print("\n TPE INF375 ")
    print("\n Choisissez la méthode à utiliser :")
    print("\n 1. Méthode de balayage")
    print("\n 2. Méthode de dichotomie") 
    print("\n 3. Méthode de Lagrange")
    print("\n 4. Méthode de Newton-Raphson")
    print("\n 5. Méthode de Newton-Côte (intégration)")
    print("\n 6. Pivot de Gauss (systèmes linéaires)")
    print("\n 7. Gauss-Jordan (systèmes linéaires)")
    print("\n 8. Méthode de Crout (systèmes linéaires)")
    
    choix = input("\nEntrez votre choix (1-8): ")
    
    if choix == "1":
        print("\n MÉTHODE DE BALAYAGE ")
        f1 = input("Entrez la fonction f(x) exemple: (x**2 - 2*x + 1)\n")
        a = float(input("Entrez la borne inférieure: "))
        b = float(input("Entrez la borne supérieure: "))
        pas = float(input("Entrez le pas de balayage: "))
        # L'utilisateur entre le maximum d'itérations (valeur par défaut: 10000)
        max_iter_input = input("Entrez le maximum d'itérations (défaut: 10000): ")
        max_iterations = int(max_iter_input) if max_iter_input.strip() != "" else 10000
        
        try:
            # Appel de la méthode de balayage avec affichage détaillé
            resultat = balayage(f1, a, b, pas, verbose=True)
            if resultat is not None:
                print(f"\nRésultat: La solution approximative est x = {resultat}")
                # Vérification du résultat en évaluant la fonction
                print(f"Vérification: f({resultat}) = {eval(f1, {'x': resultat})}")
            else:
                print("Aucune solution trouvée dans l'intervalle donné.")
        except Exception as e:
            print(f"Erreur: {e}")
    
    elif choix == "2":
        print("\n MÉTHODE DE DICHOTOMIE ")
        f1 = input("Entrez la fonction f(x) exemple: (x**2 - 2*x + 1)\n")
        a = float(input("Entrez la borne inférieure: "))
        b = float(input("Entrez la borne supérieure: "))
        seuil = float(input("Entrez le seuil de précision: "))
        # L'utilisateur entre le maximum d'itérations (valeur par défaut: 10000)
        max_iter_input = input("Entrez le maximum d'itérations (défaut: 10000): ")
        max_iterations = int(max_iter_input) if max_iter_input.strip() != "" else 10000
        
        try:
            # Appel de la méthode de dichotomie avec affichage détaillé
            resultat = dichotomie(f1, a, b, seuil, verbose=True)
            if resultat is not None:
                print(f"\nRésultat: La solution approximative est x = {resultat}")
                print(f"Vérification: f({resultat}) = {eval(f1, {'x': resultat})}")
            else:
                print("Aucune solution trouvée dans l'intervalle donné.")
        except Exception as e:
            print(f"Erreur: {e}")
    
    elif choix == "3":
        print("\n MÉTHODE DE LAGRANGE ")
        f1 = input("Entrez la fonction f(x) exemple: (x**2 - 2*x + 1)\n")
        a = float(input("Entrez la borne inférieure: "))
        b = float(input("Entrez la borne supérieure: "))
        seuil = float(input("Entrez le seuil de précision: "))
        # L'utilisateur entre le maximum d'itérations (valeur par défaut: 10000)
        max_iter_input = input("Entrez le maximum d'itérations (défaut: 10000): ")
        max_iterations = int(max_iter_input) if max_iter_input.strip() != "" else 10000
        
        try:
            # Appel de la méthode de Lagrange avec le maximum d'itérations spécifié
            resultat = lagrange(f1, a, b, seuil, max_iterations)
            print(f"\nRésultat: La solution approximative est x = {resultat}")
            print(f"Vérification: f({resultat}) = {eval(f1, {'x': resultat})}")
        except Exception as e:
            print(f"Erreur: {e}")
    
    elif choix == "4":
        print("\n MÉTHODE DE NEWTON-RAPHSON")
        f1 = input("Entrez la fonction f(x) exemple: (x**2 - 2*x + 1)\n")
        derivee = input("Entrez la dérivée f'(x) exemple: (2*x - 2)\n")
        x0 = float(input("Entrez la valeur initiale x0: "))
        seuil = float(input("Entrez le seuil de précision: "))
        # L'utilisateur entre le maximum d'itérations (valeur par défaut: 10000)
        max_iter_input = input("Entrez le maximum d'itérations (défaut: 10000): ")
        max_iterations = int(max_iter_input) if max_iter_input.strip() != "" else 10000
        
        try:
            # Appel de la méthode de Newton-Raphson avec le maximum d'itérations spécifié
            resultat = newton_raphson(f1, derivee, x0, seuil, max_iterations)
            print(f"\nRésultat: La solution approximative est x = {resultat}")
            print(f"Vérification: f({resultat}) = {eval(f1, {'x': resultat})}")
        except Exception as e:
            print(f"Erreur: {e}")
    
    elif choix == "5":
        print("\n MÉTHODE DE NEWTON-CÔTE (INTÉGRATION)")
        f1 = input("Entrez la fonction f(x) à intégrer exemple: (x**2 + 1)\n")
        a = float(input("Entrez la borne inférieure d'intégration: "))
        b = float(input("Entrez la borne supérieure d'intégration: "))
        n = int(input("Entrez le degré de la méthode (2: trapèzes, 3: Simpson, etc.): "))
        
        try:
            # La méthode de Newton-Côte ne nécessite pas d'itérations maximum
            resultat = newton_cote(f1, a, b, n)
            # Calcul de la valeur exacte pour les fonctions polynomiales simples
            if "x**2" in f1 and "+ 1" in f1:
                valeur_exacte = (b**3/3 + b) - (a**3/3 + a)
                print(f"\nRésultat: ∫[{a},{b}] {f1} dx ≈ {resultat}")
                print(f"Valeur exacte pour x² + 1: {valeur_exacte}")
                print(f"Erreur: {abs(resultat - valeur_exacte)}")
            else:
                print(f"\nRésultat: ∫[{a},{b}] {f1} dx ≈ {resultat}")
        except Exception as e:
            print(f"Erreur: {e}")
    
    elif choix == "6":
        print("\n PIVOT DE GAUSS")
        print("Entrez la matrice A (système AX = B):")
        n = int(input("Entrez la taille du système (n): "))
        
        print("Entrez les coefficients de la matrice A ligne par ligne:")
        A = []
        for i in range(n):
            ligne = input(f"Ligne {i+1} (séparer les coefficients par des espaces): ").split()
            A.append([float(x) for x in ligne])
        
        print("\nEntrez le vecteur B:")
        B = []
        for i in range(n):
            b_val = float(input(f"B[{i+1}]: "))
            B.append(b_val)
        
        try:
            A_np = np.array(A)
            B_np = np.array(B)
            resultat = pivot_gauss(A_np, B_np)
            print(f"\nSolution du système:")
            for i in range(n):
                print(f"x[{i+1}] = {resultat[i]}")
        except Exception as e:
            print(f"Erreur: {e}")
    
    elif choix == "7":
        print("\n GAUSS-JORDAN ")
        print("Entrez la matrice A (système AX = B):")
        n = int(input("Entrez la taille du système (n): "))
        
        print("Entrez les coefficients de la matrice A ligne par ligne:")
        A = []
        for i in range(n):
            ligne = input(f"Ligne {i+1} (séparer les coefficients par des espaces): ").split()
            A.append([float(x) for x in ligne])
        
        print("\nEntrez le vecteur B:")
        B = []
        for i in range(n):
            b_val = float(input(f"B[{i+1}]: "))
            B.append(b_val)
        
        try:
            A_np = np.array(A)
            B_np = np.array(B)
            resultat = gauss_jordan(A_np, B_np)
            print(f"\nSolution du système:")
            for i in range(n):
                print(f"x[{i+1}] = {resultat[i]}")
        except Exception as e:
            print(f"Erreur: {e}")
    
    elif choix == "8":
        print("\n MÉTHODE DE CROUT ")
        print("Entrez la matrice A (système AX = B):")
        n = int(input("Entrez la taille du système (n): "))
        
        print("Entrez les coefficients de la matrice A ligne par ligne:")
        A = []
        for i in range(n):
            ligne = input(f"Ligne {i+1} (séparer les coefficients par des espaces): ").split()
            A.append([float(x) for x in ligne])
        
        print("\nEntrez le vecteur B:")
        B = []
        for i in range(n):
            b_val = float(input(f"B[{i+1}]: "))
            B.append(b_val)
        
        try:
            A_np = np.array(A)
            B_np = np.array(B)
            resultat = methode_crout(A_np, B_np)
            print(f"\nSolution du système:")
            for i in range(n):
                print(f"x[{i+1}] = {resultat[i]}")
        except Exception as e:
            print(f"Erreur: {e}")
    
    else:
        print("Choix invalide!")

if __name__ == "__main__":
    main()