import numpy as np
import logging
from typing import Optional, Tuple, Union, List
import numbers

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ErreurNumerique(Exception):
    """Exception personnalisée pour les erreurs numériques"""
    pass

class SolutionSysteme:
    """Classe pour représenter la solution d'un système linéaire"""
    
    def __init__(self, type_solution: str, solution_particuliere: Optional[np.ndarray] = None,
                 espace_noyau: Optional[List[np.ndarray]] = None):
        self.type_solution = type_solution  # 'unique', 'infinite', 'none'
        self.solution_particuliere = solution_particuliere
        self.espace_noyau = espace_noyau  # Base du noyau pour les solutions infinies
    
    def __str__(self):
        if self.type_solution == 'unique':
            return f"Solution unique: {self.solution_particuliere}"
        elif self.type_solution == 'infinite':
            if self.espace_noyau:
                base_str = " + ".join([f"t{i}*{vec}" for i, vec in enumerate(self.espace_noyau)])
                return f"Solutions infinies: {self.solution_particuliere} + [{base_str}]"
            else:
                return f"Solution particulière: {self.solution_particuliere}"
        else:
            return "Aucune solution"

def _valider_type(variable, types_attendus, nom_variable):
    """Valide le type d'une variable"""
    if not isinstance(variable, types_attendus):
        raise ErreurNumerique(
            f"{nom_variable} doit être de type {types_attendus}, "
            f"mais est de type {type(variable)}"
        )

def _valider_matrice_vecteur(matrice, vecteur):
    """Valide la compatibilité d'une matrice et d'un vecteur"""
    _valider_type(matrice, (np.ndarray, list), "La matrice")
    _valider_type(vecteur, (np.ndarray, list), "Le vecteur")
    
    if isinstance(matrice, list):
        matrice = np.array(matrice, dtype=float)
    if isinstance(vecteur, list):
        vecteur = np.array(vecteur, dtype=float)
    
    if vecteur.ndim != 1:
        raise ErreurNumerique("Le vecteur doit être de dimension 1")
    
    n = len(vecteur)
    if matrice.shape[0] != n:
        raise ErreurNumerique(
            f"Dimensions incompatibles: matrice {matrice.shape} "
            f"mais vecteur de taille {n}"
        )
    
    return matrice, vecteur

def balayage(fonction: str, borne_inf: float, borne_sup: float, pas: float, verbose: bool = False) -> Optional[float]:
    """
    
    
    Args:
        fonction: Expression mathématique de la fonction sous forme de chaîne
        borne_inf: Borne inférieure de l'intervalle de recherche
        borne_sup: Borne supérieure de l'intervalle de recherche  
        pas: Pas de déplacement pour le balayage
        verbose: Si True, affiche les détails des calculs
    
    Returns:
        float or None: Approximation de la racine ou None si non trouvée
        
    Raises:
        ErreurNumerique: Si les paramètres sont invalides
    """
    if verbose:
        logging.info(f"Début méthode balayage: f(x)={fonction}, intervalle=[{borne_inf}, {borne_sup}], pas={pas}")
    
    def f(x: float) -> float:
        """Évalue la fonction au point x en utilisant eval()"""
        try:
            resultat = eval(fonction)
            if verbose:
                logging.debug(f"f({x}) = {resultat}")
            return resultat
        except Exception as e:
            raise ErreurNumerique(f"Erreur lors de l'évaluation de f({x}): {e}")
    
    # Vérification de la validité des paramètres d'entrée
    if borne_inf >= borne_sup:
        raise ErreurNumerique("La borne inférieure doit être strictement inférieure à la borne supérieure")
    
    if pas <= 0:
        raise ErreurNumerique("Le pas de balayage doit être strictement positif")
    
    # Évaluation de la fonction aux bornes pour vérifier le changement de signe
    f_a, f_b = f(borne_inf), f(borne_sup)
    if verbose:
        logging.info(f"f({borne_inf})={f_a}, f({borne_sup})={f_b}")
    
    # Vérification de la fonction doit changer de signe sur l'intervalle
    if f_a * f_b > 0:
        msg = "Pas de changement de signe détecté - la méthode de balayage ne peut pas être appliquée"
        logging.warning(msg)
        return None
    
    # Début du balayage systématique de l'intervalle
    x_courant = borne_inf
    iteration = 0
    
    # Parcours de l'intervalle avec le pas spécifié
    while x_courant <= borne_sup - pas:
        iteration += 1
        # Évaluation de la fonction aux points x et x+pas
        f_x1, f_x2 = f(x_courant), f(x_courant + pas)
        
        if verbose:
            logging.info(f"Itération {iteration}: x={x_courant}, f(x)={f_x1}, f(x+pas)={f_x2}")
        
        # Détection du changement de signe: la racine se trouve entre x et x+pas
        if f_x1 * f_x2 <= 0:
            # La racine est approximée au milieu du segment où le changement de signe se produit
            solution = (x_courant + (x_courant + pas)) / 2
            if verbose:
                logging.info(f"Solution trouvée: {solution} après {iteration} itérations")
            return solution
        
        # Avancement au point suivant
        x_courant += pas
    
    # Si on arrive ici, aucune racine n'a été détectée
    if verbose:
        logging.warning("Aucune solution trouvée dans l'intervalle donné avec le pas spécifié")
    return None

def dichotomie(fonction: str, borne_inf: float, borne_sup: float, seuil: float, max_iterations: int = 10000, verbose: bool = False) -> Optional[float]:
    """
    
    Args:
        fonction: Expression mathématique de la fonction sous forme de chaîne
        borne_inf: Borne inférieure de l'intervalle de recherche
        borne_sup: Borne supérieure de l'intervalle de recherche
        seuil: Seuil de précision pour la convergence (erreur maximale acceptée)
        max_iterations: Nombre maximum d'itérations autorisées
        verbose: Si True, affiche les détails des calculs
    
    Returns:
        float or None: Approximation de la racine ou None si non trouvée
    """
    if verbose:
        logging.info(f"Début méthode dichotomie: f(x)={fonction}, intervalle=[{borne_inf}, {borne_sup}], seuil={seuil}, max_iterations={max_iterations}")
    
    def f(x: float) -> float:
        """Évalue la fonction au point x en utilisant eval()"""
        try:
            resultat = eval(fonction)
            if verbose:
                logging.debug(f"f({x}) = {resultat}")
            return resultat
        except Exception as e:
            raise ErreurNumerique(f"Erreur lors de l'évaluation de f({x}): {e}")
    
    # Validation des paramètres d'entrée
    if borne_inf >= borne_sup:
        raise ErreurNumerique("La borne inférieure doit être strictement inférieure à la borne supérieure")
    
    if seuil <= 0:
        raise ErreurNumerique("Le seuil de précision doit être strictement positif")
    
    if max_iterations <= 0:
        raise ErreurNumerique("Le nombre maximum d'itérations doit être strictement positif")
    
    # Évaluation initiale aux bornes
    f_a, f_b = f(borne_inf), f(borne_sup)
    if verbose:
        logging.info(f"f({borne_inf})={f_a}, f({borne_sup})={f_b}")
    
    # Vérification du changement de signe (condition nécessaire)
    if f_a * f_b > 0:
        msg = "Aucun changement de signe détecté - la méthode de dichotomie ne peut pas être appliquée"
        logging.warning(msg)
        return None
    
    # Vérification si une des bornes est déjà une solution acceptable
    if abs(f_a) < seuil:
        if verbose:
            logging.info(f"Solution exacte trouvée à la borne inférieure: {borne_inf}")
        return borne_inf
    
    if abs(f_b) < seuil:
        if verbose:
            logging.info(f"Solution exacte trouvée à la borne supérieure: {borne_sup}")
        return borne_sup
    
    # Initialisation des variables pour l'algorithme
    a, b = borne_inf, borne_sup
    solution_approx = a
    iteration = 0
    f_a, f_b = f(a), f(b)
    
    # Algorithme principal de dichotomie
    while abs(b - a) >= seuil and f_a * f_b < 0 and iteration < max_iterations:
        iteration += 1
        # Le point milieu est la nouvelle approximation
        solution_approx = (a + b) / 2
        f_solution = f(solution_approx)
        
        if verbose:
            logging.info(f"Itération {iteration}: a={a}, b={b}, solution_approx={solution_approx}, f(solution_approx)={f_solution}")
        
        # Critère d'arrêt supplémentaire: la valeur de la fonction est suffisamment proche de zéro
        if abs(f_solution) < seuil:
            if verbose:
                logging.info(f"Solution trouvée: {solution_approx} après {iteration} itérations (f(solution) ≈ 0)")
            return solution_approx
        
        # Mise à jour de l'intervalle: on conserve le sous-intervalle contenant la racine
        if f_a * f_solution < 0:
            b = solution_approx
            f_b = f_solution
        else:
            a = solution_approx
            f_a = f_solution
    
    # Vérification si le maximum d'itérations a été atteint
    if iteration >= max_iterations:
        logging.warning(f"Maximum d'itérations ({max_iterations}) atteint sans convergence")
    
    # Résultat final après convergence ou arrêt
    if verbose:
        logging.info(f"Solution finale: {solution_approx} après {iteration} itérations")
    
    return solution_approx

def lagrange(fonction: str, borne_inf: float, borne_sup: float, seuil: float, max_iterations: int = 10000, verbose: bool = False) -> float:
    """
   
    Args:
        fonction: Expression mathématique de la fonction sous forme de chaîne
        borne_inf: Borne inférieure de l'intervalle de recherche
        borne_sup: Borne supérieure de l'intervalle de recherche  
        seuil: Seuil de précision pour la convergence (erreur maximale acceptée)
        max_iterations: Nombre maximum d'itérations autorisées
        verbose: Si True, affiche les détails des calculs
    
    Returns:
        float: Approximation de la racine
        
    Raises:
        ErreurNumerique: Si la méthode ne converge pas
    """
    if verbose:
        logging.info(f"Début méthode Lagrange: f(x)={fonction}, intervalle=[{borne_inf}, {borne_sup}], seuil={seuil}, max_iterations={max_iterations}")
    
    def f(x: float) -> float:
        """Évalue la fonction au point x"""
        try:
            return eval(fonction)
        except Exception as e:
            raise ErreurNumerique(f"Erreur lors de l'évaluation de f({x}): {e}")
    
    # Vérification de la validité de l'intervalle
    if borne_inf >= borne_sup:
        raise ErreurNumerique("La borne inférieure doit être strictement inférieure à la borne supérieure")
    
    if seuil <= 0:
        raise ErreurNumerique("Le seuil de précision doit être strictement positif")
    
    if max_iterations <= 0:
        raise ErreurNumerique("Le nombre maximum d'itérations doit être strictement positif")
    
    # Évaluation initiale et vérification du changement de signe
    f_a, f_b = f(borne_inf), f(borne_sup)
    if verbose:
        logging.info(f"f({borne_inf}) = {f_a}, f({borne_sup}) = {f_b}")
    
    if f_a * f_b > 0:
        raise ErreurNumerique("Pas de changement de signe détecté - méthode de Lagrange non applicable")
    
    # Vérification si une des bornes est déjà une solution
    if abs(f_a) < seuil:
        if verbose:
            logging.info(f"Solution exacte trouvée à la borne inférieure: {borne_inf}")
        return borne_inf
    
    if abs(f_b) < seuil:
        if verbose:
            logging.info(f"Solution exacte trouvée à la borne supérieure: {borne_sup}")
        return borne_sup
    
    # Initialisation des variables
    a, b = borne_inf, borne_sup
    racine_approx = a
    iteration = 0
    
    # Algorithme de Lagrange (régula falsi)
    while iteration < max_iterations:
        iteration += 1
        
        # Calcul du point d'intersection de la sécante avec l'axe des x
        f_a, f_b = f(a), f(b)
        if abs(f_b - f_a) < 1e-15:  # Éviter la division par zéro
            raise ErreurNumerique("Dénominateur trop petit - méthode de Lagrange instable")
        
        # Formule de la méthode de la fausse position
        racine_approx = b - f_b * (b - a) / (f_b - f_a)
        f_c = f(racine_approx)
        
        if verbose:
            logging.debug(f"Itération {iteration}: a={a:.6f}, b={b:.6f}, racine_approx={racine_approx:.6f}, f(racine_approx)={f_c:.6e}")
        
        # Critères d'arrêt: précision suffisante atteinte
        if abs(f_c) < seuil or abs(b - a) < seuil:
            if verbose:
                logging.info(f"Convergence atteinte après {iteration} itérations")
                logging.info(f"Racine approximative: {racine_approx}, f(racine)={f_c}")
            return racine_approx
        
        # Mise à jour de l'intervalle basée sur la position de la racine
        if f_a * f_c < 0:
            b = racine_approx
        else:
            a = racine_approx
    
    raise ErreurNumerique(f"Maximum d'itérations ({max_iterations}) atteint sans convergence")

def newton_raphson(fonction: str, derivee: str, point_initial: float, seuil: float, max_iterations: int = 10000, verbose: bool = False) -> float:
    """   
    Args:
        fonction: Expression mathématique de la fonction sous forme de chaîne
        derivee: Expression de la dérivée de la fonction
        point_initial: Point de départ pour l'itération
        seuil: Seuil de précision pour la convergence (erreur maximale acceptée)
        max_iterations: Nombre maximum d'itérations autorisées
        verbose: Si True, affiche les détails des calculs
    
    Returns:
        float: Approximation de la racine
    """
    if verbose:
        logging.info(f"Début méthode Newton-Raphson: f(x)={fonction}, x0={point_initial}, seuil={seuil}, max_iterations={max_iterations}")
    
    def f(x: float) -> float:
        try:
            return eval(fonction)
        except Exception as e:
            raise ErreurNumerique(f"Erreur lors de l'évaluation de f({x}): {e}")
    
    def f_prime(x: float) -> float:
        try:
            return eval(derivee)
        except Exception as e:
            raise ErreurNumerique(f"Erreur lors de l'évaluation de f'({x}): {e}")
    
    # Validation des paramètres
    if seuil <= 0:
        raise ErreurNumerique("Le seuil de précision doit être strictement positif")
    
    if max_iterations <= 0:
        raise ErreurNumerique("Le nombre maximum d'itérations doit être strictement positif")
    
    x_n = point_initial
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Évaluation de la fonction et de sa dérivée au point courant
        f_x = f(x_n)
        f_prime_x = f_prime(x_n)
        
        # Vérification que la dérivée n'est pas nulle (condition essentielle)
        if abs(f_prime_x) < 1e-15:
            raise ErreurNumerique(f"Dérivée nulle au point x={x_n} - méthode de Newton-Raphson impossible")
        
        # Formule de Newton-Raphson: x_{n+1} = x_n - f(x_n)/f'(x_n)
        x_suivant = x_n - f_x / f_prime_x
        ecart = abs(x_suivant - x_n)
        
        if verbose:
            logging.debug(f"Itération {iteration}: x_n={x_n:.6f}, f(x_n)={f_x:.6e}, f'(x_n)={f_prime_x:.6e}, x_suivant={x_suivant:.6f}")
        
        # Critères d'arrêt: l'écart entre les itérations ou la valeur de la fonction est inférieure au seuil
        if ecart < seuil or abs(f(x_suivant)) < seuil:
            if verbose:
                logging.info(f"Convergence atteinte après {iteration} itérations")
                logging.info(f"Racine approximative: {x_suivant}, f(racine)={f(x_suivant):.6e}")
            return x_suivant
        
        x_n = x_suivant
    
    raise ErreurNumerique(f"Maximum d'itérations ({max_iterations}) atteint sans convergence")



def _calculer_rang(matrice: np.ndarray, tol: float = 1e-10) -> int:
    """
    Calcule le rang d'une matrice par élimination de Gauss
    """
    matrice_temp = matrice.copy().astype(float)
    n, m = matrice_temp.shape
    rang = 0
    pivot_col = 0
    
    for ligne in range(n):
        # Trouver le pivot
        pivot_ligne = ligne
        while pivot_ligne < n and abs(matrice_temp[pivot_ligne, pivot_col]) < tol:
            pivot_ligne += 1
        
        if pivot_ligne < n:
            # Échanger les lignes si nécessaire
            if pivot_ligne != ligne:
                matrice_temp[[ligne, pivot_ligne]] = matrice_temp[[pivot_ligne, ligne]]
            
            # Normaliser et éliminer
            pivot_val = matrice_temp[ligne, pivot_col]
            matrice_temp[ligne] /= pivot_val
            
            for i in range(n):
                if i != ligne:
                    facteur = matrice_temp[i, pivot_col]
                    matrice_temp[i] -= facteur * matrice_temp[ligne]
            
            rang += 1
            pivot_col += 1
            if pivot_col >= m:
                break
    
    return rang

def _trouver_base_noyau(matrice: np.ndarray, tol: float = 1e-10) -> List[np.ndarray]:
    """
    Trouve une base du Kerf de la matrice 
    """
    n, m = matrice.shape
    matrice_temp = matrice.copy().astype(float)
    
    # Échelonnage pour trouver les variables libres
    for pivot in range(min(n, m)):
        # Trouver le pivot
        ligne_pivot = pivot
        while ligne_pivot < n and abs(matrice_temp[ligne_pivot, pivot]) < tol:
            ligne_pivot += 1
        
        if ligne_pivot < n:
            if ligne_pivot != pivot:
                matrice_temp[[pivot, ligne_pivot]] = matrice_temp[[ligne_pivot, pivot]]
            
            pivot_val = matrice_temp[pivot, pivot]
            matrice_temp[pivot] /= pivot_val
            
            for i in range(pivot + 1, n):
                facteur = matrice_temp[i, pivot]
                matrice_temp[i] -= facteur * matrice_temp[pivot]
    
    # Identifier variables libres et liées
    variables_libres = []
    variables_liees = []
    
    for col in range(m):
        est_libre = True
        for ligne in range(min(n, m)):
            if abs(matrice_temp[ligne, col]) > tol and ligne == col:
                est_libre = False
                variables_liees.append(col)
                break
        if est_libre:
            variables_libres.append(col)
    
    # Construire la base du noyau
    base_noyau = []
    for var_libre in variables_libres:
        vecteur_sol = np.zeros(m)
        vecteur_sol[var_libre] = 1
        
        # Remplir les variables liées
        for i in range(min(n, m)-1, -1, -1):
            if i in variables_liees:
                somme = 0
                for j in range(i+1, m):
                    somme += matrice_temp[i, j] * vecteur_sol[j]
                vecteur_sol[i] = -somme
        
        base_noyau.append(vecteur_sol)
    
    return base_noyau

def _resoudre_systeme_complet(matrice: np.ndarray, vecteur: np.ndarray, tol: float = 1e-10) -> Optional[np.ndarray]:
    """
    Résout le système AX = B et retourne une solution particulière
    """
    n, m = matrice.shape
    matrice_aug = np.column_stack((matrice.copy(), vecteur.copy())).astype(float)
    
    # Échelonnage
    for pivot in range(min(n, m)):
        # Trouver le pivot
        ligne_pivot = pivot
        while ligne_pivot < n and abs(matrice_aug[ligne_pivot, pivot]) < tol:
            ligne_pivot += 1
        
        if ligne_pivot < n:
            if ligne_pivot != pivot:
                matrice_aug[[pivot, ligne_pivot]] = matrice_aug[[ligne_pivot, pivot]]
            
            pivot_val = matrice_aug[pivot, pivot]
            matrice_aug[pivot] /= pivot_val
            
            for i in range(n):
                if i != pivot:
                    facteur = matrice_aug[i, pivot]
                    matrice_aug[i] -= facteur * matrice_aug[pivot]
    
    # Vérifier la compatibilité
    for ligne in range(n):
        if abs(matrice_aug[ligne, m]) > tol and all(abs(matrice_aug[ligne, j]) < tol for j in range(m)):
            return None  # Système incompatible
    
    # Construire la solution particulière
    solution = np.zeros(m)
    variables_traitees = set()
    
    for ligne in range(min(n, m)):
        for col in range(m):
            if abs(matrice_aug[ligne, col]) > tol and col not in variables_traitees:
                solution[col] = matrice_aug[ligne, m]
                variables_traitees.add(col)
                break
    
    return solution

def resoudre_systeme_lineaire(matrice: np.ndarray, vecteur: np.ndarray, verbose: bool = False) -> SolutionSysteme:
    """
   
    Args:
        matrice: Matrice A du système
        vecteur: Vecteur B du système
        verbose: Si True, affiche les détails
        
    Returns:
        SolutionSysteme: Objet contenant la solution complète
    """
    if verbose:
        logging.info("Début résolution complète du système linéaire")
    
    # Validation des paramètres
    matrice, vecteur = _valider_matrice_vecteur(matrice, vecteur)
    
    n = len(vecteur)
    m = matrice.shape[1]
    
    # Calcul des rangs
    rang_A = _calculer_rang(matrice)
    matrice_augmentee = np.column_stack((matrice, vecteur))
    rang_AB = _calculer_rang(matrice_augmentee)
    
    if verbose:
        logging.info(f"Rang(A) = {rang_A}, Rang(A|B) = {rang_AB}, n = {n}, m = {m}")
    
    # Application du théorème de Rouché-Fontené
    if rang_A != rang_AB:
        if verbose:
            logging.warning("Système sans solution: rang(A) ≠ rang(A|B)")
        return SolutionSysteme('none')
    
    elif rang_A == m:  # Solution unique
        if verbose:
            logging.info("Système avec solution unique")
        solution_part = _resoudre_systeme_complet(matrice, vecteur)
        return SolutionSysteme('unique', solution_part)
    
    else:  # Infinité de solutions
        if verbose:
            logging.info(f"Système sous-déterminé: infinité de solutions (rang = {rang_A} < m = {m})")
        
        # Solution particulière
        solution_part = _resoudre_systeme_complet(matrice, vecteur)
        
        # Base du noyau (solutions de AX = 0)
        base_noyau = _trouver_base_noyau(matrice)
        
        if verbose:
            logging.info(f"Solution particulière: {solution_part}")
            logging.info(f"Dimension du noyau: {len(base_noyau)}")
            for i, vec in enumerate(base_noyau):
                logging.info(f"Vecteur de base {i}: {vec}")
        
        return SolutionSysteme('infinite', solution_part, base_noyau)

def pivot_gauss(matrice: np.ndarray, vecteur: np.ndarray, verbose: bool = False) -> SolutionSysteme:
    """
    Résout le système AX = B par pivot de Gauss avec analyse complète
    """
    return resoudre_systeme_lineaire(matrice, vecteur, verbose)


def gauss_jordan(matrice: np.ndarray, vecteur: Optional[np.ndarray] = None, verbose: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    
    Args:
        matrice: Matrice A du système (carrée)
        vecteur: Vecteur B du système (optionnel pour l'inversion)
        verbose: Si True, affiche les détails des calculs
    
    Returns:
        Solution X ou (A^{-1}, X) selon le cas
    """
    if verbose:
        logging.info("Début méthode de Gauss-Jordan")
    
    n = matrice.shape[0]
    
    if vecteur is not None:
        # Résolution du système AX = B
        matrice_augmentee = np.column_stack((matrice.copy(), vecteur.copy()))
        if verbose:
            logging.info(f"Matrice augmentée initiale:\n{matrice_augmentee}")
    else:
        # Inversion de matrice
        matrice_augmentee = np.column_stack((matrice.copy(), np.eye(n)))
        if verbose:
            logging.info(f"Matrice augmentée pour inversion:\n{matrice_augmentee}")
    
    # Algorithme de Gauss-Jordan
    for pivot in range(n):
        # Recherche du pivot maximal
        ligne_pivot = pivot
        for i in range(pivot + 1, n):
            if abs(matrice_augmentee[i, pivot]) > abs(matrice_augmentee[ligne_pivot, pivot]):
                ligne_pivot = i
        
        # Échange des lignes
        if ligne_pivot != pivot:
            matrice_augmentee[[pivot, ligne_pivot]] = matrice_augmentee[[ligne_pivot, pivot]]
            if verbose:
                logging.debug(f"Échange des lignes {pivot} et {ligne_pivot}")
        
        # Normalisation de la ligne pivot
        pivot_val = matrice_augmentee[pivot, pivot]
        if abs(pivot_val) < 1e-15:
            raise ErreurNumerique("Matrice singulière - inversion impossible")
        
        matrice_augmentee[pivot] /= pivot_val
        
        # Élimination dans toutes les autres lignes
        for ligne in range(n):
            if ligne != pivot:
                facteur = matrice_augmentee[ligne, pivot]
                matrice_augmentee[ligne] -= facteur * matrice_augmentee[pivot]
    
    if vecteur is not None:
        solution = matrice_augmentee[:, n]
        if verbose:
            logging.info("Système résolu avec succès")
            logging.info(f"Solution: {solution}")
        return solution
    else:
        inverse = matrice_augmentee[:, n:]
        if verbose:
            logging.info("Matrice inversée avec succès")
            logging.info(f"Inverse:\n{inverse}")
        return inverse

def methode_crout(matrice: np.ndarray, vecteur: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
  
    
    Args:
        matrice: Matrice A du système (carrée)
        vecteur: Vecteur B du système
        verbose: Si True, affiche les détails des calculs
    
    Returns:
        np.ndarray: Solution X du système
    """
    if verbose:
        logging.info("Début méthode de Crout (décomposition LU)")
    
    n = len(vecteur)
    
    if matrice.shape != (n, n):
        raise ErreurNumerique("Dimensions incompatibles")
    
    # Initialisation des matrices L et U
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    # Décomposition LU selon l'algorithme de Crout
    for i in range(n):
        # Calcul des éléments de U (ligne i)
        for j in range(i, n):
            U[i, j] = matrice[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        
        # Calcul des éléments de L (colonne i)
        for j in range(i, n):
            if i == j:
                L[i, i] = 1.0  # Diagonale de L = 1
            else:
                L[j, i] = (matrice[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    
    if verbose:
        logging.debug(f"Matrice L:\n{L}")
        logging.debug(f"Matrice U:\n{U}")
    
    # Résolution du système triangulaire inférieur LY = B
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = vecteur[i] - sum(L[i, j] * Y[j] for j in range(i))
    
    # Résolution du système triangulaire supérieur UX = Y
    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        X[i] = (Y[i] - sum(U[i, j] * X[j] for j in range(i+1, n))) / U[i, i]
    
    if verbose:
        logging.info("Système résolu avec succès par décomposition LU")
        logging.info(f"Solution: {X}")
    
    return X
def newton_cote(fonction: str, borne_inf: float, borne_sup: float, degre: int, verbose: bool = False) -> float:
    """
    
    Args:
        fonction: Expression mathématique de la fonction à intégrer
        borne_inf: Borne inférieure d'intégration
        borne_sup: Borne supérieure d'intégration
        degre: Degré de la méthode (nombre de points - 1)
        verbose: Si True, affiche les détails des calculs
    
    Returns:
        float: Approximation de l'intégrale
    """
    if verbose:
        logging.info(f"Début méthode Newton-Côte: ∫[{borne_inf},{borne_sup}] {fonction} dx, degré={degre}")
    
    def f(x: float) -> float:
        try:
            return eval(fonction)
        except Exception as e:
            raise ErreurNumerique(f"Erreur lors de l'évaluation de f({x}): {e}")
    
    # Validation des paramètres
    if borne_inf >= borne_sup:
        raise ErreurNumerique("La borne inférieure doit être strictement inférieure à la borne supérieure")
    
    if degre < 2:
        raise ErreurNumerique("Le degré doit être au moins 2")
    
    # Points d'intégration équidistants
    points_x = np.linspace(borne_inf, borne_sup, degre)
    points_y = [f(x) for x in points_x]
    
    # Calcul des coefficients de Newton-Côte
    h = (borne_sup - borne_inf) / (degre - 1)
    integrale = 0.0
    
    # Méthode des trapèzes (degre=2)
    if degre == 2:
        integrale = (borne_sup - borne_inf) * (f(borne_inf) + f(borne_sup)) / 2
    
    # Méthode de Simpson (degre=3)  
    elif degre == 3:
        milieu = (borne_inf + borne_sup) / 2
        integrale = (borne_sup - borne_inf) * (f(borne_inf) + 4*f(milieu) + f(borne_sup)) / 6
    
    # Méthode générale pour les degrés supérieurs
    else:
        # Construction du système pour les coefficients
        matrice_vandermonde = np.vander(points_x, increasing=True)[:, :degre]
        seconds_membres = [(borne_sup**(k+1) - borne_inf**(k+1)) / (k+1) for k in range(degre)]
        
        # Résolution du système linéaire
        coefficients = np.linalg.solve(matrice_vandermonde.T, seconds_membres)
        integrale = np.dot(coefficients, points_y)
    
    if verbose:
        logging.info(f"Intégrale approximative: {integrale}")
    
    return integrale