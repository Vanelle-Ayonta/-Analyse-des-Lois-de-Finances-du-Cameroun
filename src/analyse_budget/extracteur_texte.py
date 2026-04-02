
import pdfplumber
import re
from typing import List, Dict, Tuple


def extraire_articles_loi_finances(chemin_pdf: str) -> List[Dict[str, str]]:
    """
    Extrait tous les articles d'une loi de finances PDF et les structure par chapitre.
    
    Cette fonction analyse un PDF de loi de finances camerounaise et extrait :
    - Les articles de la partie "haute" (avant "ÉVALUATION DES RESSOURCES")
    - Les articles de la partie "DISPOSITIONS SPECIALES"
    
    Pour chaque article, elle récupère :
    - Le numéro du chapitre
    - Le titre du chapitre
    - Le texte complet de l'article
    
    Args:
        chemin_pdf (str): Chemin absolu ou relatif vers le fichier PDF de la loi de finances.
                         Exemple: "LOI DES FINANCES 2024-2025.pdf"
    
    Returns:
        List[Dict[str, str]]: Liste de dictionnaires où chaque dictionnaire représente un article
                             avec les clés suivantes:
                             - 'chapitre_numero': str - Ex: "CHAPITRE 20"
                             - 'chapitre_titre': str - Ex: "MINISTERE DE LA SANTE PUBLIQUE"
                             - 'texte_complet': str - Texte intégral de l'article
    
    Raises:
        FileNotFoundError: Si le fichier PDF n'existe pas
        Exception: Si le PDF ne peut pas être lu ou est corrompu
    
    Example:
        >>> articles = extraire_articles_loi_finances("LOI DES FINANCES 2024-2025.pdf")
        >>> print(f"Nombre d'articles extraits: {len(articles)}")
        >>> print(f"Premier article: {articles[0]['chapitre_numero']}")
        >>> print(articles[0]['texte_complet'][:100])
    
    Note:
        - La fonction ignore automatiquement les en-têtes et pieds de page administratifs
        - Les articles sont extraits dans l'ordre d'apparition dans le PDF
        - Les chapitres sont détectés automatiquement via le mot-clé "CHAPITRE"
    """
    
    # ÉTAPE 1: Chargement et extraction de toutes les lignes du PDF
    print(f" Chargement du fichier : {chemin_pdf}...")
    toutes_les_lignes = _charger_et_extraire_lignes(chemin_pdf)
    print(f" Extraction terminée : {len(toutes_les_lignes)} lignes récupérées")
    
    # ÉTAPE 2: Identification des sections principales du document
    print(" Identification des sections du document...")
    sections = _identifier_sections(toutes_les_lignes)
    
    # ÉTAPE 3: Extraction des articles de chaque section
    print(" Extraction des articles...")
    articles_partie_haute = _extraire_articles_depuis_lignes(sections['articles_partie_haute'])
    articles_dispositions = _extraire_articles_depuis_lignes(sections['dispositions_speciales'])
    
    # ÉTAPE 4: Fusion et retour des résultats
    tous_les_articles = articles_partie_haute + articles_dispositions
    print(f" Extraction terminée : {len(tous_les_articles)} articles extraits")
    
    return tous_les_articles


def _charger_et_extraire_lignes(chemin_pdf: str) -> List[str]:
    """
    Charge un PDF et extrait toutes les lignes en ignorant les en-têtes/pieds de page.
    
    Cette fonction interne:
    1. Ouvre le PDF page par page
    2. Rogne 5% du haut et 10% du bas pour éliminer les tampons administratifs
    3. Extrait le texte ligne par ligne
    
    Args:
        chemin_pdf (str): Chemin vers le fichier PDF
    
    Returns:
        List[str]: Liste de toutes les lignes de texte extraites
    """
    toutes_les_lignes = []
    
    with pdfplumber.open(chemin_pdf) as pdf:
        for page in pdf.pages:
            # Définition de la zone utile (cropping pour éviter en-têtes/pieds de page)
            # Les tampons administratifs (Presidence, Secretariat General) sont en haut/bas
            largeur = page.width
            hauteur = page.height
            bbox = (0, hauteur * 0.05, largeur, hauteur * 0.95)  # (x0, top, x1, bottom)
            
            # Extraction du texte de la zone utile
            page_utile = page.crop(bbox=bbox)
            texte_brut = page_utile.extract_text()
            
            if texte_brut:
                # Découpage en lignes et nettoyage
                lignes_page = texte_brut.split('\n')
                toutes_les_lignes.extend([ligne.strip() for ligne in lignes_page])
    
    return toutes_les_lignes


def _identifier_sections(lignes: List[str]) -> Dict[str, List[str]]:
    """
    Identifie et découpe le document en sections principales.
    
    Structure type d'une loi de finances camerounaise:
    1. Articles partie haute (de la ligne 1 à "ÉVALUATION DES RESSOURCES")
    2. Section budgétaire (ignorée dans cette extraction)
    3. Dispositions spéciales (de "DISPOSITIONS SPECIALES" à la fin)
    
    Args:
        lignes (List[str]): Toutes les lignes du document
    
    Returns:
        Dict[str, List[str]]: Dictionnaire avec les sections:
                             - 'articles_partie_haute': lignes des articles du début
                             - 'dispositions_speciales': lignes des dispositions spéciales
    """
    # Recherche des marqueurs de section
    ligne_fin_partie_haute = _rechercher_ligne_exacte(lignes, ["ÉVALUATION", "DES", "RESSOURCES"])
    ligne_debut_dispositions = _rechercher_ligne_exacte(lignes, ["DISPOSITIONS", "SPECIALES"])
    
    # Découpage du document
    return {
        'articles_partie_haute': lignes[1:ligne_fin_partie_haute],
        'dispositions_speciales': lignes[ligne_debut_dispositions:len(lignes) - 1]
    }


def _rechercher_ligne_exacte(lignes: List[str], mots_cles: List[str]) -> int:
    """
    Recherche la première ligne contenant EXACTEMENT les mots-clés spécifiés.
    
    La recherche ignore:
    - La casse (majuscules/minuscules)
    - L'ordre des mots
    - La ponctuation
    
    Args:
        lignes (List[str]): Liste de lignes à analyser
        mots_cles (List[str]): Mots qui doivent être présents dans la ligne
    
    Returns:
        int: Index de la première ligne correspondante, ou -1 si non trouvé
    
    Example:
        >>> lignes = ["Titre", "ÉVALUATION DES RESSOURCES", "Suite"]
        >>> idx = _rechercher_ligne_exacte(lignes, ["ÉVALUATION", "DES", "RESSOURCES"])
        >>> print(idx)  # Affiche: 1
    """
    # Préparation de l'ensemble de référence (en minuscules, sans espaces inutiles)
    set_mots_cles = set(mot.lower().strip() for mot in mots_cles)
    
    for i, ligne in enumerate(lignes):
        if not ligne:
            continue
        
        # Nettoyage de la ligne: suppression de la ponctuation et découpage
        ligne_nettoyee = re.sub(r'[^\w\s]', ' ', ligne.lower())
        mots_ligne = ligne_nettoyee.split()
        set_ligne = set(mots_ligne)
        
        # Comparaison stricte des ensembles
        if set_ligne == set_mots_cles:
            return i
    
    return -1


def _extraire_articles_depuis_lignes(lignes: List[str]) -> List[Dict[str, str]]:
    """
    Extrait les articles structurés depuis une liste de lignes.
    
    Cette fonction analyse les lignes pour détecter:
    1. Les numéros de chapitre (lignes commençant par "CHAPITRE")
    2. Les titres de chapitre (ligne suivant le numéro)
    3. Les articles (lignes commençant par "ARTICLE")
    4. Le contenu de chaque article (jusqu'au prochain article ou section)
    
    Args:
        lignes (List[str]): Liste de lignes à analyser
    
    Returns:
        List[Dict[str, str]]: Liste des articles avec leur structure complète
    
    Note:
        - Un titre de chapitre est considéré valide s'il contient plus de 2 mots
          de 4 caractères ou plus
        - Les sections (SECTION, TITRE, SOUS-SECTION, LIVRE) marquent la fin
          d'un article
    """
    articles = []
    article_courant = None
    en_article = False
    
    # Variables de contexte pour les chapitres
    chapitre_numero = "Non spécifié"
    chapitre_titre = "Non spécifié"
    attente_titre_chapitre = False
    
    def est_titre_valide(texte: str) -> bool:
        """Vérifie si un texte est un titre de chapitre valide."""
        mots = texte.split()
        mots_valides = [mot for mot in mots if len(mot) >= 4]
        return len(mots_valides) > 2
    
    for ligne in lignes:
        if not isinstance(ligne, str):
            continue
        
        ligne_clean = ligne.strip()
        ligne_upper = ligne_clean.upper()
        
        # Gestion du titre de chapitre attendu
        if attente_titre_chapitre and ligne_clean and est_titre_valide(ligne_clean):
            chapitre_titre = ligne_clean
            attente_titre_chapitre = False
            continue
        elif attente_titre_chapitre and ligne_clean:
            # Si pas valide, on continue à chercher
            continue
        
        # Détection d'un nouveau chapitre
        if ligne_upper.startswith('CHAPITRE') and len(ligne_clean) > 8:
            chapitre_numero = ligne_clean
            attente_titre_chapitre = True
            en_article = False
            continue
        
        # Détection d'un nouvel article
        if ligne_upper.startswith('ARTICLE'):
            # Sauvegarde de l'article précédent s'il existe
            if article_courant:
                articles.append(article_courant)
            
            # Création d'un nouvel article
            article_courant = {
                'chapitre_numero': chapitre_numero,
                'chapitre_titre': chapitre_titre,
                'contenu': [ligne_clean]
            }
            en_article = True
            attente_titre_chapitre = False
        
        # Détection de fin d'article (nouvelle section)
        elif any(mot in ligne_upper for mot in ['SECTION', 'TITRE', 'SOUS-SECTION', 'LIVRE']) \
             and ligne_upper.isupper() and len(ligne_clean) > 5:
            if article_courant:
                articles.append(article_courant)
                article_courant = None
            en_article = False
            attente_titre_chapitre = False
        
        # Ajout de contenu à l'article en cours
        elif en_article and article_courant and ligne_clean:
            article_courant['contenu'].append(ligne_clean)
    
    # Ajout du dernier article si nécessaire
    if article_courant:
        articles.append(article_courant)
    
    # Formatage final des articles
    return [
        {
            'chapitre_numero': art['chapitre_numero'],
            'chapitre_titre': art['chapitre_titre'],
            'texte_complet': '\n'.join(art['contenu'])
        }
        for art in articles
    ]

