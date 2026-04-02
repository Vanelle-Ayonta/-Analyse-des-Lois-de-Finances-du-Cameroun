# -*- coding: utf-8 -*-
"""
Module de prétraitement des textes de loi de finances
"""

import re
import unicodedata
import pandas as pd
from typing import List, Union


def pretraiter_texte_loi_finances(texte_brut: str) -> str:
    """
    Nettoyage expert pour les articles de Loi de Finances (Cameroun).
    Traite : Bruit OCR sévère, tampons, puces de remplissage et en-têtes.
    Preserve : Chiffres, taux, dates et mots clés juridiques.
    
    Args:
        texte_brut: Texte brut à nettoyer
        
    Returns:
        Texte nettoyé
    """
    if not texte_brut or not isinstance(texte_brut, str):
        return ""

    # 1. Normalisation Unicode (Gère les variantes de points/puces comme •, ·, .)
    texte = unicodedata.normalize('NFKC', texte_brut)
    texte = texte.lower()

    # 2. Reconstruction linéaire (Supprime les sauts de ligne pour recoller les phrases)
    texte = texte.replace('\n', ' ')

    # --- ZONE A : FILTRES ADMINISTRATIFS & TAMPONS (Sources 2, 54, 276, 320) ---
    patterns_bruit = [
        r"p\s*r\s*e\s*s\s*i\s*d\s*e\s*n\s*c\s*e",  # PRESIDENCE "éclaté"
        r"r\s*e\s*p\s*u\s*b\s*l\s*i\s*q\s*u\s*e",  # REPUBLIQUE
        r"s\s*e\s*c\s*r\s*e\s*t\s*a\s*r\s*i\s*a\s*t",  # SECRETARIAT
        r"service du fichier",
        r"legislative and statutory",
        r"certified true copy",
        r"copie certifiee conforme",
        r"er\s*-\s*tif\s*-\s*ie",  # "certifie" cassé (Source 38)
        r"rl\s*i\s*fied\s*true\s*y",  # "certified true copy" cassé
        r"y\s*affa\s*irs",  # "affairs" cassé
        r"yaounde.*?biya",  # Signature (Source 78)
        r"le reste sans changement"  # (Source 16)
    ]

    for pat in patterns_bruit:
        texte = re.sub(pat, " ", texte, flags=re.IGNORECASE)

    # --- ZONE B : NETTOYAGE VISUEL & OCR (Sources 23, 40) ---

    # 1. Suppression des séries de puces/points indiquant "pas de changement"
    # Cible : ........, ••••••••, ········ (plus de 2 occurrences)
    texte = re.sub(r"(?:[\.•·-]\s*){3,}", " ", texte)

    # 2. Suppression du "charabia" OCR (Backslashes, chevrons, suites de ponctuation)
    # Ex: \ 11 \.., l l. (Source 40)
    texte = re.sub(r'[\\\/_<>«»|]{2,}', ' ', texte)  # Symboles répétés
    texte = re.sub(r'\\[\w\.]+', ' ', texte)  # Mots commençant par backslash

    # 3. Suppression du préfixe "Article X" (On garde le fond du texte)
    # Ex: "Article 93 quater.-" -> Supprimé
    pattern_article = r"^article\s+(?:\d+|[a-z]+)(?:\s+(?:bis|ter|quater|quinquies))?[\s\.\-]+(?:\(\d+\))?"
    texte = re.sub(pattern_article, " ", texte).strip()

    # --- ZONE C : NETTOYAGE DES MOTS PARASITES ---

    # Liste des mots courts valides à conserver (Stopwords utiles pour le sens)
    mots_courts_gardes = {
        "a", "à", "y", "et", "ou", "au", "du", "en", "un", "le", "la", "de", "ne", "ni", "se", "ce", "il", "on"
    }

    mots = texte.split()
    mots_propres = []

    for mot in mots:
        # Nettoyage ponctuation collée
        mot_clean = mot.strip(".,;:-'\"")

        # 1. Si c'est un chiffre ou contient un chiffre (ex: "1er", "5%"), ON GARDE
        if any(c.isdigit() for c in mot_clean):
            mots_propres.append(mot_clean)
            continue

        # 2. Si le mot est assez long (>2 lettres), ON GARDE
        if len(mot_clean) > 2:
            # Filtre supplémentaire anti-bruit (ex: "rvice" ou "co")
            if mot_clean not in ["rvice", "ncy", "secr"]:
                mots_propres.append(mot_clean)

        # 3. Si c'est un mot court valide, ON GARDE
        elif mot_clean in mots_courts_gardes:
            mots_propres.append(mot_clean)

        # Sinon (lettres isolées w, c, m, r...), on jette.

    # Reconstitution finale
    return " ".join(mots_propres)



def pretraiter_liste_articles(articles: List[str]) -> List[str]:
    """
    Applique le prétraitement à une liste d'articles.
    
    Args:
        articles: Liste d'articles bruts
        
    Returns:
        Liste d'articles nettoyés
    """
    return [pretraiter_texte_loi_finances(article) for article in articles]


def charger_articles_json(filepath: str) -> pd.DataFrame:
    """
    Charge les articles depuis un fichier JSON.
    
    Args:
        filepath: Chemin vers le fichier JSON
        
    Returns:
        DataFrame contenant les articles
    """
    import json
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return pd.DataFrame(data)


def nettoyer_montant(valeur: Union[str, int, float]) -> float:
    """
    Nettoie et convertit un montant budgétaire en float.
    Gère les formats comme "1 000 000" ou "1.000.000,50"
    
    Args:
        valeur: Montant à nettoyer (str, int ou float)
        
    Returns:
        Montant en float
    """
    if isinstance(valeur, str):
        # Enlève les espaces insécables et convertit en float
        clean_val = valeur.replace(' ', '').replace('.', '').replace(',', '.')
        try:
            return float(clean_val)
        except:
            return 0.0
    return float(valeur) if isinstance(valeur, (int, float)) else 0.0


def preparer_donnees_budget(budget_df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Prépare un DataFrame budget avec nettoyage des montants.
    
    Args:
        budget_df: DataFrame budget brut
        source: Identifiant de la source (ex: "2023-2024")
        
    Returns:
        DataFrame nettoyé avec colonnes ae_clean et cp_clean
    """
    df = budget_df.copy()
    df['source'] = source
    
    # Nettoyage des montants si les colonnes existent
    if 'ae' in df.columns:
        df['ae_clean'] = df['ae'].apply(nettoyer_montant)
    if 'cp' in df.columns:
        df['cp_clean'] = df['cp'].apply(nettoyer_montant)
    
    return df
