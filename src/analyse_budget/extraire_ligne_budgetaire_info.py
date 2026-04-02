from extract_budget_info import *
import pdfplumber
from dotenv import load_dotenv

def trouver_page(pdf_path, elements):
    # Normaliser les éléments recherchés (majuscule)
    elements_norm = [e.upper() for e in elements]
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extraire le texte brut
            texte = page.extract_text()
            if not texte:
                continue
            
            # Découper en lignes
            lignes = texte.split("\n")
            
            for ligne in lignes:
                # Découper la ligne en mots
                mots = ligne.split()
                mots_norm = [m.upper() for m in mots]
                
                # Vérifier si la ligne correspond exactement aux éléments
                if mots_norm == elements_norm:
                    return page_num, ligne
    
    return None, None

def extraire_ligne_par_page(pdf_path="LOI DES FINANCES 2023-2024.pdf",json_output_name='budget_extract_1.json'):
    pdf_lien = pdf_path

    # Exemple d’utilisation
    page_debut_ligne_budgetaire, ligne = trouver_page(pdf_lien,[ 'MOYENS', 'DES', 'POLITIQUES', 'PUBLIQUES', 'ET', 'DISPOSITIONS', 'SPÉCIALES'])
    if page_debut_ligne_budgetaire:
        print(f"Première occurrence trouvée à la page {page_debut_ligne_budgetaire} : {ligne}")
    else:
        print("Aucune ligne correspondante trouvée.")


    # Exemple d’utilisation
    page_fin_ligne_budgetaire, ligne = trouver_page(pdf_lien, ["DISPOSITIONS", "SPECIALES"])

    page_fin_ligne_budgetaire = page_fin_ligne_budgetaire

    if page_fin_ligne_budgetaire:
        print(f"Première occurrence trouvée à la page {page_fin_ligne_budgetaire} : {ligne}")
    else:
        print("Aucune ligne correspondante trouvée.")





    
    # Exemple: extraire les pages 1 et 2 d'un PDF
    pdf_path = pdf_lien
    pages_to_extract = [i for i in range(page_debut_ligne_budgetaire, page_fin_ligne_budgetaire + 1)] 

    # Extraire les informations
    results = extract_budget_info_from_pages(pdf_path, pages_to_extract)

    # Afficher les résultats
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # Sauvegarder dans un fichier
    save_results_to_json(results, "budget_extract_1.json")
    return None


