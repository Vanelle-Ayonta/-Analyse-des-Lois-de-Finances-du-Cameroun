import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
import fitz  # PyMuPDF
from src.analyse_budget.models_config import get_model_config, IMAGE_RESOLUTION_MULTIPLIER, IMAGE_DETAIL_LEVEL


def pdf_page_to_base64(pdf_path: str, page_num: int) -> str:
    """
    Convertit une page PDF en image base64.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        page_num: Numéro de la page (commence à 0)
    
    Returns:
        Image de la page en base64
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    
    # Convertir la page en image avec résolution configurable
    # Résolution déterminée par IMAGE_RESOLUTION_MULTIPLIER dans models_config.py
    pix = page.get_pixmap(matrix=fitz.Matrix(IMAGE_RESOLUTION_MULTIPLIER, IMAGE_RESOLUTION_MULTIPLIER))
    img_bytes = pix.tobytes("png")
    
    doc.close()
    
    return base64.b64encode(img_bytes).decode('utf-8')


def extract_all_chapters_summary(pdf_path: str, pages: List[int], model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Extrait un résumé de tous les chapitres présents dans les pages spécifiées.
    Utile pour vérifier qu'aucun chapitre n'a été manqué.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        pages: Liste des numéros de pages à analyser
        model_name: Nom du modèle à utiliser
    
    Returns:
        Dictionnaire avec les chapitres trouvés et leur première page d'apparition
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model_config = get_model_config(model_name)
    
    print(f"\n Analyse des chapitres dans le document...")
    print(f"   Modèle: {model_config['name']}\n")
    
    chapters_found = {}
    
    for page_num in pages:
        page_index = page_num - 1
        base64_image = pdf_page_to_base64(pdf_path, page_index)
        
        prompt = """
Scanne cette page et identifie TOUS les chapitres présents.

Cherche TOUT texte au format:
- "CHAPITRE XX - TITRE"
- "CHAPITRE XX- TITRE" 
- "CHAPITREXX-TITRE"

Où XX est un numéro (01, 02, 03, etc.)

Retourne UNIQUEMENT un JSON avec un tableau de chapitres:
{
    "chapitres": ["CHAPITRE 01 - TITRE 1", "CHAPITRE 02 - TITRE 2", ...]
}

Si aucun chapitre n'est trouvé, retourne {"chapitres": []}
"""
        
        response = client.chat.completions.create(
            model=model_config['name'],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": IMAGE_DETAIL_LEVEL
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0
        )
        
        content = response.choices[0].message.content
        
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            chapitres = data.get("chapitres", [])
            
            for chapitre in chapitres:
                if chapitre not in chapters_found:
                    chapters_found[chapitre] = page_num
                    print(f"   ✓ Page {page_num}: {chapitre}")
        
        except Exception as e:
            print(f"Erreur page {page_num}: {e}")
    
    print(f"\n Résumé: {len(chapters_found)} chapitre(s) unique(s) trouvé(s)\n")
    
    return {
        "total_chapters": len(chapters_found),
        "chapters": chapters_found
    }


def extract_budget_info_from_pages(
    pdf_path: str, 
    pages: List[int],
    model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extrait les informations budgétaires (Code, Libellé, CP, AE, Chapitre) 
    d'un PDF pour les pages spécifiées.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        pages: Liste des numéros de pages à traiter (commence à 1)
        model_name: Nom du modèle à utiliser (None = utilise DEFAULT_MODEL)
                   Options: "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"
    
    Returns:
        Liste des projets extraits avec leurs informations en format JSON
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Récupérer la configuration du modèle
    model_config = get_model_config(model_name)
    
    print(f"\n Utilisation du modèle: {model_config['name']}")
    print(f"   {model_config['description']}")
    print(f"   Max tokens: {model_config['max_tokens']}\n")
    
    results = []
    current_chapter = None
    
    for page_num in pages:
        # Convertir le numéro de page (1-indexed vers 0-indexed)
        page_index = page_num - 1
        
        # Convertir la page en base64
        base64_image = pdf_page_to_base64(pdf_path, page_index)
        
        # Créer le prompt pour l'extraction - prompt ultra-optimisé pour capturer TOUS les chapitres
        prompt = f"""
Tu es un expert en extraction de données budgétaires. Ta mission est d'extraire TOUTES les informations avec une précision MAXIMALE.

ÉTAPE 1 - DÉTECTER TOUS LES CHAPITRES (CRITIQUE):
Scanne TOUTE la page et identifie CHAQUE occurrence de chapitre, même s'il est seul sur une ligne.
Formats possibles:
- "CHAPITRE 01 - TITRE"
- "CHAPITRE 02 - TITRE"  
- "CHAPITRE 03 - TITRE"
- "CHAPITRE 04- TITRE" (même sans espace)
- "CHAPITRE05-TITRE" (même collé)
- Les chapitres peuvent être en gras, soulignés, ou en majuscules
- Les chapitres peuvent être sur une ligne séparée AVANT les projets

RÈGLE ABSOLUE: Si tu vois le mot "CHAPITRE" suivi d'un numéro QUELQUE PART sur la page, tu DOIS l'extraire.

ÉTAPE 2 - EXTRAIRE TOUS LES PROJETS:
Pour CHAQUE ligne qui contient un code de programme:
   
a) **Code** (N°): Numéro du programme (2-3 chiffres)
b) **Libellé**: Description complète (peut être sur plusieurs lignes)
c) **CP**: Crédits de Paiement (montant en milliers FCFA)
d) **AE**: Autorisations d'Engagement (montant en milliers FCFA)

RÈGLES STRICTES:
✓ Extrais TOUS les chapitres visibles (même plusieurs sur une page)
✓ Extrais TOUTES les lignes avec un code de programme
✓ Recopie EXACTEMENT le texte (espaces, majuscules, accents)
✓ Pour les montants: garde les espaces (ex: "21 459 760")
✓ Si une colonne est vide/illisible: utilise "N/A"
✓ NE REFORMULE JAMAIS
✓ NE SAUTE AUCUNE ligne

ATTENTION PARTICULIÈRE:
- Une page peut contenir PLUSIEURS chapitres (extrait-les TOUS)
- Les chapitres sont souvent sur une ligne séparée AVANT les projets
- Même si un chapitre n'a pas de projets après lui, extrais-le quand même
- Les titres de chapitres peuvent avoir des variations typographiques

Format JSON requis:
{{
    "chapitres": ["CHAPITRE 01 - TITRE", "CHAPITRE 02 - TITRE", ...],
    "projets": [
        {{
            "code": "XXX",
            "libelle": "TEXTE EXACT COMPLET",
            "cp": "XXX XXX XXX",
            "ae": "XXX XXX XXX"
        }}
    ]
}}

IMPORTANT: Retourne un tableau "chapitres" avec TOUS les chapitres trouvés sur cette page (peut être vide si aucun).
"""
        
        # Appeler l'API OpenAI Vision avec le modèle configuré
        response = client.chat.completions.create(
            model=model_config['name'],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": IMAGE_DETAIL_LEVEL
                            }
                        }
                    ]
                }
            ],
            max_tokens=model_config['max_tokens'],
            temperature=0  # Température à 0 pour des résultats déterministes
        )
        
        # Extraire la réponse
        content = response.choices[0].message.content
        
        # Parser le JSON de la réponse
        try:
            # Nettoyer la réponse si elle contient des balises markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            page_data = json.loads(content)
            
            # Gérer les chapitres trouvés sur cette page
            chapitres_trouves = page_data.get("chapitres", [])
            
            # Si des chapitres sont trouvés, mettre à jour le chapitre courant
            # On prend le dernier chapitre trouvé comme chapitre actuel
            if chapitres_trouves and len(chapitres_trouves) > 0:
                # Si plusieurs chapitres sur la page, on utilise le dernier
                current_chapter = chapitres_trouves[-1]
                print(f"   Page {page_num}: Chapitre(s) détecté(s): {', '.join(chapitres_trouves)}")
                print(f"      → Chapitre actif: {current_chapter}")
            else:
                print(f"   Page {page_num}: Aucun nouveau chapitre (chapitre actif: {current_chapter})")
            
            # Ajouter le chapitre courant à chaque projet
            for projet in page_data.get("projets", []):
                projet["chapitre"] = current_chapter
                projet["page"] = page_num
                results.append(projet)
                
        except json.JSONDecodeError as e:
            print(f" Erreur lors du parsing JSON pour la page {page_num}: {e}")
            print(f" Contenu reçu: {content[:200]}...")
            
            # Tentative de récupération: essayer de parser manuellement
            # Chercher un format alternatif dans la réponse
            try:
                # Parfois le modèle retourne un format légèrement différent
                import re
                # Chercher les chapitres dans le texte brut
                chapter_matches = re.findall(r'CHAPITRE\s*\d+\s*-\s*[A-Z\s]+', content, re.IGNORECASE)
                if chapter_matches:
                    current_chapter = chapter_matches[-1].strip()
                    print(f" Récupération: Chapitre extrait du texte brut: {current_chapter}")
            except:
                pass
    
    return results



def extract_chapters_with_context(pdf_path: str, pages: List[int]) -> List[Dict[str, Any]]:
    """
    Extrait les chapitres avec le projet précédent et suivant.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        pages: Liste des numéros de pages à analyser (commence à 1)
    
    Returns:
        Liste de dictionnaires avec pour chaque chapitre trouvé:
        {
            "chapitre": "CHAPITRE XX - TITRE EXACT",
            "projet_precedent": {
                "code": "XXX",
                "libelle": "...",
                "cp": "...",
                "ae": "..."
            } ou null,
            "projet_suivant": {
                "code": "XXX",
                "libelle": "...",
                "cp": "...",
                "ae": "..."
            } ou null
        }
        
        Retourne liste vide si aucun chapitre trouvé
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    results = []
    
    for page_num in pages:
        page_index = page_num - 1
        base64_image = pdf_page_to_base64(pdf_path, page_index)
        
        prompt = """
Analyse cette page et identifie TOUS les chapitres présents.

Pour CHAQUE chapitre trouvé, extrais:
1. L'intitulé EXACT du chapitre (format "CHAPITRE XX - TITRE")
2. Le projet qui vient JUSTE AVANT le chapitre (s'il existe sur la page)
3. Le projet qui vient JUSTE APRÈS le chapitre

Un projet contient:
- code: Numéro du programme (2-3 chiffres)
- libelle: Description complète
- cp: Crédits de Paiement
- ae: Autorisations d'Engagement

RÈGLES IMPORTANTES:
- Recopie EXACTEMENT l'intitulé du chapitre tel qu'écrit
- Si pas de projet avant le chapitre: projet_precedent = null
- Si pas de projet après le chapitre: projet_suivant = null
- Si AUCUN chapitre sur la page: retourne liste vide []
- Un chapitre DOIT avoir AU MOINS un projet avant OU après (sinon ignore-le)

Format JSON:
{
    "chapitres": [
        {
            "chapitre": "CHAPITRE XX - TITRE EXACT",
            "projet_precedent": {
                "code": "XXX",
                "libelle": "TEXTE EXACT",
                "cp": "XXX XXX",
                "ae": "XXX XXX"
            },
            "projet_suivant": {
                "code": "XXX",
                "libelle": "TEXTE EXACT",
                "cp": "XXX XXX",
                "ae": "XXX XXX"
            }
        }
    ]
}

Si aucun chapitre: {"chapitres": []}
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0
            )
            
            content = response.choices[0].message.content
            
            # Nettoyer le JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            chapitres = data.get("chapitres", [])
            
            # Ajouter le numéro de page à chaque chapitre
            for chapitre_data in chapitres:
                chapitre_data["page"] = page_num
                results.append(chapitre_data)
        
        except Exception as e:
            print(f"Erreur page {page_num}: {e}")
            continue
    
    return results




def save_results_to_json(results: List[Dict[str, Any]], output_path: str):
    """
    Sauvegarde les résultats dans un fichier JSON.
    
    Args:
        results: Liste des projets extraits
        output_path: Chemin du fichier de sortie
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Résultats sauvegardés dans: {output_path}")



def return_resulta_chapter(resultats):
    '''retourne une liste de dictionnaire avec chapitre, libelle_precedent et libelle_suivant'''
    resultat_list_chapter = []
    for i in resultats:
        chapitre = i['chapitre']
        try:
            libelle_precedent = i['projet_precedent'].get('libelle')
        except Exception as e:
            libelle_precedent = None
        try:
            libelle_suivant = i['projet_suivant'].get('libelle')
        except Exception as e:
            libelle_suivant = None
            
        resultat_list_chapter.append({
                'chapitre': chapitre,
                'libelle_precedent': libelle_precedent,
                'libelle_suivant': libelle_suivant
            })    
    return resultat_list_chapter

#save_results_to_json(return_resulta_chapter(resultats), 'budget_extract_chapter_2024_2025.json')
