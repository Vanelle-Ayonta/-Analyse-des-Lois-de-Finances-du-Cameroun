"""
Configuration pour l'extraction budgétaire - Choisir le meilleur modèle
"""

# =============================================================================
# MODÈLES DISPONIBLES ET LEURS CARACTÉRISTIQUES
# =============================================================================

MODELS_CONFIG = {
    "gpt-4.1": {
        "name": "gpt-4.1",
        "description": "Le modèle le plus puissant d'OpenAI (lancé en avril 2025)",
        "avantages": [
            "Meilleure précision d'extraction (+50% vs GPT-4o sur documents complexes)",
            "Contexte 1M tokens (gère de très longs documents)",
            "Excellente compréhension des tableaux et structures",
            "Meilleur suivi des instructions",
            "OCR amélioré pour textes complexes"
        ],
        "inconvénients": [
            "Plus coûteux que GPT-4o",
            "Latence légèrement plus élevée"
        ],
        "prix_input": "$2.50 / 1M tokens",
        "prix_output": "$10.00 / 1M tokens",
        "recommandé_pour": "Documents complexes, extraction critique, haute précision requise",
        "max_tokens": 3000
    },
    
    "gpt-4o": {
        "name": "gpt-4o", 
        "description": "Modèle multimodal rapide et économique (lancé en mai 2024)",
        "avantages": [
            "Très bon rapport qualité/prix",
            "Rapide (2x plus rapide que GPT-4 Turbo)",
            "Bon pour OCR standard",
            "Largement testé et stable"
        ],
        "inconvénients": [
            "Moins précis que GPT-4.1 sur documents complexes",
            "Peut manquer certains détails subtils"
        ],
        "prix_input": "$2.50 / 1M tokens",
        "prix_output": "$10.00 / 1M tokens",
        "recommandé_pour": "Usage général, documents simples à moyennement complexes",
        "max_tokens": 2000
    },
    
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "description": "Modèle compact et rapide",
        "avantages": [
            "Très économique (90% moins cher que GPT-4o)",
            "Très rapide",
            "Bon pour documents simples"
        ],
        "inconvénients": [
            "Moins précis sur documents complexes",
            "Peut manquer des informations"
        ],
        "prix_input": "$0.15 / 1M tokens",
        "prix_output": "$0.60 / 1M tokens",
        "recommandé_pour": "Tests, prototypes, budgets limités",
        "max_tokens": 2000
    },
    
    "gpt-4.1-mini": {
        "name": "gpt-4.1-mini",
        "description": "Version compacte de GPT-4.1 (lancé en avril 2025)",
        "avantages": [
            "Bat GPT-4o sur plusieurs benchmarks",
            "50% plus rapide que GPT-4o",
            "83% moins cher que GPT-4.1",
            "Contexte 1M tokens",
            "Excellent pour documents structurés"
        ],
        "inconvénients": [
            "Peut être moins performant que GPT-4.1 sur cas très complexes"
        ],
        "prix_input": "$0.40 / 1M tokens",
        "prix_output": "$1.60 / 1M tokens",
        "recommandé_pour": "Meilleur équilibre performance/coût, documents moyennement complexes",
        "max_tokens": 2500
    }
}

# =============================================================================
# RECOMMANDATIONS PAR CAS D'USAGE
# =============================================================================

USAGE_RECOMMENDATIONS = {
    "haute_precision": {
        "model": "gpt-4.1",
        "raison": "Pour ne manquer AUCUNE information critique"
    },
    
    "equilibre": {
        "model": "gpt-4.1-mini",
        "raison": "Meilleur rapport qualité/prix/performance"
    },
    
    "economique": {
        "model": "gpt-4o-mini",
        "raison": "Pour tests ou budgets très limités"
    },
    
    "production_standard": {
        "model": "gpt-4o",
        "raison": "Bon équilibre pour la plupart des cas"
    }
}

# =============================================================================
# CONFIGURATION PAR DÉFAUT
# =============================================================================

# Changez cette valeur pour utiliser un modèle différent
# Options: "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"
DEFAULT_MODEL = "gpt-4.1"  # Le plus puissant

# Résolution d'image (Matrix multiplier pour PyMuPDF)
# Plus élevé = meilleure qualité mais fichiers plus gros
# 2 = 200 DPI (standard)
# 3 = 300 DPI (haute qualité, recommandé)
# 4 = 400 DPI (très haute qualité, pour documents très denses)
IMAGE_RESOLUTION_MULTIPLIER = 3

# Détail de l'image pour l'API Vision
# "low" = plus rapide, moins cher, moins précis
# "high" = plus lent, plus cher, plus précis
# "auto" = laisse l'API décider
IMAGE_DETAIL_LEVEL = "high"


def get_model_config(model_name=None):
    """
    Récupère la configuration d'un modèle
    
    Args:
        model_name: Nom du modèle (si None, utilise DEFAULT_MODEL)
    
    Returns:
        Configuration du modèle
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    if model_name not in MODELS_CONFIG:
        raise ValueError(
            f"Modèle '{model_name}' non reconnu. "
            f"Modèles disponibles: {list(MODELS_CONFIG.keys())}"
        )
    
    return MODELS_CONFIG[model_name]


def print_model_comparison():
    """
    Affiche un tableau comparatif des modèles
    """
    print("\n" + "="*80)
    print("COMPARAISON DES MODÈLES OPENAI POUR EXTRACTION BUDGÉTAIRE")
    print("="*80)
    
    for model_name, config in MODELS_CONFIG.items():
        print(f"\n {model_name.upper()}")
        print(f"   {config['description']}")
        print(f"    Prix: {config['prix_input']} (input) / {config['prix_output']} (output)")
        print(f"   ✓ Recommandé pour: {config['recommandé_pour']}")
        print(f"    Avantages:")
        for avantage in config['avantages']:
            print(f"      • {avantage}")
    
    print("\n" + "="*80)
    print("RECOMMANDATIONS PAR CAS D'USAGE")
    print("="*80)
    
    for usage, rec in USAGE_RECOMMENDATIONS.items():
        print(f"\n• {usage.upper().replace('_', ' ')}: {rec['model']}")
        print(f"  → {rec['raison']}")
    
    print("\n" + "="*80 + "\n")

