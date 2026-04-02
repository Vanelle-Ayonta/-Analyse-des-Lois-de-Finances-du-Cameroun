# 🇨🇲 Analyse des Lois de Finances du Cameroun

Plateforme d'analyse budgétaire et sémantique des lois de finances du Cameroun.  
Elle mesure la **conformité des allocations budgétaires avec les piliers de la SND30** (Stratégie Nationale de Développement 2030).

Lien de l'application streamlit : https://analyse-loi-des-finances-cameroun.streamlit.app/

---

## Architecture du projet

```
Analyse-budget-cameroun/
├── app.py              # Point d'entrée Streamlit
├── src/
│   └── analyse_budget/           # Package Python principal
│       ├── models_config.py      # Config modèles OpenAI
│       ├── pretraitement.py      # Nettoyage OCR & textes de loi
│       ├── extracteur_texte.py   # Extraction articles depuis PDF (pdfplumber)
│       ├── extract_budget_info.py# Extraction lignes budgétaires (Vision GPT-4.1)
│       ├── extraire_ligne_budgetaire_info.py
│       ├── classification.py     # Zero-shot CamemBERT XNLI → piliers SND30
│       ├── analyse_budgetaire.py # Conformité, Gini, graphiques Plotly
│       └── analyse_semantique.py # BiEncoder CamemBERT, ruptures, similarité
├── data/                         # Données JSON extraites (incluses)
│   ├── budget_extract_2023_2024.json
│   ├── budget_extract_2024_2025.json
│   ├── articles_extract_2023_2024.json
│   └── articles_extract_2024_2025.json
├── .streamlit/
│   └── config.toml               # Thème et config serveur Streamlit
├── .gitignore                    # Fichiers à ignorer lors du push sur github
├── pyproject.toml                # Dépendances Poetry
├── Dockerfile                    # Déploiement containerisé
├── Makefile                      # Commandes raccourcies
└── .env                  # Template variables d'environnement
```

---

## Démarrage rapide

### Prérequis

- Python **3.10+**
- [Poetry](https://python-poetry.org/docs/#installation) — gestionnaire de dépendances

```bash
# Installer Poetry (si pas déjà installé)
curl -sSL https://install.python-poetry.org | python3 -
```

### 1. Cloner et installer

```bash
git clone https://github.com/AssaAllo/ANALYSE-LOI-DES-FINANCES-CAMEROUN
cd analyse-budget-cameroun

# Installer toutes les dépendances
poetry install
# ou via le Makefile :
make install
```

### 2. Configurer les variables d'environnement

```bash
cp .env
# Ouvrir .env et renseigner votre clé OpenAI
nano .env
```

Contenu minimal du `.env` :
```env
OPENAI_API_KEY=sk-...
```

> La clé OpenAI est **uniquement nécessaire** pour la page "Extraction" en mode Vision IA.  
> Les analyses sémantiques et budgétaires tournent **entièrement en local** apès avoir téméchargé le modèle de Hugging Face.

### 3. Lancer l'application

```bash
poetry run streamlit run app.py
# ou via le Makefile :
make run
```


---

## Pages de l'application

  - **Vue d'ensemble** : KPIs globaux (nombre de lignes, articles, enveloppes CP par exercice) et graphiques de synthèse.
  - **Extraction et données brutes** :
    - Consultation des JSON budgétaires et des articles déjà extraits.
    - Extraction avancée depuis PDF via `pdfplumber`, `PyMuPDF` et les appels OpenAI (si configurés).
  - **Prétraitement des textes** :
    - Utilisation de des modules pour nettoyage OCR, tampons et bruit.
  - **Classification SND30** :
    - Zero-shot classification des libellés et articles avec CamemBERT XNLI stocké localement dans `modele/`.
  - **Analyse budgétaire & conformité** :
    - Fusion budget + classification puis analyse d'alignement (fréquence vs budget) et de concentration (indice de Gini).
  - **Audit sémantique** :
    - Encodage SentenceTransformer et matrice de similarité cosinus entre articles de deux exercices.


---

## Déploiement Docker

```bash
# Build
make docker-build
# ou
docker build -t analyse-budget-cameroun .

# Lancement
make docker-run
# ou
docker run -p 8501:8501 --env-file .env analyse-budget-cameroun
```

---


## Stack technique

| Composant | Technologie |
|---|---|
| Interface | Streamlit 1.35+ |
| Extraction PDF texte | pdfplumber |
| Extraction PDF Vision | GPT-4.1 / OpenAI Vision API |
| Classification SND30 | CamemBERT + XNLI (mtheo/camembert-base-xnli) |
| Analyse sémantique | BiEncoder CamemBERT (antoinelouis/biencoder-camembert-base-mmarcoFR) |
| Visualisation | Plotly, Matplotlib, WordCloud |
| Packaging | Poetry |
| Déploiement | Docker / Streamlit Cloud |

---

## Données incluses

Les fichiers JSON dans `data/` sont déjà extraits et prêts à l'emploi :

- `budget_extract_2023_2024.json` — lignes budgétaires LFI 2023-2024
- `budget_extract_2024_2025.json` — lignes budgétaires LFI 2024-2025  
- `articles_extract_2023_2024.json` — articles de loi 2023-2024
- `articles_extract_2024_2025.json` — articles de loi 2024-2025

Pour ré-extraire depuis les PDF originaux, utilisez la **Extraction et données brutes**.

---

## Notes importantes

- **Modèles HuggingFace** : téléchargés automatiquement au premier lancement (444 Mo) car n'étant pas push sur Github.  
  Ils sont mis en cache dans `./modele/`. Ne pas supprimer ce dossier entre les sessions.

- **GPU** : la classification et l'analyse sémantique détectent automatiquement un GPU CUDA.  
  En l'absence de GPU, le traitement se fait sur CPU (plus lent).

- **Coûts API** : l'extraction Vision (Page 1) consomme des crédits OpenAI.  
  Utilisez `gpt-4.1-mini` pour réduire les coûts sur des volumes importants.