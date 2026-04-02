# -*- coding: utf-8 -*-
"""
Module de classification des lignes budgétaires selon les piliers SND30
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import pipeline
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.font_manager import FontProperties


def classer_ligne_dep_snd30(
    liste_libelles: List[str],
    batch_size: int = 16,
    device: str = "auto",
    model_name: str = "mtheo/camembert-base-xnli",
    modele_dir: str = "./modele",
    hypothesis_template: str = "Ce texte parle de {}."
) -> pd.DataFrame:
    """
    Classifie des libellés budgétaires en français via zero-shot classification
    avec un modèle CamemBERT fine-tuné sur XNLI.
    
    Args:
        liste_libelles: Liste des libellés à classifier
        batch_size: Taille des lots pour le traitement
        device: "auto", 0 (GPU) ou -1 (CPU)
        model_name: Nom du modèle HuggingFace
        modele_dir: Répertoire de sauvegarde du modèle
        hypothesis_template: Template pour l'hypothèse
        
    Returns:
        DataFrame avec colonnes Libellé, Pilier, Score
    """
    os.makedirs(modele_dir, exist_ok=True)
    safe_folder = model_name.replace("/", "__")
    local_model_path = os.path.join(modele_dir, safe_folder)

    # Charger depuis le disque si déjà téléchargé
    if os.path.isdir(local_model_path) and os.listdir(local_model_path):
        model_source = local_model_path
        print(f" Chargement du modèle depuis : {model_source}")
    else:
        print(f" Téléchargement initial du modèle {model_name}...")
        try:
            path = snapshot_download(
                repo_id=model_name,
                local_dir=local_model_path,
                local_dir_use_symlinks=False
            )
            model_source = path
            print(f" Téléchargement terminé : {model_source}")
        except Exception as e:
            raise RuntimeError(
                f" Échec du téléchargement de {model_name}. "
                f"Vérifiez votre connexion. Erreur : {e}"
            )

    # Gestion du device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else -1
    elif isinstance(device, str):
        raise ValueError("Paramètre 'device' doit être 'auto', 0 (GPU) ou -1 (CPU).")

    print(f" Utilisation du device : {'GPU' if device >= 0 else 'CPU'}")

    # Création du pipeline
    classifier = pipeline(
        "zero-shot-classification",
        model=model_source,
        tokenizer=model_source,
        device=device
    )

    piliers = [
        "Transformation structurelle et économique",
        "Capital humain et bien-être",
        "Gouvernance et administration",
        "Développement régional et décentralisation"
    ]

    resultats = []
    for i in tqdm(range(0, len(liste_libelles), batch_size), desc="Classification"):
        batch = liste_libelles[i:i + batch_size]
        predictions = classifier(
            batch,
            candidate_labels=piliers,
            hypothesis_template=hypothesis_template,
            multi_label=False,
            batch_size=min(batch_size, len(batch))
        )
        if isinstance(predictions, dict):
            predictions = [predictions]
        for lib, pred in zip(batch, predictions):
            resultats.append({
                "Libellé": lib,
                "Pilier": pred['labels'][0],
                "Score": round(pred['scores'][0], 4)
            })

    return pd.DataFrame(resultats)


def plot_repartition_piliers_par_annee(df_classification: pd.DataFrame, 
                                       source_col: str = 'source') -> go.Figure:
    """
    Crée un graphique en barres empilées de la répartition des piliers par année.
    
    Args:
        df_classification: DataFrame avec colonnes Pilier et source
        source_col: Nom de la colonne source
        
    Returns:
        Figure Plotly
    """
    df = df_classification.copy()
    df['Annee'] = df[source_col].str.split('-').str[0].astype(int)
    
    counts = df.groupby(['Annee', 'Pilier']).size().reset_index(name='Count')
    
    fig = go.Figure()
    
    for year in sorted(counts['Annee'].unique()):
        year_data = counts[counts['Annee'] == year]
        fig.add_trace(
            go.Bar(
                y=year_data['Pilier'],
                x=year_data['Count'],
                name=str(year),
                orientation='h',
                text=year_data['Count'],
                textposition='auto',
                hoverinfo='text',
                hovertext=[f"{row.Pilier}: {row.Count} lignes ({year})" for _, row in year_data.iterrows()]
            )
        )
    
    fig.update_layout(
        title=" Répartition des piliers par année",
        xaxis_title="Nombre de libellés classifiés",
        yaxis_title="Pilier stratégique",
        barmode='stack',
        height=600,
        legend_title="Année",
        template="plotly_white"
    )
    
    return fig


def plot_distribution_scores_classification(df_classification: pd.DataFrame,
                                           source_col: str = 'source') -> go.Figure:
    """
    Crée un histogramme de distribution des scores de classification par année.
    
    Args:
        df_classification: DataFrame avec colonnes Score et source
        source_col: Nom de la colonne source
        
    Returns:
        Figure Plotly
    """
    df = df_classification.copy()
    df['Annee'] = df[source_col].str.split('-').str[0].astype(int)
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3[:len(df['Annee'].unique())]
    
    for i, (year, group) in enumerate(df.groupby('Annee')):
        fig.add_trace(
            go.Histogram(
                x=group['Score'],
                name=str(year),
                nbinsx=20,
                opacity=0.75,
                marker_color=colors[i],
                histnorm='probability',
                hovertemplate=(
                    f"<b>Année {year}</b><br>"
                    "Score: %{x:.3f}<br>"
                    "Fréquence: %{y:.2%}<extra></extra>"
                )
            )
        )
    
    fig.update_layout(
        title=" Distribution des scores de classification par année",
        xaxis_title="Score de confiance (0–1)",
        yaxis_title="Proportion des libellés",
        barmode='overlay',
        legend_title="Année",
        template="plotly_white",
        height=500
    )
    fig.update_xaxes(range=[0, 1])
    
    return fig


def plot_boxplot_scores_par_pilier(df_classification: pd.DataFrame,
                                   source_col: str = 'source') -> go.Figure:
    """
    Crée des boxplots des scores de classification par pilier et année.
    
    Args:
        df_classification: DataFrame avec colonnes Pilier, Score et source
        source_col: Nom de la colonne source
        
    Returns:
        Figure Plotly
    """
    df = df_classification.copy()
    df['Annee'] = df[source_col].str.split('-').str[0].astype(int)
    
    piliers = df['Pilier'].unique()
    n_piliers = len(piliers)
    
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=piliers,
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )
    
    annees = sorted(df['Annee'].unique())
    colors_annee = px.colors.qualitative.Plotly[:len(annees)]
    annee_to_color = dict(zip(annees, colors_annee))
    
    row_col = [(1,1), (1,2), (2,1), (2,2)]
    
    for idx, pilier in enumerate(piliers):
        row, col = row_col[idx]
        data_pilier = df[df['Pilier'] == pilier]
        
        for year in annees:
            subset = data_pilier[data_pilier['Annee'] == year]
            if not subset.empty:
                fig.add_trace(
                    go.Box(
                        y=subset['Score'],
                        x=[str(year)] * len(subset),
                        name=str(year),
                        marker_color=annee_to_color[year],
                        showlegend=(idx == 0),
                        boxmean=True,
                        hovertemplate=(
                            f"<b>{pilier}</b><br>"
                            f"Année: {year}<br>"
                            "Score: %{y:.3f}<extra></extra>"
                        )
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(
        title=" Scores de classification par pilier et année",
        height=800,
        template="plotly_white",
        legend_title="Année"
    )
    
    fig.update_yaxes(title_text="Score de confiance", range=[0, 1])
    fig.update_xaxes(title_text="Année")
    
    return fig


def preparer_dataframe_classification(df_classification_1: pd.DataFrame,
                                     df_classification_2: pd.DataFrame,
                                     source_1: str = '2023-2024',
                                     source_2: str = '2024-2025') -> pd.DataFrame:
    """
    Concatène deux DataFrames de classification avec leurs sources.
    
    Args:
        df_classification_1: Premier DataFrame de classification
        df_classification_2: Deuxième DataFrame de classification
        source_1: Label de la première source
        source_2: Label de la deuxième source
        
    Returns:
        DataFrame concaténé
    """
    df1 = df_classification_1.copy()
    df2 = df_classification_2.copy()
    
    df1['source'] = source_1
    df2['source'] = source_2
    
    return pd.concat([df1, df2], ignore_index=True)


# ==========================================
# SECTION: CLASSIFICATION DES ARTICLES
# ==========================================

def classer_articles_snd30(
    liste_articles: List[str],
    batch_size: int = 16,
    device: str = "auto",
    model_name: str = "mtheo/camembert-base-xnli",
    modele_dir: str = "./modele",
    hypothesis_template: str = "Ce texte parle de {}."
) -> pd.DataFrame:
    """
    Classifie des articles de loi en français via zero-shot classification
    avec un modèle CamemBERT fine-tuné sur XNLI.
    
    Args:
        liste_articles: Liste des articles à classifier
        batch_size: Taille des lots pour le traitement
        device: "auto", 0 (GPU) ou -1 (CPU)
        model_name: Nom du modèle HuggingFace
        modele_dir: Répertoire de sauvegarde du modèle
        hypothesis_template: Template pour l'hypothèse
        
    Returns:
        DataFrame avec colonnes Article, Pilier, Score
    """
    os.makedirs(modele_dir, exist_ok=True)
    safe_folder = model_name.replace("/", "__")
    local_model_path = os.path.join(modele_dir, safe_folder)

    # Charger depuis le disque si déjà téléchargé
    if os.path.isdir(local_model_path) and os.listdir(local_model_path):
        model_source = local_model_path
        print(f" Chargement du modèle depuis : {model_source}")
    else:
        print(f" Téléchargement initial du modèle {model_name}...")
        try:
            path = snapshot_download(
                repo_id=model_name,
                local_dir=local_model_path,
                local_dir_use_symlinks=False
            )
            model_source = path
            print(f" Téléchargement terminé : {model_source}")
        except Exception as e:
            raise RuntimeError(
                f" Échec du téléchargement de {model_name}. "
                f" Vérifiez votre connexion. Erreur : {e}"
            )

    # Gestion du device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else -1
    elif isinstance(device, str):
        raise ValueError("Paramètre 'device' doit être 'auto', 0 (GPU) ou -1 (CPU).")

    print(f" Utilisation du device : {'GPU' if device >= 0 else 'CPU'}")

    # Création du pipeline
    classifier = pipeline(
        "zero-shot-classification",
        model=model_source,
        tokenizer=model_source,
        device=device
    )

    piliers = [
        "Transformation structurelle et économique",
        "Capital humain et bien-être",
        "Gouvernance et administration",
        "Développement régional et décentralisation"
    ]

    resultats = []
    for i in tqdm(range(0, len(liste_articles), batch_size), desc="Classification des articles"):
        batch = liste_articles[i:i + batch_size]
        predictions = classifier(
            batch,
            candidate_labels=piliers,
            hypothesis_template=hypothesis_template,
            multi_label=False,
            batch_size=min(batch_size, len(batch))
        )
        if isinstance(predictions, dict):
            predictions = [predictions]
        for article, pred in zip(batch, predictions):
            resultats.append({
                "Article": article,
                "Pilier": pred['labels'][0],
                "Score": round(pred['scores'][0], 4)
            })

    return pd.DataFrame(resultats)


def plot_repartition_articles_piliers(df_articles_classifies: pd.DataFrame, 
                                      source_col: str = 'source',
                                      pilier_col: str = 'Pilier') -> go.Figure:
    """
    Crée un graphique en barres empilées horizontales de la répartition des articles par pilier.
    Similaire à plot_repartition_piliers_par_annee mais pour les articles.
    
    Args:
        df_articles_classifies: DataFrame avec colonnes Pilier et source
        source_col: Nom de la colonne source
        pilier_col: Nom de la colonne pilier
        
    Returns:
        Figure Plotly
    """
    df = df_articles_classifies.copy()
    
    # Extraire l'année si nécessaire
    if '-' in str(df[source_col].iloc[0]):
        df['Annee'] = df[source_col].str.split('-').str[0].astype(int)
    else:
        df['Annee'] = df[source_col]
    
    counts = df.groupby(['Annee', pilier_col]).size().reset_index(name='Count')
    
    fig = go.Figure()
    
    for year in sorted(counts['Annee'].unique()):
        year_data = counts[counts['Annee'] == year]
        fig.add_trace(
            go.Bar(
                y=year_data[pilier_col],
                x=year_data['Count'],
                name=str(year),
                orientation='h',
                text=year_data['Count'],
                textposition='auto',
                hoverinfo='text',
                hovertext=[f"{row[pilier_col]}: {row.Count} articles ({year})" for _, row in year_data.iterrows()]
            )
        )
    
    fig.update_layout(
        title=" Répartition des articles de loi par pilier SND30",
        xaxis_title="Nombre d'articles classifiés",
        yaxis_title="Pilier stratégique",
        barmode='stack',
        height=600,
        legend_title="Année",
        template="plotly_white"
    )
    
    return fig


def plot_distribution_scores_articles(df_articles_classifies: pd.DataFrame,
                                      source_col: str = 'source',
                                      score_col: str = 'Score') -> go.Figure:
    """
    Crée un histogramme de distribution des scores de classification des articles par année.
    Similaire à plot_distribution_scores_classification mais pour les articles.
    
    Args:
        df_articles_classifies: DataFrame avec colonnes Score et source
        source_col: Nom de la colonne source
        score_col: Nom de la colonne score
        
    Returns:
        Figure Plotly
    """
    df = df_articles_classifies.copy()
    
    # Extraire l'année si nécessaire
    if '-' in str(df[source_col].iloc[0]):
        df['Annee'] = df[source_col].str.split('-').str[0].astype(int)
    else:
        df['Annee'] = df[source_col]
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3[:len(df['Annee'].unique())]
    
    for i, (year, group) in enumerate(df.groupby('Annee')):
        fig.add_trace(
            go.Histogram(
                x=group[score_col],
                name=str(year),
                nbinsx=20,
                opacity=0.75,
                marker_color=colors[i],
                histnorm='probability',
                hovertemplate=(
                    f"<b>Année {year}</b><br>"
                    "Score: %{x:.3f}<br>"
                    "Fréquence: %{y:.2%}<extra></extra>"
                )
            )
        )
    
    fig.update_layout(
        title=" Distribution des scores de classification des articles par année",
        xaxis_title="Score de confiance (0–1)",
        yaxis_title="Proportion des articles",
        barmode='overlay',
        legend_title="Année",
        template="plotly_white",
        height=500
    )
    fig.update_xaxes(range=[0, 1])
    
    return fig


def plot_boxplot_scores_articles(df_articles_classifies: pd.DataFrame,
                                 source_col: str = 'source',
                                 pilier_col: str = 'Pilier',
                                 score_col: str = 'Score') -> go.Figure:
    """
    Crée des boxplots des scores de classification des articles par pilier et année.
    Similaire à plot_boxplot_scores_par_pilier mais pour les articles.
    
    Args:
        df_articles_classifies: DataFrame avec colonnes Pilier, Score et source
        source_col: Nom de la colonne source
        pilier_col: Nom de la colonne pilier
        score_col: Nom de la colonne score
        
    Returns:
        Figure Plotly
    """
    df = df_articles_classifies.copy()
    
    # Extraire l'année si nécessaire
    if '-' in str(df[source_col].iloc[0]):
        df['Annee'] = df[source_col].str.split('-').str[0].astype(int)
    else:
        df['Annee'] = df[source_col]
    
    piliers = df[pilier_col].unique()
    n_piliers = len(piliers)
    
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=piliers,
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )
    
    annees = sorted(df['Annee'].unique())
    colors_annee = px.colors.qualitative.Plotly[:len(annees)]
    annee_to_color = dict(zip(annees, colors_annee))
    
    row_col = [(1,1), (1,2), (2,1), (2,2)]
    
    for idx, pilier in enumerate(piliers):
        if idx >= 4:  # Maximum 4 subplots
            break
        row, col = row_col[idx]
        data_pilier = df[df[pilier_col] == pilier]
        
        for year in annees:
            subset = data_pilier[data_pilier['Annee'] == year]
            if not subset.empty:
                fig.add_trace(
                    go.Box(
                        y=subset[score_col],
                        x=[str(year)] * len(subset),
                        name=str(year),
                        marker_color=annee_to_color[year],
                        showlegend=(idx == 0),
                        boxmean=True,
                        hovertemplate=(
                            f"<b>{pilier}</b><br>"
                            f"Année: {year}<br>"
                            "Score: %{y:.3f}<extra></extra>"
                        )
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(
        title=" Scores de classification des articles par pilier et année",
        height=800,
        template="plotly_white",
        legend_title="Année"
    )
    
    fig.update_yaxes(title_text="Score de confiance", range=[0, 1])
    fig.update_xaxes(title_text="Année")
    
    return fig


def analyser_distribution_scores_articles(df_articles_classifies: pd.DataFrame,
                                         score_col: str = 'Score',
                                         pilier_col: str = 'Pilier') -> pd.DataFrame:
    """
    Analyse statistique de la distribution des scores de classification par pilier.
    
    Args:
        df_articles_classifies: DataFrame avec colonnes Score et Pilier
        score_col: Nom de la colonne score
        pilier_col: Nom de la colonne pilier
        
    Returns:
        DataFrame avec statistiques par pilier
    """
    stats = df_articles_classifies.groupby(pilier_col)[score_col].agg([
        ('Nombre', 'count'),
        ('Score_Moyen', 'mean'),
        ('Score_Median', 'median'),
        ('Score_Min', 'min'),
        ('Score_Max', 'max'),
        ('Ecart_Type', 'std')
    ]).reset_index()
    
    stats = stats.round(3)
    stats = stats.sort_values('Score_Moyen', ascending=False)
    
    return stats


# ==========================================
# SECTION: NUAGES DE MOTS
# ==========================================


def plot_wordcloud_projets_par_pilier(df_classification: pd.DataFrame,
                                      annee: str = '2023-2024',
                                      libelle_col: str = 'Libellé',
                                      pilier_col: str = 'Pilier',
                                      source_col: str = 'source') -> plt.Figure:
    """
    Crée un nuage de mots pour chaque pilier (grille 2x2) pour les projets budgétaires.
    """
    # Filtrer par année
    df_annee = df_classification[df_classification[source_col] == annee].copy()
    
    if len(df_annee) == 0:
        raise ValueError(f"Aucune donnée trouvée pour l'année {annee}")
    
    # Obtenir les 4 piliers
    piliers = sorted(df_annee[pilier_col].unique())
    if len(piliers) < 4:
        raise ValueError(f"Besoin de 4 piliers, seulement {len(piliers)} trouvés")
    
    # Configuration globale
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['figure.dpi'] = 150
    
    # Créer la figure avec 2x2 subplots (taille optimisée pour la lisibilité)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=150)
    fig.suptitle(f" Nuages de mots des projets budgétaires par pilier ({annee})", 
                 fontsize=24, fontweight='bold', y=0.95)
    
    # Stopwords français personnalisés
    stopwords_fr = {
        'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 'à', 'au', 'aux',
        'en', 'dans', 'pour', 'par', 'sur', 'avec', 'sans', 'sous', 'entre', 'vers',
        'chez', 'dont', 'que', 'qui', 'quoi', 'se', 'ce', 'cette', 'ces', 'son', 'sa',
        'ses', 'leur', 'leurs', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'notre', 'nos',
        'votre', 'vos', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles', 'je', 'tu',
        'est', 'sont', 'était', 'étaient', 'été', 'être', 'avoir', 'a', 'ont', 'avait',
        'avaient', 'eu', 'y', 'en', 'ne', 'pas', 'plus', 'moins', 'très', 'tout', 'tous',
        'toute', 'toutes', 'autre', 'autres', 'même', 'si', 'comme', 'mais', 'car', 'donc'
    }
    
    # Paramètres communs pour les nuages de mots
    common_params = {
        'width': 1200,
        'height': 900,
        'background_color': 'white',
        'stopwords': stopwords_fr,
        'colormap': 'viridis',
        'max_words': 35,
        'relative_scaling': 0.8,
        'min_font_size': 20,
        'prefer_horizontal': 0.95,
        'font_path': None  # Utilise la police par défaut de Matplotlib
    }
    
    # Générer chaque nuage de mots
    for idx, (ax, pilier) in enumerate(zip(axes.flatten(), piliers[:4])):
        # Texte du pilier
        textes_pilier = df_annee[df_annee[pilier_col] == pilier][libelle_col].tolist()
        texte_combine = ' '.join(textes_pilier).lower()
        
        if not texte_combine.strip():
            ax.axis('off')
            ax.set_title(f"{pilier} (Aucune donnée)", fontsize=18)
            continue
        
        # Générer le wordcloud
        wordcloud = WordCloud(**common_params).generate(texte_combine)
        
        # Afficher le nuage de mots
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(pilier, fontsize=20, fontweight='bold', pad=15)
        ax.axis('off')
    
    # Ajustement des espaces
    plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    plt.subplots_adjust(top=0.9)
    
    return fig


def plot_wordcloud_articles_par_pilier(df_articles_classifies: pd.DataFrame,
                                       annee: str = '2023-2024',
                                       article_col: str = 'Article',
                                       pilier_col: str = 'Pilier',
                                       source_col: str = 'source') -> plt.Figure:
    """
    Crée un nuage de mots pour chaque pilier (grille 2x2) pour les articles de loi.
    """
    # Filtrer par année
    df_annee = df_articles_classifies[df_articles_classifies[source_col] == annee].copy()
    
    if len(df_annee) == 0:
        raise ValueError(f"Aucune donnée trouvée pour l'année {annee}")
    
    # Obtenir les 4 piliers
    piliers = sorted(df_annee[pilier_col].unique())
    if len(piliers) < 4:
        raise ValueError(f"Besoin de 4 piliers, seulement {len(piliers)} trouvés")
    
    # Configuration globale
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['figure.dpi'] = 150
    
    # Créer la figure avec 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=150)
    fig.suptitle(f" Nuages de mots des articles de loi par pilier ({annee})", 
                 fontsize=24, fontweight='bold', y=0.95)
    
    # Stopwords français personnalisés (avec "article" en plus)
    stopwords_fr = {
        'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 'à', 'au', 'aux',
        'en', 'dans', 'pour', 'par', 'sur', 'avec', 'sans', 'sous', 'entre', 'vers',
        'chez', 'dont', 'que', 'qui', 'quoi', 'se', 'ce', 'cette', 'ces', 'son', 'sa',
        'ses', 'leur', 'leurs', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'notre', 'nos',
        'votre', 'vos', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles', 'je', 'tu',
        'est', 'sont', 'était', 'étaient', 'été', 'être', 'avoir', 'a', 'ont', 'avait',
        'avaient', 'eu', 'y', 'en', 'ne', 'pas', 'plus', 'moins', 'très', 'tout', 'tous',
        'toute', 'toutes', 'autre', 'autres', 'même', 'si', 'comme', 'mais', 'car', 'donc',
        'article', 'articles', 'alinéa', 'alinéas', 'paragraphe', 'alinéas','fcfa','cfa','f'
    }
    
    # Paramètres communs
    common_params = {
        'width': 1200,
        'height': 900,
        'background_color': 'white',
        'stopwords': stopwords_fr,
        'colormap': 'plasma',
        'max_words': 35,
        'relative_scaling': 0.8,
        'min_font_size': 20,
        'prefer_horizontal': 0.95,
        'font_path': None
    }
    
    # Générer chaque nuage de mots
    for idx, (ax, pilier) in enumerate(zip(axes.flatten(), piliers[:4])):
        # Texte du pilier
        textes_pilier = df_annee[df_annee[pilier_col] == pilier][article_col].tolist()
        texte_combine = ' '.join(textes_pilier).lower()
        
        if not texte_combine.strip():
            ax.axis('off')
            ax.set_title(f"{pilier} (Aucune donnée)", fontsize=18)
            continue
        
        # Générer le wordcloud
        wordcloud = WordCloud(**common_params).generate(texte_combine)
        
        # Afficher le nuage de mots
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(pilier, fontsize=20, fontweight='bold', pad=15)
        ax.axis('off')
    
    # Ajustement des espaces
    plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    plt.subplots_adjust(top=0.9)
    
    return fig


def plot_wordcloud_par_pilier(df_classification: pd.DataFrame,
                              texte_col: str = 'Libellé',
                              pilier_col: str = 'Pilier',
                              source_col: str = 'source',
                              annee_cible: str = '2023-2024',
                              max_words: int = 50,
                              background_color: str = 'white') -> go.Figure:
    """
    Crée un graphique 2x2 avec un nuage de mots par pilier pour une année donnée.
    
    Args:
        df_classification: DataFrame avec colonnes texte, Pilier et source
        texte_col: Nom de la colonne contenant le texte (Libellé ou Article)
        pilier_col: Nom de la colonne pilier
        source_col: Nom de la colonne source
        annee_cible: Année à visualiser (ex: '2023-2024')
        max_words: Nombre maximum de mots par nuage
        background_color: Couleur de fond
        
    Returns:
        Figure Plotly avec 4 subplots (un par pilier)
    """
    from wordcloud import WordCloud
    from PIL import Image
    import io
    import base64
    
    # Filtrer pour l'année cible
    df_annee = df_classification[df_classification[source_col] == annee_cible].copy()
    
    if len(df_annee) == 0:
        raise ValueError(f"Aucune donnée trouvée pour l'année {annee_cible}")
    
    # Obtenir les 4 piliers
    piliers = sorted(df_annee[pilier_col].unique())[:4]  # Limiter à 4 piliers
    
    if len(piliers) < 4:
        print(f" Attention : seulement {len(piliers)} piliers trouvés pour {annee_cible}")
    
    # Créer la grille de subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=piliers,
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
        specs=[[{'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # Liste de mots vides français à ignorer
    stopwords_fr = {
        'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 'mais', 
        'pour', 'dans', 'sur', 'avec', 'par', 'au', 'aux', 'ce', 'ces', 'son',
        'sa', 'ses', 'qui', 'que', 'quoi', 'dont', 'où', 'il', 'elle', 'on',
        'nous', 'vous', 'ils', 'elles', 'à', 'en', 'y', 'ne', 'pas', 'plus',
        'tout', 'tous', 'toute', 'toutes', 'leur', 'leurs', 'mon', 'ma', 'mes',
        'ton', 'ta', 'tes', 'notre', 'nos', 'votre', 'vos', 'se', 'si', 'être',
        'avoir', 'faire', 'dit', 'comme', 'bien', 'aussi', 'très', 'peut',
        'sans', 'sous', 'entre', 'après', 'avant', 'pendant', 'depuis','fcfa','cfa','f'
    }
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for idx, pilier in enumerate(piliers):
        if idx >= 4:
            break
            
        row, col = positions[idx]
        
        # Récupérer tous les textes pour ce pilier
        textes_pilier = df_annee[df_annee[pilier_col] == pilier][texte_col].tolist()
        
        # Combiner tous les textes
        texte_complet = ' '.join([str(t).lower() for t in textes_pilier if pd.notna(t)])
        
        if len(texte_complet.strip()) == 0:
            # Pas de texte pour ce pilier, afficher un message
            fig.add_annotation(
                text=f"Pas de données pour<br>{pilier}",
                xref=f"x{idx+1}" if idx > 0 else "x",
                yref=f"y{idx+1}" if idx > 0 else "y",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color='gray'),
                row=row, col=col
            )
            continue
        
        # Générer le nuage de mots
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color=background_color,
            max_words=max_words,
            stopwords=stopwords_fr,
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=8
        ).generate(texte_complet)
        
        # Convertir le wordcloud en image
        img = wordcloud.to_image()
        
        # Convertir l'image PIL en base64 pour Plotly
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        
        # Ajouter l'image au subplot
        fig.add_layout_image(
            dict(
                source=f'data:image/png;base64,{img_base64}',
                xref=f"x{idx+1}" if idx > 0 else "x",
                yref=f"y{idx+1}" if idx > 0 else "y",
                x=0, y=1,
                sizex=1, sizey=1,
                sizing="stretch",
                layer="below"
            )
        )
        
        # Configurer les axes pour chaque subplot
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)
    
    # Titre principal
    type_doc = "projets budgétaires" if texte_col == "Libellé" else "articles de loi"
    fig.update_layout(
        title=f" Nuages de mots par pilier SND30 ({annee_cible})<br><sub>Analyse des {type_doc}</sub>",
        height=800,
        showlegend=False,
        template='plotly_white',
        margin=dict(t=100, b=20, l=20, r=20)
    )
    
    return fig

