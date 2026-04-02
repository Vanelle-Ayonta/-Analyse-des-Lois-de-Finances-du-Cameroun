# -*- coding: utf-8 -*-
"""
Module d'analyse sémantique pour la comparaison d'articles de loi
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.mixture import GaussianMixture
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def audit_semantique_lois(
    articles_2024: List[str],
    articles_2025: List[str],
    model_name: str = "antoinelouis/biencoder-camembert-base-mmarcoFR",
    local_model_path: Optional[str] = None,
) -> Dict:
    """
    Audit sémantique avec SentenceTransformer (BiEncoder CamemBERT).
    
    Args:
        articles_2024: Liste des articles de la première année
        articles_2025: Liste des articles de la deuxième année
        model_name: Nom du modèle HuggingFace à utiliser (si local_model_path n'est pas fourni)
        local_model_path: Chemin local vers un modèle SentenceTransformer déjà téléchargé.
            Si renseigné, aucun appel réseau à HuggingFace n'est effectué (local_files_only=True).
        
    Returns:
        Dictionnaire avec similarités, ruptures et évolutions thématiques
    """
    # Chargement du modèle
    print(" Chargement modèle BiEncoder CamemBERT...")
    try:
        if local_model_path is not None:
            # Utilise exclusivement les fichiers locaux, sans contact réseau
            model = SentenceTransformer(local_model_path, local_files_only=True)
        else:
            # Comportement par défaut : téléchargement possible depuis HuggingFace
            model = SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(
            "Échec du chargement du modèle sémantique. "
            "Si vous êtes hors-ligne, renseignez un chemin local dans l'application "
            "ou assurez-vous que le modèle est déjà mis en cache.\n"
            f"Détail de l'erreur : {e}"
        ) from e

    # Calcul des embeddings (vectorisation en batch = plus rapide)
    print(" Encodage des articles 2024...")
    emb_2024 = model.encode(articles_2024, show_progress_bar=True, convert_to_tensor=True)

    print(" Encodage des articles 2025...")
    emb_2025 = model.encode(articles_2025, show_progress_bar=True, convert_to_tensor=True)

    # Matrice de similarité cosinus (optimisée GPU si disponible)
    print(" Calcul matrice de similarité...")
    similarity_matrix = util.cos_sim(emb_2024, emb_2025)

    # Extraction des paires les plus similaires/divergentes
    resultats = []
    for i in range(len(articles_2024)):
        for j in range(len(articles_2025)):
            sim = similarity_matrix[i][j].item()
            resultats.append({
                'idx_2024': i,
                'idx_2025': j,
                'similarite': sim,
                'article_2024': articles_2024[i][:150],
                'article_2025': articles_2025[j][:150]
            })

    # Tri et analyse
    resultats.sort(key=lambda x: x['similarite'], reverse=True)

    return {
        'top_10_similaires': resultats[:10],
        'top_10_ruptures': resultats[-10:],
        'score_moyen': np.mean([r['similarite'] for r in resultats]),
        'matrice_complete': similarity_matrix.cpu().numpy()
    }


def calculer_seuil_gmm(similarites: np.ndarray) -> float:
    """
    Calcule le seuil optimal de similarité avec un GMM à 2 composantes.
    
    Args:
        similarites: Array des scores de similarité
        
    Returns:
        Seuil optimal
    """
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(similarites.reshape(-1, 1))

    # Moyennes des 2 clusters
    mu1, mu2 = sorted(gmm.means_.flatten())
    
    # Seuil = intersection des gaussiennes
    seuil = (mu1 + mu2) / 2
    return seuil


def trouver_seuil_optimal(data: pd.DataFrame, col_similarite: str = 'similarite_max_artice', 
                         seuils: np.ndarray = None) -> Dict:
    """
    Trouve le seuil optimal de similarité par méthode du coude.
    
    Args:
        data: DataFrame avec similarités
        col_similarite: Nom colonne des scores
        seuils: Liste de seuils à tester (None = auto)
        
    Returns:
        Dictionnaire avec seuil optimal et métriques
    """
    similarites = data[col_similarite].values

    # Générer seuils à tester
    if seuils is None:
        seuils = np.arange(0, 1.0, 0.05)

    # Méthode du coude (Elbow) - non supervisé
    coudes = []
    for seuil in seuils:
        similaires = (similarites >= seuil).sum()
        coudes.append({'seuil': seuil, 'nb_similaires': similaires})

    # Trouver le point d'inflexion (max de dérivée seconde)
    nb_sim = [c['nb_similaires'] for c in coudes]
    derivee_2 = np.diff(np.diff(nb_sim))
    idx_optimal = np.argmax(np.abs(derivee_2)) + 1

    return {
        'seuil_optimal': seuils[idx_optimal],
        'distribution': coudes
    }


def extraire_ruptures(data: pd.DataFrame, seuil: float = 0.45, 
                     col_similarite: str = 'similarite_max_artice') -> pd.DataFrame:
    """
    Extrait les articles en rupture selon un seuil de similarité.
    
    Args:
        data: DataFrame avec les articles et leurs scores
        seuil: Seuil de similarité
        col_similarite: Nom de la colonne de similarité
        
    Returns:
        DataFrame contenant uniquement les articles en rupture
    """
    df = data.copy()
    
    # Normalisation
    df[col_similarite] = pd.to_numeric(df[col_similarite], errors='coerce').fillna(0.0).clip(0.0, 1.0)
    
    # Marquer ruptures
    df['is_rupture'] = df[col_similarite] < seuil
    
    # Extraire les lignes en rupture
    ruptures_df = df[df['is_rupture']].copy().reset_index()
    
    return ruptures_df


def plot_distribution_similarite(data: pd.DataFrame, col_similarite: str = 'similarite_max_artice',
                                 source_col: str = 'source') -> go.Figure:
    """
    Crée un histogramme de distribution des scores de similarité.
    
    Args:
        data: DataFrame avec les scores
        col_similarite: Nom de la colonne de similarité
        source_col: Nom de la colonne source
        
    Returns:
        Figure Plotly
    """
    df = data.copy()
    
    # Nettoyage
    df[col_similarite] = pd.to_numeric(df[col_similarite], errors='coerce').fillna(0.0).clip(0.0, 1.0)
    
    unique_sources = df[source_col].astype(str).unique().tolist()
    base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    color_map = {s: base_colors[i % len(base_colors)] for i, s in enumerate(unique_sources)}
    
    traces = []
    for source in unique_sources:
        subset = df[df[source_col].astype(str) == str(source)]
        traces.append(
            go.Histogram(
                x=subset[col_similarite],
                name=str(source),
                marker=dict(color=color_map.get(source)),
                opacity=0.6,
                nbinsx=50,
                histnorm=None
            )
        )
    
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title="Distribution des scores de similarité par source",
        xaxis=dict(title="Score de similarité (cosinus)", range=[0, 1], tickformat=".2f"),
        yaxis=dict(title="Nombre d'articles"),
        barmode='overlay',
        template='plotly_white',
        legend=dict(title="Source", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=80, b=60),
        height=560,
        font=dict(family="DejaVu Sans", size=12)
    )
    
    return fig


def plot_boxplot_similarite(data: pd.DataFrame, col_similarite: str = 'similarite_max_artice',
                           source_col: str = 'source') -> go.Figure:
    """
    Crée des boxplots des scores de similarité par source.
    
    Args:
        data: DataFrame avec les scores
        col_similarite: Nom de la colonne de similarité
        source_col: Nom de la colonne source
        
    Returns:
        Figure Plotly
    """
    df = data.copy()
    
    df[col_similarite] = pd.to_numeric(df[col_similarite], errors='coerce').fillna(0.0).clip(0.0, 1.0)
    df[source_col] = df[source_col].astype(str)
    
    unique_sources = df[source_col].unique().tolist()
    base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    color_map = {s: base_colors[i % len(base_colors)] for i, s in enumerate(unique_sources)}
    
    traces = []
    for src in unique_sources:
        subset = df[df[source_col] == src]
        traces.append(
            go.Box(
                y=subset[col_similarite],
                name=str(src),
                marker_color=color_map.get(src),
                boxmean='sd',
                jitter=0.3,
                pointpos=-1.8,
                boxpoints='outliers',
                hovertemplate="Source: %{name}<br>Score: %{y:.3f}<extra></extra>"
            )
        )
    
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title=dict(text="Boxplots des scores de similarité par source", x=0.5, xanchor='center', font=dict(size=16)),
        yaxis=dict(title="Score de similarité (cosinus)", range=[0, 1], tickformat=".2f"),
        xaxis=dict(title="Source"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=90, b=60),
        height=480,
        font=dict(family="DejaVu Sans", size=12)
    )
    
    return fig


def plot_ruptures_piechart(data: pd.DataFrame, seuil: float = 0.45, 
                          col_similarite: str = 'similarite_max_artice',
                          source_col: str = 'source') -> go.Figure:
    """
    Crée des piecharts montrant la répartition des ruptures.
    
    Args:
        data: DataFrame avec les scores
        seuil: Seuil de rupture
        col_similarite: Nom de la colonne de similarité
        source_col: Nom de la colonne source
        
    Returns:
        Figure Plotly
    """
    df = data.copy()
    
    df[col_similarite] = pd.to_numeric(df[col_similarite], errors='coerce').fillna(0.0).clip(0.0, 1.0)
    df[source_col] = df[source_col].astype(str)
    
    unique_sources = df[source_col].unique().tolist()
    if len(unique_sources) == 0:
        raise RuntimeError("Aucune source détectée dans le DataFrame.")
    elif len(unique_sources) == 1:
        sources_to_plot = [unique_sources[0], unique_sources[0]]
    else:
        sources_to_plot = unique_sources[:2]
    
    src_a, src_b = sources_to_plot[0], sources_to_plot[1]
    
    rupt_a_mask = df[df[source_col] == src_a][col_similarite] < seuil
    rupt_b_mask = df[df[source_col] == src_b][col_similarite] < seuil
    
    count_a_total = int((df[source_col] == src_a).sum())
    count_b_total = int((df[source_col] == src_b).sum())
    count_a_rupt = int(rupt_a_mask.sum())
    count_b_rupt = int(rupt_b_mask.sum())
    count_a_ok = count_a_total - count_a_rupt
    count_b_ok = count_b_total - count_b_rupt
    
    labels = [f"Rupture (< {seuil:.2f})", f"Correspondance (≥ {seuil:.2f})"]
    values_a = [count_a_rupt, count_a_ok]
    values_b = [count_b_rupt, count_b_ok]
    
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=(f"{src_a} (n={count_a_total})", f"{src_b} (n={count_b_total})"))
    
    colors = ['#d62728', '#2ca02c']
    
    fig.add_trace(go.Pie(labels=labels, values=values_a, name=str(src_a),
                         marker=dict(colors=colors), hole=0.35, sort=False,
                         textinfo='percent+label', hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"),
                  row=1, col=1)
    
    fig.add_trace(go.Pie(labels=labels, values=values_b, name=str(src_b),
                         marker=dict(colors=colors), hole=0.35, sort=False,
                         textinfo='percent+label', hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"),
                  row=1, col=2)
    
    fig.update_layout(
        title_text=f"Répartition des ruptures (seuil = {seuil:.2f}) — comparaison {src_a} vs {src_b}",
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=-0.12, xanchor='center', x=0.5),
        height=570,
        margin=dict(l=40, r=40, t=90, b=40)
    )
    
    return fig


def plot_taille_vs_similarite(data: pd.DataFrame, col_similarite: str = 'similarite_max_artice',
                              text_col: str = 'texte_complet', source_col: str = 'source',
                              excerpt_len: int = 200) -> go.Figure:
    """
    Visualise la relation entre taille du texte et score de similarité.
    
    Args:
        data: DataFrame avec les données
        col_similarite: Nom de la colonne de similarité
        text_col: Nom de la colonne de texte
        source_col: Nom de la colonne source
        excerpt_len: Longueur de l'extrait pour le hover
        
    Returns:
        Figure Plotly
    """
    df = data.copy()
    
    df[col_similarite] = pd.to_numeric(df[col_similarite], errors='coerce').fillna(0.0).clip(0.0, 1.0)
    if text_col not in df.columns:
        df[text_col] = ''
    df[text_col] = df[text_col].astype(str)
    df[source_col] = df[source_col].astype(str)
    
    df['char_count'] = df[text_col].str.len()
    df['excerpt'] = df[text_col].str.replace('\n', ' ').str.slice(0, excerpt_len)
    
    x = df['char_count'].values
    y = df[col_similarite].values
    
    x_norm = (x - x.mean()) / (x.std() + 1e-9)
    a, b = np.polyfit(x_norm, y, 1)
    x_line = np.linspace(x_norm.min(), x_norm.max(), 200)
    y_line = a * x_line + b
    x_line_chars = x_line * (x.std() + 1e-9) + x.mean()
    
    unique_sources = df[source_col].unique().tolist()
    base_colors = px.colors.qualitative.Plotly
    color_map = {s: base_colors[i % len(base_colors)] for i, s in enumerate(unique_sources)}
    df['color'] = df[source_col].map(color_map).fillna('#7f7f7f')
    
    fig = go.Figure()
    
    for src in unique_sources:
        sub = df[df[source_col] == src]
        fig.add_trace(
            go.Scatter(
                x=sub['char_count'],
                y=sub[col_similarite],
                mode='markers',
                name=str(src),
                marker=dict(size=8, color=color_map.get(src)),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Source: " + str(src) + "<br>"
                    "Taille (chars): %{x}<br>"
                    "Score: %{y:.3f}<br>"
                    "Extrait: %{customdata}<extra></extra>"
                ),
                text=sub.index.astype(str),
                customdata=sub['excerpt']
            )
        )
    
    fig.add_trace(
        go.Scatter(
            x=x_line_chars,
            y=y_line,
            mode='lines',
            name='Tendance linéaire',
            line=dict(color='black', width=2, dash='dash')
        )
    )
    
    corr = np.corrcoef(x, y)[0,1] if len(x) > 1 else np.nan
    median_len = int(np.median(x)) if len(x) > 0 else 0
    median_score = float(np.median(y)) if len(y) > 0 else 0.0
    
    fig.update_layout(
        title=dict(text="Taille du texte (nombre de caractères) vs Score de similarité", x=0.5),
        xaxis=dict(title="Taille du texte (caractères)", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(title="Score de similarité (cosinus)", range=[0, 1.02], tickformat=".2f", showgrid=True, gridcolor="#f0f0f0"),
        template='plotly_white',
        legend=dict(title="Source", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=90, b=60),
        height=600,
        annotations=[
            dict(
                x=0.01, y=0.99, xref='paper', yref='paper',
                text=f"n = {len(df)} • corr = {corr:.3f} • médiane taille = {median_len} • médiane score = {median_score:.3f}",
                showarrow=False, align='left', font=dict(size=11)
            )
        ]
    )
    
    return fig
