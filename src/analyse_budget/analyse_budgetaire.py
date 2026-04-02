# -*- coding: utf-8 -*-
"""
Module d'analyse budgétaire et conformité avec les piliers SND30
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Tuple


def fusionner_budget_classification(df_budget: pd.DataFrame, 
                                   df_classification: pd.DataFrame,
                                   budget_libelle_col: str = 'libelle',
                                   classif_libelle_col: str = 'Libellé',
                                   source_col: str = 'source') -> pd.DataFrame:
    """
    Fusionne les données budgétaires avec les classifications.
    
    Args:
        df_budget: DataFrame budget avec montants
        df_classification: DataFrame avec classifications
        budget_libelle_col: Nom de la colonne libellé dans budget
        classif_libelle_col: Nom de la colonne libellé dans classification
        source_col: Nom de la colonne source
        
    Returns:
        DataFrame fusionné
    """
    df_final = pd.merge(
        df_budget,
        df_classification[[classif_libelle_col, 'Pilier', source_col]],
        left_on=[budget_libelle_col, source_col],
        right_on=[classif_libelle_col, source_col],
        how='inner'
    )
    
    return df_final


def analyser_conformite_snd30(df_final: pd.DataFrame,
                             source_col: str = 'source',
                             pilier_col: str = 'Pilier',
                             cp_col: str = 'cp_clean',
                             libelle_col: str = 'Libellé') -> pd.DataFrame:
    """
    Analyse la conformité entre fréquence des projets et allocation budgétaire.
    
    Args:
        df_final: DataFrame fusionné budget + classification
        source_col: Nom de la colonne source
        pilier_col: Nom de la colonne pilier
        cp_col: Nom de la colonne montant CP
        libelle_col: Nom de la colonne libellé
        
    Returns:
        DataFrame d'analyse avec parts et gaps
    """
    # Agrégation par Année et par Pilier
    analyse_conformite = df_final.groupby([source_col, pilier_col]).agg({
        libelle_col: 'count',
        cp_col: 'sum'
    }).reset_index()
    
    analyse_conformite.columns = ['Année', 'Pilier_SND30', 'Nombre_Projets', 'Budget_Total_CP']
    
    # Calcul des parts relatives (en %)
    def calculer_parts(groupe):
        total_projets = groupe['Nombre_Projets'].sum()
        total_budget = groupe['Budget_Total_CP'].sum()
        
        groupe['Part_Frequence (%)'] = (groupe['Nombre_Projets'] / total_projets) * 100
        groupe['Part_Budget (%)'] = (groupe['Budget_Total_CP'] / total_budget) * 100
        
        # Le "Gap" mesure l'incohérence
        groupe['Gap_Conformité'] = groupe['Part_Budget (%)'] - groupe['Part_Frequence (%)']
        return groupe
    
    df_analyse = analyse_conformite.groupby('Année').apply(calculer_parts).reset_index(drop=True)
    
    return df_analyse


def calculer_gini(array: np.ndarray) -> float:
    """
    Calcule l'indice de Gini pour un tableau de montants.
    
    Args:
        array: Tableau de montants
        
    Returns:
        Indice de Gini (0 = égalité parfaite, 1 = concentration maximale)
    """
    array = np.array(array, dtype=np.float64)
    array = array[array > 0]
    
    if array.size == 0:
        return 0.0
    
    array = array.flatten()
    array = np.sort(array)
    
    n = len(array)
    index = np.arange(1, n + 1)
    
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def analyse_concentration_par_pilier(df: pd.DataFrame,
                                    source_col: str = 'source',
                                    pilier_col: str = 'Pilier',
                                    cp_col: str = 'cp_clean') -> pd.DataFrame:
    """
    Analyse la concentration budgétaire par pilier avec l'indice de Gini.
    
    Args:
        df: DataFrame avec budget et classification
        source_col: Nom de la colonne source
        pilier_col: Nom de la colonne pilier
        cp_col: Nom de la colonne montant CP
        
    Returns:
        DataFrame avec indices de Gini par pilier
    """
    resultats = []
    
    if 'Pilier_SND30_Predit' in df.columns:
        col_pilier = 'Pilier_SND30_Predit'
    else:
        col_pilier = pilier_col
    
    groupes = df.groupby([source_col, col_pilier])
    
    for (annee, pilier), groupe in groupes:
        col_montant = cp_col if cp_col in groupe.columns else 'cp'
        montants_cp = groupe[col_montant].values
        
        gini_score = calculer_gini(montants_cp)
        
        total_budget = groupe[col_montant].sum()
        nb_projets = len(groupe)
        
        ticket_moyen = total_budget / nb_projets if nb_projets > 0 else 0
        
        resultats.append({
            'Année': annee,
            'Pilier_SND30': pilier,
            'Indice_Gini': round(gini_score, 3),
            'Budget_Total_CP': total_budget,
            'Ticket_Moyen_Projet': round(ticket_moyen, 0),
            'Nombre_Projets': nb_projets
        })
    
    return pd.DataFrame(resultats)


def plot_alignement_budget_frequence(df_analyse: pd.DataFrame) -> go.Figure:
    """
    Crée un scatter plot de l'alignement entre fréquence et budget.
    
    Args:
        df_analyse: DataFrame d'analyse conformité
        
    Returns:
        Figure Plotly
    """
    piliers = df_analyse['Pilier_SND30'].unique()
    colors = px.colors.qualitative.Plotly[:len(piliers)]
    pilier_to_color = dict(zip(piliers, colors))
    
    annees = sorted(df_analyse['Année'].unique())
    symbols = [0, 1, 2, 3, 4, 100, 101, 102][:len(annees)]
    annee_to_symbol = dict(zip(annees, symbols))
    
    fig = go.Figure()
    
    for pilier in piliers:
        subset = df_analyse[df_analyse['Pilier_SND30'] == pilier]
        fig.add_trace(
            go.Scatter(
                x=subset['Part_Frequence (%)'],
                y=subset['Part_Budget (%)'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=pilier_to_color[pilier],
                    symbol=[annee_to_symbol[year] for year in subset['Année']],
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name=pilier,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Fréquence : %{x:.2f}%<br>"
                    "Budget : %{y:.2f}%<br>"
                    "<extra></extra>"
                ),
                text=subset.apply(lambda row: f"{row['Pilier_SND30']} ({row['Année']})", axis=1)
            )
        )
    
    min_val = min(df_analyse['Part_Frequence (%)'].min(), df_analyse['Part_Budget (%)'].min())
    max_val = max(df_analyse['Part_Frequence (%)'].max(), df_analyse['Part_Budget (%)'].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Alignement parfait (x=y)',
            showlegend=True
        )
    )
    
    fig.update_layout(
        title=" Alignement entre Discours (Fréquence) et Budget (Montant)",
        xaxis_title="% du Nombre de Lignes Budgétaires",
        yaxis_title="% du Montant Financier (CP)",
        legend_title="Pilier / Année",
        template="plotly_white",
        height=500,
        width=1200,
        hovermode='closest'
    )
    
    return fig


def plot_alignement_concentration(df_analyse: pd.DataFrame, 
                                 df_gini: pd.DataFrame) -> go.Figure:
    """
    Crée un scatter plot avec taille des bulles selon l'indice de Gini.
    
    Args:
        df_analyse: DataFrame d'analyse conformité
        df_gini: DataFrame avec indices de Gini
        
    Returns:
        Figure Plotly
    """
    df_plot = pd.merge(
        df_analyse,
        df_gini[['Année', 'Pilier_SND30', 'Indice_Gini']],
        on=['Année', 'Pilier_SND30'],
        how='left'
    )
    
    df_plot['Année'] = df_plot['Année'].astype(str).str.strip()
    
    if 'Indice_Gini' not in df_plot.columns:
        df_plot['Indice_Gini'] = 0.1
    else:
        df_plot['Indice_Gini'] = df_plot['Indice_Gini'].fillna(0.1)
    
    annees_uniques = sorted(df_plot['Année'].unique())
    piliers_uniques = sorted(df_plot['Pilier_SND30'].unique())
    
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Bold
    pilier_to_color = {pilier: colors[i % len(colors)] for i, pilier in enumerate(piliers_uniques)}
    
    symboles_dispo = [0, 1, 2, 3, 4, 100, 101, 102]
    annee_to_symbol = {
        annee: symboles_dispo[i % len(symboles_dispo)]
        for i, annee in enumerate(annees_uniques)
    }
    
    fig = go.Figure()
    
    for pilier in piliers_uniques:
        subset = df_plot[df_plot['Pilier_SND30'] == pilier]
        
        tailles = (subset['Indice_Gini'] * 50) + 10
        symboles_subset = [annee_to_symbol[a] for a in subset['Année']]
        
        fig.add_trace(
            go.Scatter(
                x=subset['Part_Frequence (%)'],
                y=subset['Part_Budget (%)'],
                mode='markers',
                marker=dict(
                    size=tailles,
                    sizemode='diameter',
                    color=pilier_to_color[pilier],
                    symbol=symboles_subset,
                    line=dict(width=1, color='DarkSlateGrey'),
                    opacity=0.8
                ),
                name=pilier,
                customdata=subset['Indice_Gini'],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "----------------<br>"
                    " Fréquence (Bruit) : %{x:.2f}%<br>"
                    " Budget (Argent) : %{y:.2f}%<br>"
                    " Concentration (Gini) : %{customdata:.3f}<br>"
                    "<extra></extra>"
                ),
                text=subset.apply(lambda row: f"{row['Pilier_SND30']} ({row['Année']})", axis=1)
            )
        )
    
    max_val = max(df_plot['Part_Frequence (%)'].max(), df_plot['Part_Budget (%)'].max()) + 5
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=1),
            name='Alignement Parfait',
            showlegend=True
        )
    )
    
    fig.update_layout(
        title="<b>Analyse Stratégique : Alignement & Concentration Budgétaire</b>",
        xaxis_title="Part du Discours (% du Nombre de Projets)",
        yaxis_title="Part du Budget Réel (% des CP)",
        legend_title="Piliers SND30",
        template="plotly_white",
        height=600
    )
    
    return fig
