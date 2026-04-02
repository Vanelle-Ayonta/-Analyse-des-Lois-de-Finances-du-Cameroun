import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ---------------------------------------------------------------------------
# Configuration des chemins et import des modules existants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

EXTRACTION_PRETRAIT_DIR = BASE_DIR / "src" / "analyse_budget" 
ANALYSE_CLASSIF_DIR = BASE_DIR / "src" / "analyse_budget" 
ANALYSE_CONFORMITE_DIR = BASE_DIR / "src"  / "analyse_budget" 
ANALYSE_AUDIT_DIR = BASE_DIR / "src" / "analyse_budget" 

for _p in [
    EXTRACTION_PRETRAIT_DIR,
    ANALYSE_CLASSIF_DIR,
    ANALYSE_CONFORMITE_DIR,
    ANALYSE_AUDIT_DIR,
]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.append(str(_p))

# Imports basés sur les modules déjà présents dans le projet

try:
    import pretraitement
except ImportError:
    pretraitement = None

try:
    import classification as classification_mod
except ImportError:
    classification_mod = None

try:
    import analyse_budgetaire
except ImportError:
    analyse_budgetaire = None

try:
    import analyse_semantique
except ImportError:
    analyse_semantique = None

try:
    from extracteur_texte import *
    from extracteur_texte import extraire_articles_loi_finances
except Exception:
    extraire_articles_loi_finances = None

try:
    import extraire_ligne_budgetaire_info
    from extraire_ligne_budgetaire_info import extraire_ligne_par_page
except Exception:
    extraire_ligne_par_page = None

try:
    import extract_budget_info
    from extract_budget_info import (
        extract_budget_info_from_pages,
        extract_all_chapters_summary,
        extract_chapters_with_context,
        save_results_to_json,
        return_resulta_chapter,
    )
except Exception:
    extract_budget_info_from_pages = None
    extract_all_chapters_summary = None
    extract_chapters_with_context = None
    save_results_to_json = None
    return_resulta_chapter = None


AVAILABLE_BUDGET_FILES: Dict[str, str] = {
    "2023-2024": "budget_extract_2023_2024.json",
    "2024-2025": "budget_extract_2024_2025.json",
}

AVAILABLE_ARTICLE_FILES: Dict[str, str] = {
    "2023-2024": "articles_extract_2023_2024.json",
    "2024-2025": "articles_extract_2024_2025.json",
}


# ---------------------------------------------------------------------------
# Style global et helpers UI
# ---------------------------------------------------------------------------

def inject_global_css() -> None:
    """Applique un thème clair moderne avec cartes et sections colorées."""
    st.markdown(
        """
        <style>
        :root {
            --taf-primary: #2563eb;
            --taf-primary-soft: rgba(37, 99, 235, 0.08);
            --taf-bg: #f5f5f7;
            --taf-bg-soft: #ffffff;
            --taf-border-soft: rgba(148, 163, 184, 0.7);
            --taf-accent: #f97316;
            --taf-text-main: #111827;
            --taf-text-soft: #6b7280;
        }

        /* Arrière-plan global et couleur de texte */
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, rgba(37, 99, 235, 0.07), transparent 50%),
                        radial-gradient(circle at bottom right, rgba(249, 115, 22, 0.05), transparent 55%),
                        var(--taf-bg);
            color: var(--taf-text-main);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f3f4f6 45%, #e5e7eb 100%);
            border-right: 1px solid rgba(209, 213, 219, 0.9);
        }
        [data-testid="stSidebar"] * {
            color: var(--taf-text-main);
        }

        /* Titre & sous-titres */
        h1, h2, h3, h4 {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        /* Hero header principal */
        .taf-hero {
            padding: 1.6rem 1.4rem 1.1rem 1.4rem;
            border-radius: 1.4rem;
            border: 1px solid var(--taf-border-soft);
            background:
                radial-gradient(circle at top left, rgba(191, 219, 254, 0.8), transparent 55%),
                linear-gradient(120deg, rgba(37, 99, 235, 0.9), rgba(30, 64, 175, 0.9));
            box-shadow:
                0 18px 45px rgba(15, 23, 42, 0.2),
                0 0 0 1px rgba(15, 23, 42, 0.05);
            margin-bottom: 1.5rem;
        }
        .taf-hero-main h1 {
            margin: 0 0 0.4rem 0;
            font-size: 1.6rem;
            color: #f9fafb;
        }
        .taf-hero-main p {
            margin: 0;
            color: #e5e7eb;
            font-size: 0.98rem;
        }
        .taf-hero-chip {
            margin-top: 0.85rem;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.24rem 0.9rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(191, 219, 254, 0.9);
            color: #e5e7eb;
            font-size: 0.8rem;
            letter-spacing: 0.02em;
        }

        /* Cartes metrics */
        [data-testid="stMetric"] {
            background: radial-gradient(circle at top left, rgba(219, 234, 254, 0.9), #ffffff);
            padding: 0.95rem 1.15rem;
            border-radius: 1.1rem;
            border: 1px solid rgba(191, 219, 254, 0.9);
            box-shadow: 0 16px 32px rgba(15, 23, 42, 0.08);
        }
        [data-testid="stMetric"] > div {
            color: var(--taf-text-main);
        }

        /* En-têtes de section (par onglet) */
        .taf-section-header {
            margin: 0 0 1.0rem 0;
            padding: 0.9rem 1.1rem;
            border-radius: 0.9rem;
            background: #ffffff;
            border: 1px solid rgba(209, 213, 219, 0.9);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }
        .taf-section-header-left h2 {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--taf-text-main);
        }
        .taf-section-header-left p {
            margin: 0.18rem 0 0 0;
            font-size: 0.83rem;
            color: var(--taf-text-soft);
        }
        .taf-section-badge {
            padding: 0.2rem 0.85rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 500;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            border: 1px solid transparent;
        }
        .taf-section-badge--overview {
            background: rgba(59, 130, 246, 0.08);
            color: #1d4ed8;
            border-color: rgba(59, 130, 246, 0.45);
        }
        .taf-section-badge--extraction {
            background: rgba(14, 159, 110, 0.08);
            color: #047857;
            border-color: rgba(16, 185, 129, 0.5);
        }
        .taf-section-badge--pretraitement {
            background: rgba(14, 116, 144, 0.08);
            color: #0369a1;
            border-color: rgba(14, 165, 233, 0.5);
        }
        .taf-section-badge--classification {
            background: rgba(219, 39, 119, 0.08);
            color: #be185d;
            border-color: rgba(236, 72, 153, 0.55);
        }
        .taf-section-badge--analyse {
            background: rgba(180, 83, 9, 0.08);
            color: #b45309;
            border-color: rgba(234, 179, 8, 0.6);
        }
        .taf-section-badge--audit {
            background: rgba(88, 28, 135, 0.08);
            color: #6b21a8;
            border-color: rgba(168, 85, 247, 0.55);
        }
        .taf-section-badge--apropos {
            background: rgba(31, 41, 55, 0.06);
            color: #111827;
            border-color: rgba(75, 85, 99, 0.5);
        }

        /* Tables (dataframes) */
        [data-testid="stDataFrame"] {
            border-radius: 1.1rem;
            overflow: hidden;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
            border: 1px solid rgba(209, 213, 219, 0.9);
            background: #ffffff;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            border-radius: 999px !important;
            padding: 0.18rem 0.95rem !important;
            font-size: 0.88rem !important;
            border: 0 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: var(--taf-primary) !important;
            color: #ecfeff !important;
        }
        button[data-baseweb="tab"][aria-selected="false"] {
            background-color: transparent !important;
            color: var(--taf-text-soft) !important;
        }

        /* Inputs texte */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            background-color: #ffffff !important;
            border-radius: 0.75rem !important;
            border: 1px solid rgba(209, 213, 219, 0.9) !important;
            color: var(--taf-text-main) !important;
        }

        /* Sliders */
        [data-testid="stSlider"] > div > div > div {
            color: var(--taf-text-main);
        }

        /* Download button & primary buttons */
        [data-testid="baseButton-secondary"] {
            border-radius: 999px !important;
        }
        [data-testid="baseButton-primary"] {
            border-radius: 999px !important;
            box-shadow: 0 14px 30px rgba(37, 99, 235, 0.35);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_json_as_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        return pd.DataFrame(data)
    except ValueError:
        return pd.json_normalize(data)


@st.cache_data(show_spinner=False)
def load_default_datasets() -> Dict[str, Any]:
    budget_dfs: Dict[str, pd.DataFrame] = {}
    for label, filename in AVAILABLE_BUDGET_FILES.items():
        df = load_json_as_df(BASE_DIR / "data" / filename)
        if not df.empty:
            df = df.copy()
            df["source"] = label
            budget_dfs[label] = df

    article_dfs: Dict[str, pd.DataFrame] = {}
    for label, filename in AVAILABLE_ARTICLE_FILES.items():
        df = load_json_as_df(BASE_DIR / "data" / filename)
        if not df.empty:
            df = df.copy()
            df["source"] = label
            article_dfs[label] = df

    return {
        "budget": budget_dfs,
        "articles": article_dfs,
    }


def get_concat_df(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs.values(), ignore_index=True)


def display_df_with_download(df: pd.DataFrame, label: str) -> None:
    """Propose uniquement le téléchargement des données, sans affichage de tableau."""
    if df.empty:
        st.info("Aucune donnée disponible à télécharger pour cette section.")
        return
    st.caption(f"{len(df):,} lignes prêtes pour analyse détaillée hors dashboard.".replace(",", " "))
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"Télécharger {label} (CSV)",
        data=csv,
        file_name=f"{label.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )


def parse_amount_to_float(val: Any) -> float:
    if isinstance(val, str):
        clean = val.replace(" ", "").replace(".", "").replace(",", ".")
        try:
            return float(clean)
        except Exception:
            return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    return 0.0


def render_section_header(title: str, subtitle: str, badge_label: str, badge_variant: str) -> None:
    """Affiche un en-tête de section avec badge coloré par onglet."""
    badge_class = f"taf-section-badge--{badge_variant}"
    st.markdown(
        f"""
        <div class="taf-section-header">
            <div class="taf-section-header-left">
                <h2>{title}</h2>
                <p>{subtitle}</p>
            </div>
            <div class="taf-section-header-right">
                <span class="taf-section-badge {badge_class}">{badge_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pages du tableau de bord
# ---------------------------------------------------------------------------

def page_overview(datasets: Dict[str, Any]) -> None:
    render_section_header(
        title="Vue d’ensemble",
        subtitle="Vue synthétique des volumes budgétaires et des articles extraits par exercice ",
        badge_label="Synthèse",
        badge_variant="overview",
    )

    budget_dfs: Dict[str, pd.DataFrame] = datasets["budget"]
    article_dfs: Dict[str, pd.DataFrame] = datasets["articles"]

    col1, col2, col3 = st.columns(3)

    with col1:
        total_budget_lignes = sum(len(df) for df in budget_dfs.values())
        st.metric("Nombre total de lignes budgétaires", f"{total_budget_lignes:,}".replace(",", " "))

    with col2:
        total_articles = sum(len(df) for df in article_dfs.values())
        st.metric("Nombre total d’articles extraits", f"{total_articles:,}".replace(",", " "))

    with col3:
        annees = sorted(set(list(budget_dfs.keys()) + list(article_dfs.keys())))
        st.metric("Période couverte", " – ".join(annees) if annees else "N/A")

   
    df_budget_all = get_concat_df(budget_dfs)
    df_articles_all = get_concat_df(article_dfs)

    if not df_budget_all.empty:
        st.markdown("### Synthèse budgétaire (CP par année)")
        agg = (
            df_budget_all.groupby("source")["cp"]
            .apply(lambda s: s.astype(str))
            .reset_index()
        )
        agg["cp_float"] = agg["cp"].apply(parse_amount_to_float)
        synth = agg.groupby("source")["cp_float"].sum().reset_index()
        synth.rename(columns={"cp_float": "CP_total"}, inplace=True)
        synth["CP_total"] = synth["CP_total"].round(0)
        with st.expander("Télécharger la synthèse budgétaire (CSV)"):
            display_df_with_download(synth, "synthese_budget_par_annee")
        

        # Pie chart répartition CP par exercice
        fig_pie_cp = px.pie(
            synth, values="CP_total", names="source",
            title="Répartition du budget CP par exercice",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie_cp.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie_cp, use_container_width=True)

    if not df_articles_all.empty:
        if "chapitre_numero" in df_articles_all.columns:
            top_n = st.slider(
                "Nombre de chapitres à afficher (par nombre d’articles)",
                min_value=5,
                max_value=30,
                value=10,
            )
            chap_counts = (
                df_articles_all.groupby(["source", "chapitre_numero"])
                .size()
                .reset_index(name="nb_articles")
            )
            chap_top = (
                chap_counts.sort_values("nb_articles", ascending=False)
                .head(top_n)
            )
            fig_chap = px.bar(
                chap_top,
                x="nb_articles",
                y="chapitre_numero",
                color="source",
                orientation="h",
                labels={
                    "nb_articles": "Nombre d’articles",
                    "chapitre_numero": "Chapitre",
                    "source": "Exercice",
                },
                title="Top chapitres par nombre d’articles extraits",
            )
            fig_chap.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_chap, use_container_width=True)

            # Pie chart top chapitres par nombre d'articles
            chap_pie = chap_counts.sort_values("nb_articles", ascending=False).head(8)
            fig_pie_art = px.pie(
                chap_pie, values="nb_articles", names="chapitre_numero",
                title="Répartition des articles par chapitre (top 8)",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_pie_art.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie_art, use_container_width=True)

        with st.expander("Télécharger les données articles (CSV)"):
            display_df_with_download(df_articles_all, "echantillon_articles")


def page_extraction(datasets: Dict[str, Any]) -> None:
    render_section_header(
        title="Extraction et données brutes",
        subtitle="Chargement des JSON existants ou extraction directe depuis les PDF de lois de finances.",
        badge_label="Données sources",
        badge_variant="extraction",
    )

    tabs = st.tabs(["Budget (projets)", "Articles de loi", "Extraction avancée (PDF)"])

    budget_dfs: Dict[str, pd.DataFrame] = datasets["budget"]
    article_dfs: Dict[str, pd.DataFrame] = datasets["articles"]

    with tabs[0]:
        st.markdown("#### Budgets extraits (JSON → DataFrame)")
        if not budget_dfs:
            st.info("Aucun fichier JSON de budget détecté à la racine du projet.")
        else:
            annees = sorted(budget_dfs.keys())
            annee = st.selectbox("Choisir l’année budgétaire", annees, key="budget_extraction_year")
            df = budget_dfs[annee]
            st.metric("Lignes budgétaires", len(df))

            if "cp" in df.columns and "chapitre" in df.columns:
                st.markdown("##### CP par chapitre (vue rapide)")
                tmp = df.copy()
                tmp["cp_float"] = tmp["cp"].astype(str).apply(parse_amount_to_float)
                chap_budget = (
                    tmp.groupby("chapitre")["cp_float"]
                    .sum()
                    .reset_index()
                    .sort_values("cp_float", ascending=False)
                    .head(20)
                )
                fig = px.bar(
                    chap_budget,
                    x="cp_float",
                    y="chapitre",
                    orientation="h",
                    labels={
                        "cp_float": "CP total (converti)",
                        "chapitre": "Chapitre",
                    },
                    title=f"Top 20 chapitres par CP – {annee}",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

                # Pie chart CP par chapitre (top 10)
                chap_pie_b = chap_budget.head(10)
                fig_pie_b = px.pie(
                    chap_pie_b, values="cp_float", names="chapitre",
                    title=f"Répartition CP par chapitre – {annee} (top 10)",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                fig_pie_b.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie_b, use_container_width=True)

            with st.expander("Télécharger les données budget (CSV)"):
                display_df_with_download(df, f"budget_{annee}")

    with tabs[1]:
        st.markdown("#### Articles de loi extraits")
        if not article_dfs:
            st.info("Aucun fichier JSON d’articles détecté à la racine du projet.")
        else:
            annees = sorted(article_dfs.keys())
            annee = st.selectbox("Choisir l’année des articles", annees, key="articles_extraction_year")
            df = article_dfs[annee]
            st.metric("Articles extraits", len(df))
            if "chapitre_numero" in df.columns:
                st.markdown("##### Répartition des articles par chapitre")
                chap_year = (
                    df.groupby("chapitre_numero")
                    .size()
                    .reset_index(name="nb_articles")
                    .sort_values("nb_articles", ascending=False)
                    .head(20)
                )
                fig = px.bar(
                    chap_year,
                    x="nb_articles",
                    y="chapitre_numero",
                    orientation="h",
                    labels={
                        "nb_articles": "Nombre d’articles",
                        "chapitre_numero": "Chapitre",
                    },
                    title=f"Top 20 chapitres par nombre d’articles – {annee}",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

                # Pie chart articles par chapitre (top 10)
                chap_pie_a = chap_year.head(10)
                fig_pie_a = px.pie(
                    chap_pie_a, values="nb_articles", names="chapitre_numero",
                    title=f"Répartition des articles par chapitre – {annee} (top 10)",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_pie_a.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie_a, use_container_width=True)

            with st.expander("Télécharger les articles (CSV)"):
                display_df_with_download(df, f"articles_{annee}")

    with tabs[2]:
        st.markdown("#### Extraction directe depuis un PDF de loi de finances")
        st.caption(
            "Ces fonctionnalités s’appuient sur les modules d’extraction existants "
            "(pdfplumber, PyMuPDF, OpenAI, etc.). Elles peuvent être **longues** et "
            "nécessitent que les dépendances et clés API soient correctement configurées."
        )

        uploaded_pdf = st.file_uploader(
            "Uploader un PDF de loi de finances (facultatif, pour une nouvelle extraction)",
            type=["pdf"],
        )

        if uploaded_pdf is not None:
            tmp_pdf_path = BASE_DIR / "tmp_uploaded_loi_finances.pdf"
            with tmp_pdf_path.open("wb") as f:
                f.write(uploaded_pdf.getbuffer())
            st.success(f"Fichier sauvegardé temporairement sous `{tmp_pdf_path.name}`.")
        else:
            tmp_pdf_path = None

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### Extraction des **articles de loi** (pdfplumber)")
            if extraire_articles_loi_finances is None:
                st.warning("Le module `extracteur_texte.py` n’a pas pu être importé.")
            else:
                if st.button("Lancer l’extraction des articles depuis le PDF", type="primary", disabled=tmp_pdf_path is None):
                    if tmp_pdf_path is None:
                        st.error("Aucun PDF chargé.")
                    else:
                        with st.spinner("Extraction des articles en cours..."):
                            try:
                                articles = extraire_articles_loi_finances(str(tmp_pdf_path))
                                df_articles = pd.DataFrame(articles)
                                st.write(f"Extraction terminée : **{len(df_articles)}** articles trouvés.")
                                display_df_with_download(df_articles, "articles_extraits_depuis_pdf")
                            except Exception as e:
                                st.error(f"Erreur lors de l’extraction des articles : {e}")

        with col_b:
            st.markdown("##### Extraction **budgétaire détaillée** (OpenAI Vision)")
            if extract_budget_info_from_pages is None:
                st.warning("Le module `extract_budget_info.py` n’a pas pu être importé ou dépendances manquantes.")
            else:
                pages_str = st.text_input(
                    "Pages à extraire (ex: 80-95 ou 87,88,89)",
                    value="80-95",
                )

                if st.button(
                    "Lancer l’extraction budgétaire avancée (coûteux, nécessite OPENAI_API_KEY)",
                    disabled=tmp_pdf_path is None,
                ):
                    if tmp_pdf_path is None:
                        st.error("Aucun PDF chargé.")
                    else:
                        try:
                            pages: List[int] = []
                            if "-" in pages_str:
                                debut, fin = pages_str.split("-")
                                pages = list(range(int(debut), int(fin) + 1))
                            else:
                                pages = [int(p.strip()) for p in pages_str.split(",") if p.strip()]
                        except Exception:
                            st.error("Format de pages invalide.")
                            return

                        with st.spinner("Extraction budgétaire avancée en cours..."):
                            try:
                                results = extract_budget_info_from_pages(str(tmp_pdf_path), pages)
                                df_budget = pd.DataFrame(results)
                                st.write(f"Extraction terminée : **{len(df_budget)}** lignes budgétaires trouvées.")
                                display_df_with_download(df_budget, "budget_extrait_depuis_pdf")
                            except Exception as e:
                                st.error(f"Erreur lors de l’extraction budgétaire : {e}")


def page_pretraitement(datasets: Dict[str, Any]) -> None:
    render_section_header(
        title="Prétraitement des textes",
        subtitle="Nettoyage fin des articles et des libellés budgétaires avant analyse de contenu.",
        badge_label="Nettoyage",
        badge_variant="pretraitement",
    )

    if pretraitement is None:
        st.error("Le module `pretraitement.py` n’a pas pu être importé.")
        return

    article_dfs: Dict[str, pd.DataFrame] = datasets["articles"]
    df_articles_all = get_concat_df(article_dfs)

    mode = st.radio(
        "Source du texte",
        [ "Articles extraits (JSON)", "Saisie manuelle"],
        horizontal=True,
    )

    if mode == "Articles extraits (JSON)":
        if df_articles_all.empty:
            st.info("Aucun article disponible. Utiliser d’abord l’extraction ou fournir un JSON.")
            return

        st.markdown("Sélectionner un échantillon d’articles pour visualiser le nettoyage.")
        nb_samples = st.slider("Nombre d’articles à afficher", min_value=10, max_value=min(200, len(df_articles_all)), value=min(50, len(df_articles_all)))
        sample_df = df_articles_all.sample(min(nb_samples, len(df_articles_all)), random_state=42)

        cleaned_list = pretraitement.pretraiter_liste_articles(sample_df["texte_complet"].tolist())
        sample_df = sample_df.copy()
        sample_df["texte_nettoye"] = cleaned_list
        sample_df["len_brut"] = sample_df["texte_complet"].astype(str).str.len()
        sample_df["len_nettoye"] = sample_df["texte_nettoye"].astype(str).str.len()

        st.metric("Articles traités", len(sample_df))
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Longueur moyenne (brut)", f"{sample_df['len_brut'].mean():.0f} car.")
        with r2:
            st.metric("Longueur moyenne (nettoyé)", f"{sample_df['len_nettoye'].mean():.0f} car.")

        fig_pret = px.scatter(
            sample_df, x="len_brut", y="len_nettoye",
            labels={"len_brut": "Longueur brut (car.)", "len_nettoye": "Longueur nettoyé (car.)"},
            title="Impact du prétraitement : longueur brut vs longueur nettoyée",
        )
        _mx = sample_df["len_brut"].max()
        if _mx > 0:
            fig_pret.add_shape(type="line", x0=0, y0=0, x1=_mx, y1=_mx, line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig_pret, use_container_width=True)

        with st.expander("Télécharger l'échantillon prétraité (CSV)"):
            display_df_with_download(sample_df[["chapitre_numero", "chapitre_titre", "len_brut", "len_nettoye"]], "pretraitement_echantillon")


    else:
        texte_brut = st.text_area(
            "Texte brut de loi / article à nettoyer",
            height=200,
            placeholder="Coller ici un article de Loi de Finances camerounaise...",
        )
        if st.button("Nettoyer le texte"):
            if not texte_brut.strip():
                st.warning("Merci de saisir un texte.")
            else:
                with st.spinner("Prétraitement en cours..."):
                    propre = pretraitement.pretraiter_texte_loi_finances(texte_brut)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Texte brut**")
                    st.code(texte_brut)
                with col2:
                    st.markdown("**Texte prétraité**")
                    st.code(propre)
        
def page_classification(datasets: Dict[str, Any]) -> None:
    render_section_header(
        title="Classification SND30",
        subtitle="Attribution des projets et articles aux piliers stratégiques de la SND30.",
        badge_label="Scoring SND30",
        badge_variant="classification",
    )

    if classification_mod is None:
        st.error("Le module `classification.py` n’a pas pu être importé.")
        return

    budget_dfs: Dict[str, pd.DataFrame] = datasets["budget"]
    article_dfs: Dict[str, pd.DataFrame] = datasets["articles"]

    tabs = st.tabs(["Projets budgétaires", "Articles de loi"])

    with tabs[0]:
        if not budget_dfs:
            st.info("Aucun budget JSON détecté. Fournir un fichier ou utiliser l’extraction PDF.")
        else:
            st.markdown("#### Classification des libellés budgétaires par pilier SND30")
            annees_sel = st.multiselect(
                "Années à inclure dans la classification",
                options=sorted(budget_dfs.keys()),
                default=sorted(budget_dfs.keys()),
            )
            if not annees_sel:
                st.warning("Sélectionner au moins une année.")
            else:
                df_budget_all = get_concat_df({a: budget_dfs[a] for a in annees_sel})
                st.write(f"{len(df_budget_all)} lignes budgétaires sélectionnées.")

                col_opts = st.columns(3)
                with col_opts[0]:
                    batch_size = st.slider("Batch size", min_value=4, max_value=32, value=16, step=4)
                with col_opts[1]:
                    device = st.selectbox("Device", options=["auto", "CPU (-1)", "GPU (0)"], index=0)
                    dev_param: Any = "auto"
                    if device == "CPU (-1)":
                        dev_param = -1
                    elif device == "GPU (0)":
                        dev_param = 0
                with col_opts[2]:
                    max_rows = st.slider(
                        "Limiter le nombre de lignes à classer (pour tests)",
                        min_value=50,
                        max_value=min(2000, len(df_budget_all)),
                        value=min(500, len(df_budget_all)),
                    )

                if st.button("Lancer la classification des libellés (15 secondes pour 50 lignes avec batch size de 16)", type="primary"):
                    subset = df_budget_all.head(max_rows).copy()
                    libelles = subset["libelle"].astype(str).tolist()
                    with st.spinner("Classification zero-shot en cours..."):
                        df_classif = classification_mod.classer_ligne_dep_snd30(
                            liste_libelles=libelles,
                            batch_size=batch_size,
                            device=dev_param,
                            modele_dir=str(BASE_DIR / "modele"),
                        )
                    subset = subset.reset_index(drop=True)
                    df_classif = df_classif.reset_index(drop=True)
                    df_merged = pd.concat([subset, df_classif], axis=1)

                    # Ajouter colonnes nettoyées (sans écraser source pour garder l’année par ligne)
                    if pretraitement is not None and "cp" in df_merged.columns:
                        df_merged["cp_clean"] = df_merged["cp"].astype(str).apply(parse_amount_to_float)
                    if pretraitement is not None and "ae" in df_merged.columns:
                        df_merged["ae_clean"] = df_merged["ae"].astype(str).apply(parse_amount_to_float)

                    st.session_state["df_budget_classif"] = df_merged

                    st.success("Classification terminée.")

                    # KPIs
                    k1, k2, k3 = st.columns(3)
                    with k1:
                        st.metric("Lignes classifiées", len(df_merged))
                    with k2:
                        st.metric("Piliers SND30", df_merged["Pilier"].nunique() if "Pilier" in df_merged.columns else 0)
                    with k3:
                        sc = df_merged["Score"].mean() if "Score" in df_merged.columns else 0
                        st.metric("Score moyen", f"{sc:.3f}")

                    with st.expander("Télécharger la classification (CSV)"):
                        display_df_with_download(df_merged, "classification_lignes_budgetaires")

                    # Visualisations si colonnes adéquates
                    if "source" in df_merged.columns:
                        df_plot = df_merged.rename(columns={"Libellé": "Libellé"})
                    else:
                        df_plot = df_merged.copy()
                        df_plot["source"] = ";".join(annees_sel)

                    fig_rep = classification_mod.plot_repartition_piliers_par_annee(df_plot, source_col="source")
                    st.plotly_chart(fig_rep, use_container_width=True)

                    if "Score" in df_plot.columns:
                        fig_hist = classification_mod.plot_distribution_scores_classification(df_plot, source_col="source")
                        st.plotly_chart(fig_hist, use_container_width=True)

                        fig_box = classification_mod.plot_boxplot_scores_par_pilier(df_plot, source_col="source")
                        st.plotly_chart(fig_box, use_container_width=True)

                    # Wordclouds par pilier (comme dans le notebook)
                    if hasattr(classification_mod, "plot_wordcloud_projets_par_pilier"):
                        st.markdown("#### Nuages de mots par pilier SND30")
                        for annee in annees_sel:
                            df_annee = df_merged[df_merged["source"] == annee] if "source" in df_merged.columns else df_merged
                            if not df_annee.empty and "Pilier" in df_annee.columns:
                                try:
                                    df_annee = df_annee.copy()
                                    df_annee["source"] = annee
                                    fig_wc = classification_mod.plot_wordcloud_projets_par_pilier(
                                        df_annee, annee=annee, libelle_col="libelle", pilier_col="Pilier", source_col="source"
                                    )
                                    st.pyplot(fig_wc)
                                    import matplotlib.pyplot as plt
                                    plt.close(fig_wc)
                                except Exception as e:
                                    st.caption(f"Wordcloud {annee}: {e}")

    with tabs[1]:
        if not article_dfs:
            st.info("Aucun JSON d’articles détecté.")
        else:
            st.markdown("#### Classification des articles de loi par pilier SND30")
            annees_sel_art = st.multiselect(
                "Années à inclure",
                options=sorted(article_dfs.keys()),
                default=sorted(article_dfs.keys()),
                key="classification_articles_years",
            )
            if not annees_sel_art:
                st.warning("Sélectionner au moins une année.")
            else:
                df_articles_all = get_concat_df({a: article_dfs[a] for a in annees_sel_art})
                st.write(f"{len(df_articles_all)} articles sélectionnés.")

                col_opts = st.columns(3)
                with col_opts[0]:
                    batch_size_art = st.slider("Batch size", min_value=4, max_value=32, value=16, step=4, key="batch_articles")
                with col_opts[1]:
                    device_art = st.selectbox(
                        "Device",
                        options=["auto", "CPU (-1)", "GPU (0)"],
                        index=0,
                        key="device_articles",
                    )
                    dev_param_art: Any = "auto"
                    if device_art == "CPU (-1)":
                        dev_param_art = -1
                    elif device_art == "GPU (0)":
                        dev_param_art = 0
                with col_opts[2]:
                    max_rows_art = st.slider(
                        "Limiter le nombre d’articles à classer",
                        min_value=20,
                        max_value=min(1000, len(df_articles_all)),
                        value=min(200, len(df_articles_all)),
                        key="max_rows_articles",
                    )

                if st.button("Lancer la classification des articles (15 secondes pour 50 articles avec batch size de 16)", type="primary"):
                    subset_art = df_articles_all.head(max_rows_art).copy()
                    textes = subset_art["texte_complet"].astype(str).tolist()

                    with st.spinner("Classification zero-shot des articles en cours..."):
                        df_art_classif = classification_mod.classer_articles_snd30(
                            liste_articles=textes,
                            batch_size=batch_size_art,
                            device=dev_param_art,
                            modele_dir=str(BASE_DIR / "modele"),
                        )

                    subset_art = subset_art.reset_index(drop=True)
                    df_art_classif = df_art_classif.reset_index(drop=True)
                    df_merged_art = pd.concat([subset_art, df_art_classif], axis=1)
                    if "source" not in df_merged_art.columns:
                        df_merged_art["source"] = ";".join(annees_sel_art)

                    st.session_state["df_articles_classif"] = df_merged_art

                    st.success("Classification des articles terminée.")

                    ka1, ka2, ka3 = st.columns(3)
                    with ka1:
                        st.metric("Articles classifiés", len(df_merged_art))
                    with ka2:
                        st.metric("Piliers SND30", df_merged_art["Pilier"].nunique() if "Pilier" in df_merged_art.columns else 0)
                    with ka3:
                        sca = df_merged_art["Score"].mean() if "Score" in df_merged_art.columns else 0
                        st.metric("Score moyen", f"{sca:.3f}")

                    with st.expander("Télécharger la classification des articles (CSV)"):
                        display_df_with_download(df_merged_art, "classification_articles_loi")

                    fig_rep_art = classification_mod.plot_repartition_articles_piliers(
                        df_merged_art,
                        source_col="source",
                        pilier_col="Pilier",
                    )
                    st.plotly_chart(fig_rep_art, use_container_width=True)

                    if "Score" in df_merged_art.columns:
                        fig_hist_art = classification_mod.plot_distribution_scores_articles(
                            df_merged_art,
                            source_col="source",
                            score_col="Score",
                        )
                        st.plotly_chart(fig_hist_art, use_container_width=True)

                        fig_box_art = classification_mod.plot_boxplot_scores_articles(
                            df_merged_art,
                            source_col="source",
                            pilier_col="Pilier",
                            score_col="Score",
                        )
                        st.plotly_chart(fig_box_art, use_container_width=True)

                    # Wordclouds articles par pilier (comme dans le notebook)
                    if hasattr(classification_mod, "plot_wordcloud_articles_par_pilier"):
                        st.markdown("#### Nuages de mots des articles par pilier SND30")
                        for annee in annees_sel_art:
                            df_art_annee = df_merged_art[df_merged_art["source"] == annee] if "source" in df_merged_art.columns else df_merged_art
                            if not df_art_annee.empty and "Pilier" in df_art_annee.columns:
                                try:
                                    fig_wc_art = classification_mod.plot_wordcloud_articles_par_pilier(
                                        df_art_annee, annee=annee, article_col="texte_complet", pilier_col="Pilier", source_col="source"
                                    )
                                    st.pyplot(fig_wc_art)
                                    import matplotlib.pyplot as plt
                                    plt.close(fig_wc_art)
                                except Exception as e:
                                    st.caption(f"Wordcloud articles {annee}: {e}")


def page_analyse_budget_conformite(datasets: Dict[str, Any]) -> None:
    render_section_header(
        title="Analyse budgétaire & conformité",
        subtitle="Comparaison entre poids budgétaire et fréquence des projets pour chaque pilier SND30.",
        badge_label="Alignement & Gini",
        badge_variant="analyse",
    )

    if analyse_budgetaire is None:
        st.error("Le module `analyse_budgetaire.py` n’a pas pu être importé.")
        return

    if "df_budget_classif" not in st.session_state:
        st.info(
            "Aucune classification budgétaire en mémoire. "
            "Commence par la page **Classification SND30** pour générer un DataFrame fusionné, "
            "ou importe un fichier externe."
        )

    # use_session = st.checkbox(
    #     "Utiliser la classification calculée dans ce tableau de bord (si disponible)",
    #     value="df_budget_classif" in st.session_state,
    # #)

        
    use_session = st.checkbox(
        "Utiliser la classification calculée dans ce tableau de bord (si disponible)",
        value=True,  # Toujours coché par défaut
    )
    
    if use_session and "df_budget_classif" in st.session_state:
        df_final_base = st.session_state["df_budget_classif"].copy()
    else:
        uploaded = st.file_uploader(
            "Uploader un fichier de classification fusionnée (CSV/JSON) "
            "contenant au moins les colonnes `libelle`, `cp`/`cp_clean`, `Pilier`, `source`.",
            type=["csv", "json"],
        )
        if uploaded is None:
            st.stop()
        suffix = Path(uploaded.name).suffix.lower()
        if suffix == ".csv":
            df_final_base = pd.read_csv(uploaded)
        else:
            df_final_base = pd.read_json(uploaded)

    # KPIs conformité
    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        st.metric("Lignes analysées", len(df_final_base))
    with kc2:
        pil_count = df_final_base["Pilier"].nunique() if "Pilier" in df_final_base.columns else 0
        st.metric("Piliers SND30", pil_count)
    with kc3:
        _cp = "cp_clean" if "cp_clean" in df_final_base.columns else "cp"
        if _cp in df_final_base.columns:
            _tot = df_final_base[_cp].apply(parse_amount_to_float).sum()
            st.metric("Budget total (CP)", f"{_tot:,.0f}".replace(",", " "))
        else:
            st.metric("Budget total", "N/A")
    with kc4:
        st.metric("Exercices", df_final_base["source"].nunique() if "source" in df_final_base.columns else "N/A")

    # S’assurer que les colonnes de montants nettoyés existent
    if pretraitement is not None and "cp_clean" not in df_final_base.columns:
        df_final_base = pretraitement.preparer_donnees_budget(
            df_final_base,
            source=df_final_base.get("source", "N/A").iloc[0] if not df_final_base.empty else "N/A",
        )

    df_analyse = analyse_budgetaire.analyser_conformite_snd30(
        df_final_base,
        source_col="source",
        pilier_col="Pilier",
        cp_col="cp_clean" if "cp_clean" in df_final_base.columns else "cp",
        libelle_col="Libellé" if "Libellé" in df_final_base.columns else "libelle",
    )

    df_gini = analyse_budgetaire.analyse_concentration_par_pilier(
        df_final_base,
        source_col="source",
        pilier_col="Pilier",
        cp_col="cp_clean" if "cp_clean" in df_final_base.columns else "cp",
    )

    # Donut répartition budget par pilier (moyenne des années)
    pilier_budget = df_analyse.groupby("Pilier_SND30")["Budget_Total_CP"].sum().reset_index()
    if not pilier_budget.empty:
        fig_donut = px.pie(
            pilier_budget, values="Budget_Total_CP", names="Pilier_SND30",
            title="Répartition du budget par pilier SND30",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_donut, use_container_width=True)

    fig_align = analyse_budgetaire.plot_alignement_budget_frequence(df_analyse)
    st.plotly_chart(fig_align, use_container_width=True)

    fig_align_conc = analyse_budgetaire.plot_alignement_concentration(df_analyse, df_gini)
    st.plotly_chart(fig_align_conc, use_container_width=True)

    with st.expander("Télécharger les analyses (CSV)"):
        display_df_with_download(df_analyse, "analyse_conformite_snd30")
        display_df_with_download(df_gini, "analyse_concentration_par_pilier")


def page_audit_semantique(datasets: Dict[str, Any]) -> None:
    render_section_header(
        title="Audit sémantique",
        subtitle="Analyse des similarités et ruptures de contenu entre deux exercices de lois de finances.",
        badge_label="Comparaison texte à texte",
        badge_variant="audit",
    )

    if analyse_semantique is None:
        st.error("Le module `analyse_semantique.py` n’a pas pu être importé.")
        return

    article_dfs: Dict[str, pd.DataFrame] = datasets["articles"]

    if not article_dfs:
        st.info("Aucun JSON d’articles détecté. Fournir au moins deux jeux d’articles.")
        return

    annees = sorted(article_dfs.keys())
    col1, col2 = st.columns(2)
    with col1:
        annee_a = st.selectbox("Année A (référence)", options=annees, index=0)
    with col2:
        annee_b = st.selectbox("Année B (comparée)", options=annees, index=min(1, len(annees) - 1))

    df_a = article_dfs[annee_a]
    df_b = article_dfs[annee_b]

    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.metric(f"Articles {annee_a}", f"{len(df_a):,}".replace(",", " "))
    with col_k2:
        st.metric(f"Articles {annee_b}", f"{len(df_b):,}".replace(",", " "))
    with col_k3:
        st.metric("Période", f"{annee_a} ↔ {annee_b}")

    max_articles = st.slider(
        "Nombre maximum d’articles par année pour l’audit (pour limiter le temps de calcul)",
        min_value=20,
        max_value=300,
        value=100,
    )

    st.markdown("#### Configuration du modèle sémantique")
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        mode_modele = st.radio(
            "Source du modèle",
            ["HuggingFace (en ligne)", "Chemin local (hors-ligne)"],
            horizontal=False,
        )
    with col_m2:
        local_model_path: Optional[str] = None
        if mode_modele == "Chemin local (hors-ligne)":
            local_model_path = st.text_input(
                "Chemin local vers le modèle SentenceTransformer (dossier contenant config.json, pytorch_model.bin, etc.)",
                value="",
                placeholder="ex: c:/models/biencoder-camembert-base-mmarcoFR",
            ).strip() or None

    if st.button("Lancer l’audit sémantique (SentenceTransformer – 30 secondes pour 100 articles)", type="primary"):
        articles_2024 = df_a["texte_complet"].astype(str).head(max_articles).tolist()
        articles_2025 = df_b["texte_complet"].astype(str).head(max_articles).tolist()

        with st.spinner("Calcul des embeddings et des similarités..."):
            try:
                results = analyse_semantique.audit_semantique_lois(
                    articles_2024=articles_2024,
                    articles_2025=articles_2025,
                    local_model_path=local_model_path,
                )
            except Exception as e:
                st.error(f"Erreur lors de l’audit sémantique : {e}")
                return

        st.success("Audit sémantique terminé.")

        sim_matrix = results.get("matrice_complete")
        score_moyen = results.get("score_moyen", np.nan)
        top_sim = results.get("top_10_similaires", [])
        top_rupt = results.get("top_10_ruptures", [])

        # KPIs audit
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Score moyen (cosinus)", f"{score_moyen:.3f}")
        with c2:
            st.metric("Paires similaires (top 10)", len(top_sim))
        with c3:
            st.metric("Paires en rupture (top 10)", len(top_rupt))
        with c4:
            seuil_opt = 0.45
            st.metric("Seuil rupture (défaut)", f"{seuil_opt:.2f}")

        # Construire data_combined comme dans le notebook
        df_a_work = df_a.head(max_articles).copy()
        df_b_work = df_b.head(max_articles).copy()
        if isinstance(sim_matrix, np.ndarray):
            df_b_work["similarite_max_artice"] = sim_matrix.max(axis=0)
            df_b_work["source"] = annee_b
            df_a_work["similarite_max_artice"] = sim_matrix.max(axis=1)
            df_a_work["source"] = annee_a
            data_combined = pd.concat([
                df_a_work[["texte_complet", "similarite_max_artice", "source"]],
                df_b_work[["texte_complet", "similarite_max_artice", "source"]],
            ], ignore_index=True)

            # Seuil optimal GMM
            try:
                seuil_res = analyse_semantique.trouver_seuil_optimal(data_combined)
                seuil_opt = float(seuil_res.get("seuil_optimal", 0.45))
            except Exception:
                seuil_opt = 0.45

            st.markdown("#### Graphiques d’audit sémantique (comme dans le notebook)")

            fig_dist = analyse_semantique.plot_distribution_similarite(data_combined)
            st.plotly_chart(fig_dist, use_container_width=True)

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_box = analyse_semantique.plot_boxplot_similarite(data_combined)
                st.plotly_chart(fig_box, use_container_width=True)
            with col_g2:
                fig_pie = analyse_semantique.plot_ruptures_piechart(data_combined, seuil=seuil_opt)
                st.plotly_chart(fig_pie, use_container_width=True)

            fig_taille = analyse_semantique.plot_taille_vs_similarite(
                data_combined,
                col_similarite="similarite_max_artice",
                text_col="texte_complet",
                source_col="source",
            )
            st.plotly_chart(fig_taille, use_container_width=True)

            max_dim = min(30, sim_matrix.shape[0], sim_matrix.shape[1])
            if max_dim > 0:
                fig_heat = px.imshow(
                    sim_matrix[:max_dim, :max_dim],
                    color_continuous_scale="Viridis",
                    origin="lower",
                    labels={"x": f"Articles {annee_b}", "y": f"Articles {annee_a}", "color": "Similarité"},
                    title=f"Carte de chaleur des similarités (top {max_dim}×{max_dim})",
                )
                st.plotly_chart(fig_heat, use_container_width=True)

        with st.expander("Télécharger les paires similaires et ruptures (CSV)"):
            if top_sim:
                display_df_with_download(pd.DataFrame(top_sim), "top10_articles_similaires")
            if top_rupt:
                display_df_with_download(pd.DataFrame(top_rupt), "top10_articles_ruptures")


def page_about(datasets: Dict[str, Any]) -> None:
    render_section_header(
        title="À propos du tableau de bord",
        subtitle="Equipe, prérequis techniques.",
        badge_label="Documentation",
        badge_variant="apropos",
    )

    st.markdown(
    """
    ### Équipe projet

    - **ASSA Allo** — 
    - **AYONTA NDJOUTSE Vanelle**
    - **KOULOU Anaklassè Crépin** 
    - **YAKAWOU Komlanvi Eyram** 

    ---

    ### Objectifs du tableau de bord

    - **Pilotage budgétaire** : analyser les dotations CP/AE par chapitre, ministère et pilier SND30.
    - **Traçabilité réglementaire** : rapprocher les projets budgétaires des articles de loi correspondants.
    - **Audit sémantique** : identifier continuités et ruptures de contenu entre deux exercices.

    ### Structure des onglets

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

    ### Prérequis techniques

    - **Python** ≥ 3.9, **Streamlit**, **pandas**, **numpy**, **plotly**, **sentence-transformers**, **transformers**.
    - Pour l'extraction avancée PDF :
      - `pdfplumber`, `PyMuPDF (fitz)`, bibliothèque `openai`, variable d'environnement `OPENAI_API_KEY`.
    - Pour l'audit sémantique hors ligne :
      - Modèle SentenceTransformer téléchargé localement, chemin renseigné dans l'onglet **Audit sémantique**.


    """
)


# ---------------------------------------------------------------------------
# Point d’entrée Streamlit
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Tableau de bord – Lois de finances et budget",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_css()

    datasets = load_default_datasets()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Aller vers",
        [
            "Vue d’ensemble",
            "Extraction et données brutes",
            "Prétraitement des textes",
            "Classification SND30",
            "Analyse budgétaire & conformité",
            "Audit sémantique",
            "À propos",
        ],
    )

    st.markdown(
        """
        <div class="taf-hero">
            <div class="taf-hero-main">
                <h1>Tableau de bord – Loi de finances, budget et SND30 🇨🇲 </h1>
                <p>
                    Pilotage des projets budgétaires, analyse de conformité aux piliers SND30
                    et audit sémantique des lois de finances sur plusieurs exercices.
                </p>
            </div>
           
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Vue d’ensemble":
        page_overview(datasets)
    elif page == "Extraction et données brutes":
        page_extraction(datasets)
    elif page == "Prétraitement des textes":
        page_pretraitement(datasets)
    elif page == "Classification SND30":
        page_classification(datasets)
    elif page == "Analyse budgétaire & conformité":
        page_analyse_budget_conformite(datasets)
    elif page == "Audit sémantique":
        page_audit_semantique(datasets)
    elif page == "À propos":
        page_about(datasets)


if __name__ == "__main__":
    main()

