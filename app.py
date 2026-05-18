"""
=============================================================
  HeartScan AI — Classification Maladie Cardiaque
  Projet IA · IFOAD · Dr Arthur Sawadogo
=============================================================
Lancer : streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import warnings
warnings.filterwarnings("ignore")


# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeartScan AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── THEME CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0f1929 60%, #0a0f1e 100%);
    border-right: 1px solid rgba(220,38,38,0.2);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { 
    color: #94a3b8 !important; 
    transition: all .2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #ef4444 !important; }

/* ── Main background ── */
.main { background: #070c17; }
.block-container { padding-top: 1.5rem; max-width: 1300px; }

/* ── Cards ── */
.card {
    background: linear-gradient(135deg, #0f1929 0%, #111827 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-red {
    background: linear-gradient(135deg, #1a0a0a 0%, #1f0f0f 100%);
    border: 1px solid rgba(220,38,38,0.3);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Métriques ── */
.metric-box {
    background: #0f1929;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #ef4444; line-height: 1; }
.metric-label { font-size: .75rem; color: #64748b; text-transform: uppercase; letter-spacing: .08em; margin-top: .3rem; }

/* ── Badge de risque ── */
.risk-high {
    background: linear-gradient(90deg,#7f1d1d,#991b1b);
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: .8rem 1.2rem;
    font-size: 1.1rem;
    font-weight: 600;
    color: #fca5a5;
    text-align: center;
}
.risk-low {
    background: linear-gradient(90deg,#064e3b,#065f46);
    border: 1px solid #10b981;
    border-radius: 8px;
    padding: .8rem 1.2rem;
    font-size: 1.1rem;
    font-weight: 600;
    color: #6ee7b7;
    text-align: center;
}

/* ── Titres ── */
h1 { font-size:2.2rem !important; font-weight:700 !important; color:#f8fafc !important; }
h2 { font-size:1.4rem !important; font-weight:600 !important; color:#e2e8f0 !important; }
h3 { font-size:1.1rem !important; font-weight:500 !important; color:#94a3b8 !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Inputs ── */
[data-testid="stSlider"] .st-b6 { background: #ef4444 !important; }
.stSelectbox select, .stNumberInput input { 
    background: #0f1929 !important; 
    color: #e2e8f0 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

/* ── Tables ── */
.dataframe { font-family: 'JetBrains Mono', monospace !important; font-size:.82rem !important; }

/* ── Boutons ── */
.stButton button {
    background: linear-gradient(90deg, #dc2626, #b91c1c) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: .5rem 1.5rem !important;
    transition: all .2s !important;
}
.stButton button:hover { 
    background: linear-gradient(90deg, #ef4444, #dc2626) !important;
    transform: translateY(-1px) !important;
}

/* ── Download ── */
[data-testid="stDownloadButton"] button {
    background: #0f172a !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #94a3b8 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid rgba(255,255,255,0.07); }
.stTabs [data-baseweb="tab"] { color: #64748b; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #ef4444 !important; border-bottom-color: #ef4444 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── PALETTE PLOTLY ─────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
COLORS = {
    "accent":   "#ef4444",
    "accent2":  "#f97316",
    "blue":     "#3b82f6",
    "teal":     "#14b8a6",
    "purple":   "#8b5cf6",
    "green":    "#10b981",
    "bg":       "#0f1929",
    "surface":  "#111827",
    "border":   "rgba(255,255,255,0.07)",
}
MODEL_COLORS = ["#ef4444","#f97316","#3b82f6","#14b8a6","#8b5cf6","#10b981"]
MODEL_NAMES  = ["Logistic Reg.","KNN","SVM","Decision Tree","Random Forest","AdaBoost"]

# ─── CHARGEMENT & CACHE ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    try:
        from ucimlrepo import fetch_ucirepo
        hd = fetch_ucirepo(id=45)
        X  = hd.data.features
        y  = hd.data.targets
        df = pd.concat([X, y], axis=1)
    except Exception:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
                "thalach","exang","oldpeak","slope","ca","thal","target"]
        df = pd.read_csv(url, names=cols, na_values="?")

    df.columns = ["age","sex","cp","trestbps","chol","fbs","restecg",
                  "thalach","exang","oldpeak","slope","ca","thal","target"]
    df["target"] = (df["target"] > 0).astype(int)
    df = df.dropna()
    df = df.astype(float)
    return df

@st.cache_resource(show_spinner=False)
def train_all_models(test_size=0.2, random_state=42):
    df   = load_data()
    X    = df.drop("target", axis=1)
    y    = df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler     = StandardScaler()
    X_tr_s     = scaler.fit_transform(X_tr)
    X_te_s     = scaler.transform(X_te)

    models_def = {
        "Logistic Reg.":  LogisticRegression(max_iter=1000, random_state=random_state),
        "KNN":            KNeighborsClassifier(),
        "SVM":            SVC(probability=True, random_state=random_state),
        "Decision Tree":  DecisionTreeClassifier(random_state=random_state),
        "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=random_state),
        "AdaBoost":       AdaBoostClassifier(n_estimators=100, random_state=random_state),
    }
    results, trained = [], {}
    for name, mdl in models_def.items():
        mdl.fit(X_tr_s, y_tr)
        y_pred  = mdl.predict(X_te_s)
        y_proba = mdl.predict_proba(X_te_s)[:,1]
        cv      = cross_val_score(mdl, X_tr_s, y_tr, cv=5, scoring="accuracy")
        results.append({
            "Modèle":    name,
            "Accuracy":  round(accuracy_score(y_te, y_pred),3),
            "Précision": round(precision_score(y_te, y_pred),3),
            "Rappel":    round(recall_score(y_te, y_pred),3),
            "F1-Score":  round(f1_score(y_te, y_pred),3),
            "AUC-ROC":   round(roc_auc_score(y_te, y_proba),3),
            "CV Mean":   round(cv.mean(),3),
            "CV Std":    round(cv.std(),3),
        })
        trained[name] = (mdl, y_pred, y_proba)

    df_res = pd.DataFrame(results).sort_values("AUC-ROC", ascending=False)
    return df_res, trained, scaler, X_te_s, y_te, X.columns.tolist()

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem;'>
        <div style='font-size:2.8rem;'>🫀</div>
        <div style='font-size:1.2rem;font-weight:700;color:#f8fafc;letter-spacing:.03em;'>HeartScan AI</div>
        <div style='font-size:.72rem;color:#475569;margin-top:.2rem;'>IFOAD · Machine Learning Project</div>
    </div>
    <hr style='margin:.5rem 0 1rem;opacity:.15;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Accueil",
         "📊  Analyse Exploratoire",
         "📈  Comparaison & Métriques",
         "🔮  Prédiction Patient",
         "📋  À propos"],
        label_visibility="collapsed"
    )
    page = page.split("  ",1)[1]   # strip emoji prefix

    st.markdown("<hr style='opacity:.15;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.72rem;color:#334155;text-align:center;line-height:1.6;'>
        Heart Disease UCI · 14 features<br>
        6 algorithmes · 5 métriques<br>
        <span style='color:#475569;'>Dr Arthur Sawadogo</span>
    </div>
    """, unsafe_allow_html=True)

# ─── CHARGEMENT ─────────────────────────────────────────────────────────────────
with st.spinner("⚙️ Chargement des données…"):
    df = load_data()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if page == "Accueil":
    st.markdown("## 🫀 HeartScan AI — Détection de Maladies Cardiaques")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (str(len(df)), "Patients"),
        (str(df.shape[1]-1), "Features"),
        (f"{df['target'].mean()*100:.0f}%", "Taux de maladie"),
        ("6", "Algorithmes ML"),
    ]
    for col, (val, lbl) in zip([c1,c2,c3,c4], metrics):
        col.markdown(f"""
        <div class='metric-box'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_left, c_right = st.columns([3,2])

    with c_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📌 Objectif du projet")
        st.markdown("""
Développer et comparer **6 algorithmes de classification** pour prédire
la présence d'une maladie cardiaque à partir de données cliniques
*(Heart Disease UCI Dataset)*.

**Pipeline complet :**
- Analyse exploratoire (EDA) avec visualisations interactives
- Prétraitement, normalisation, split train/test stratifié
- Entraînement des 6 modèles + validation croisée (5-fold)
- Comparaison sur 5 métriques : Accuracy, Précision, Rappel, F1, AUC-ROC
- Interface de prédiction individuelle pour un patient
""")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🧬 Distribution cible")
        counts = df["target"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["Pas de maladie","Maladie cardiaque"],
            values=[counts[0], counts[1]],
            hole=.6,
            marker_colors=["#14b8a6","#ef4444"],
            textfont_size=12,
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(size=11)),
            margin=dict(t=10,b=10,l=10,r=10),
            height=230,
            annotations=[dict(text=f"<b>{len(df)}</b><br>patients",
                              x=.5,y=.5,font_size=14,showarrow=False,
                              font_color="#94a3b8")]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Description des features
    st.markdown("#### 🗂️ Description des variables")
    feat_info = {
        "age":      ("Âge (années)",                    "Numérique"),
        "sex":      ("Sexe (1=H, 0=F)",                 "Binaire"),
        "cp":       ("Type douleur thoracique (0-3)",   "Catégorielle"),
        "trestbps": ("Pression artérielle repos (mmHg)","Numérique"),
        "chol":     ("Cholestérol (mg/dl)",              "Numérique"),
        "fbs":      ("Glycémie à jeun >120 (1=oui)",    "Binaire"),
        "restecg":  ("ECG au repos (0,1,2)",             "Catégorielle"),
        "thalach":  ("FC max atteinte",                  "Numérique"),
        "exang":    ("Angine à l'effort (1=oui)",        "Binaire"),
        "oldpeak":  ("Dépression ST (exercice vs repos)","Numérique"),
        "slope":    ("Pente ST au pic (0,1,2)",          "Catégorielle"),
        "ca":       ("Nb. vaisseaux colorés (0-3)",      "Numérique"),
        "thal":     ("Thalassémie (3=N,6=fixe,7=réver.)","Catégorielle"),
        "target":   ("Maladie cardiaque (1=oui)",        "🎯 Cible"),
    }
    df_desc = pd.DataFrame(feat_info, index=["Description","Type"]).T.reset_index()
    df_desc.columns = ["Feature","Description","Type"]
    st.dataframe(df_desc, use_container_width=True, hide_index=True, height=390)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE EXPLORATOIRE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analyse Exploratoire":
    st.markdown("## 📊 Analyse Exploratoire des Données (EDA)")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📐 Distributions",
        "🔗 Corrélations",
        "⚖️ Comparaison par groupe",
        "🔍 Questions EDA",
    ])

    # ── TAB 1 — Distributions ──
    with tab1:
        feat_num = ["age","trestbps","chol","thalach","oldpeak"]
        fig = make_subplots(rows=2, cols=3,
                            subplot_titles=[f.capitalize() for f in feat_num] + [""],
                            vertical_spacing=.14, horizontal_spacing=.08)
        for i, f in enumerate(feat_num):
            row, col = divmod(i, 3)
            fig.add_trace(go.Histogram(
                x=df[df["target"]==0][f], name="Pas maladie",
                marker_color="#14b8a6", opacity=.7,
                showlegend=(i==0), nbinsx=20,
            ), row=row+1, col=col+1)
            fig.add_trace(go.Histogram(
                x=df[df["target"]==1][f], name="Maladie",
                marker_color="#ef4444", opacity=.7,
                showlegend=(i==0), nbinsx=20,
            ), row=row+1, col=col+1)
        fig.update_layout(
            template=PLOTLY_TEMPLATE, barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=500, legend=dict(orientation="h",y=1.06),
            title_text="Distribution des variables numériques par classe",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2 — Corrélations ──
    with tab2:
        corr = df.corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values.round(2),
            x=corr.columns, y=corr.columns,
            colorscale=[[0,"#3b82f6"],[.5,"#111827"],[1,"#ef4444"]],
            zmin=-1, zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont={"size":9},
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=520,
            title="Matrice de corrélation",
        )
        st.plotly_chart(fig, use_container_width=True)
        top = corr["target"].abs().sort_values(ascending=False).iloc[1:6]
        st.markdown("**Top 5 corrélations avec la variable cible :**")
        st.dataframe(top.rename("Corrélation avec target").reset_index().rename(columns={"index":"Feature"}),
                     use_container_width=False, hide_index=True)

    # ── TAB 3 — Boxplots par groupe ──
    with tab3:
        sel = st.selectbox("Feature numérique", ["age","trestbps","chol","thalach","oldpeak"])
        fig = go.Figure()
        for label, color, name in [(0,"#14b8a6","Pas de maladie"),(1,"#ef4444","Maladie")]:
            fig.add_trace(go.Box(
                y=df[df["target"]==label][sel],
                name=name, marker_color=color,
                boxmean="sd",
            ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            title=f"Distribution de «{sel}» selon la présence de maladie",
            yaxis_title=sel,
        )
        st.plotly_chart(fig, use_container_width=True)
        col_a, col_b = st.columns(2)
        for grp, col in [(0,col_a),(1,col_b)]:
            name = "Pas de maladie" if grp==0 else "Maladie cardiaque"
            sub  = df[df["target"]==grp][sel]
            col.markdown(f"**{name}**")
            col.dataframe(sub.describe().round(2).to_frame(), use_container_width=True)

    # ── TAB 4 — Questions EDA ──
    with tab4:
        st.markdown("**Réponses aux 6 questions analytiques du sujet**")

        # Q1 – distribution âge
        with st.expander("Q1 — Distribution de l'âge", expanded=True):
            fig = px.histogram(df, x="age", nbins=20, color_discrete_sequence=["#ef4444"],
                               title="Distribution de l'âge")
            fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Âge médian : {df['age'].median():.0f} ans | Min {df['age'].min():.0f} | Max {df['age'].max():.0f}")

        # Q2 – sexe vs maladie
        with st.expander("Q2 — Différence homme / femme"):
            ct = df.groupby(["sex","target"]).size().reset_index(name="count")
            ct["Sexe"]   = ct["sex"].map({1:"Homme",0:"Femme"})
            ct["Classe"] = ct["target"].map({1:"Maladie",0:"Pas maladie"})
            fig = px.bar(ct, x="Sexe", y="count", color="Classe",
                         color_discrete_map={"Maladie":"#ef4444","Pas maladie":"#14b8a6"},
                         barmode="group", title="Maladie cardiaque selon le sexe")
            fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=320)
            st.plotly_chart(fig, use_container_width=True)

        # Q3 – cp vs target
        with st.expander("Q3 — Type de douleur thoracique (cp) vs maladie"):
            ct2 = df.groupby(["cp","target"]).size().reset_index(name="count")
            ct2["Classe"] = ct2["target"].map({1:"Maladie",0:"Pas maladie"})
            ct2["cp_label"] = ct2["cp"].map({0:"Asymptomatique",1:"Angine typique",
                                             2:"Angine atypique",3:"Douleur non-angineuse"})
            fig = px.bar(ct2, x="cp_label", y="count", color="Classe",
                         color_discrete_map={"Maladie":"#ef4444","Pas maladie":"#14b8a6"},
                         barmode="group", title="Type de douleur thoracique vs maladie")
            fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=320)
            st.plotly_chart(fig, use_container_width=True)

        # Q4 – valeurs moyennes
        with st.expander("Q4 — Valeurs moyennes trestbps / chol / thalach"):
            means = df.groupby("target")[["trestbps","chol","thalach"]].mean().round(1)
            means.index = ["Pas maladie","Maladie"]
            st.dataframe(means, use_container_width=True)

        # Q5 – fbs
        with st.expander("Q5 — Glycémie à jeun (fbs) et maladie"):
            ct3 = df.groupby(["fbs","target"]).size().reset_index(name="count")
            ct3["fbs_label"] = ct3["fbs"].map({0:"Glycémie ≤ 120",1:"Glycémie > 120"})
            ct3["Classe"]    = ct3["target"].map({1:"Maladie",0:"Pas maladie"})
            fig = px.bar(ct3, x="fbs_label", y="count", color="Classe",
                         color_discrete_map={"Maladie":"#ef4444","Pas maladie":"#14b8a6"},
                         barmode="group", title="Glycémie à jeun vs maladie")
            fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Q6 – exang
        with st.expander("Q6 — Angine à l'effort (exang) vs maladie"):
            ct4 = df.groupby(["exang","target"]).size().reset_index(name="count")
            ct4["exang_label"] = ct4["exang"].map({0:"Sans angine",1:"Avec angine"})
            ct4["Classe"]      = ct4["target"].map({1:"Maladie",0:"Pas maladie"})
            fig = px.bar(ct4, x="exang_label", y="count", color="Classe",
                         color_discrete_map={"Maladie":"#ef4444","Pas maladie":"#14b8a6"},
                         barmode="group", title="Angine à l'effort vs maladie")
            fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig, use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — COMPARAISON & MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Comparaison & Métriques":
    st.markdown("## 📈 Comparaison des Modèles")
    st.markdown("---")

    with st.spinner("Entraînement / récupération…"):
        df_res, trained, scaler, X_te_s, y_te, feat_names = train_all_models()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Tableau comparatif",
        "📡 Graphique radar",
        "📉 Courbes ROC",
        "🔲 Matrices de confusion",
    ])

    # ── Tableau ──
    with tab1:
        st.markdown("#### Résultats complets (triés par AUC-ROC)")
        def color_val(val):
            if isinstance(val, float):
                if val >= .90: return "background-color:#064e3b;color:#6ee7b7"
                if val >= .85: return "background-color:#134e4a;color:#5eead4"
                if val >= .80: return "background-color:#1e293b;color:#94a3b8"
            return ""
        styled = df_res.style.map(color_val, subset=["Accuracy","Précision","Rappel","F1-Score","AUC-ROC"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        csv = df_res.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Exporter en CSV", csv, "resultats_modeles.csv", "text/csv")

        best = df_res.iloc[0]
        st.markdown(f"""
        <div class='card-red' style='margin-top:1rem;'>
            🏆 <b>Meilleur modèle : {best['Modèle']}</b> — 
            AUC-ROC <b>{best['AUC-ROC']}</b> | 
            F1 <b>{best['F1-Score']}</b> | 
            Accuracy <b>{best['Accuracy']}</b>
        </div>
        """, unsafe_allow_html=True)

    # ── Radar ──
    with tab2:
        metrics_cols = ["Accuracy","Précision","Rappel","F1-Score","AUC-ROC"]
        fig = go.Figure()
        for i, row in df_res.iterrows():
            vals = [row[m] for m in metrics_cols]
            vals += [vals[0]]  # close the polygon
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=metrics_cols+[metrics_cols[0]],
                fill="toself", name=row["Modèle"],
                line_color=MODEL_COLORS[list(df_res["Modèle"]).index(row["Modèle"])],
                opacity=.7,
            ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0.6,1.0], color="#475569"),
                angularaxis=dict(color="#64748b"),
            ),
            height=480,
            legend=dict(orientation="h", y=-0.12),
            title="Radar des performances par algorithme",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── ROC ──
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                                 line=dict(dash="dash",color="#334155"),
                                 name="Random",showlegend=False))
        for i,(name,(mdl,_,yp)) in enumerate(trained.items()):
            fpr,tpr,_ = roc_curve(y_te, yp)
            auc_val   = roc_auc_score(y_te, yp)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc_val:.3f})",
                line=dict(color=MODEL_COLORS[i], width=2),
            ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=480, xaxis_title="Taux de faux positifs",
            yaxis_title="Taux de vrais positifs",
            title="Courbes ROC — Comparaison des 6 modèles",
            legend=dict(orientation="v",x=1.01,font_size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Confusion matrices ──
    with tab4:
        sel_model = st.selectbox("Choisir un modèle", list(trained.keys()))
        _, y_pred, _ = trained[sel_model]
        cm   = confusion_matrix(y_te, y_pred)
        fig  = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Prédit",y="Réel",color="Count"),
            x=["Pas maladie","Maladie"],
            y=["Pas maladie","Maladie"],
            color_continuous_scale=[[0,"#0f1929"],[0.5,"#7f1d1d"],[1,"#ef4444"]],
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380, title=f"Matrice de confusion — {sel_model}",
        )
        st.plotly_chart(fig, use_container_width=True)

        tn,fp,fn,tp = cm.ravel()
        c1,c2,c3,c4 = st.columns(4)
        for col, (lbl, val, clr) in zip([c1,c2,c3,c4],[
            ("Vrais Positifs",  tp, "#10b981"),
            ("Vrais Négatifs",  tn, "#14b8a6"),
            ("Faux Positifs",   fp, "#f97316"),
            ("Faux Négatifs",   fn, "#ef4444"),
        ]):
            col.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value' style='color:{clr};'>{val}</div>
                <div class='metric-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prédiction Patient":
    st.markdown("## 🔮 Prédiction pour un Patient")
    st.markdown("---")

    with st.spinner("Chargement des modèles…"):
        df_res, trained, scaler, X_te_s, y_te, feat_names = train_all_models()

    sel_model = st.selectbox(
        "🤖 Algorithme de prédiction",
        list(trained.keys()),
        index=list(trained.keys()).index(df_res.iloc[0]["Modèle"])
    )

    st.markdown("#### 👤 Données du patient")
    c1, c2, c3 = st.columns(3)

    with c1:
        age      = st.slider("Âge (ans)", 20, 80, 55)
        trestbps = st.slider("Pression artérielle (mmHg)", 80, 200, 130)
        chol     = st.slider("Cholestérol (mg/dl)", 100, 600, 250)
        oldpeak  = st.slider("Dépression ST (oldpeak)", 0.0, 6.2, 1.0, .1)
        ca       = st.selectbox("Nb. vaisseaux colorés (ca)", [0,1,2,3])

    with c2:
        sex    = st.radio("Sexe", ["Femme","Homme"])
        sex    = 1 if sex=="Homme" else 0
        cp     = st.selectbox("Type douleur thoracique",
                              [0,1,2,3],
                              format_func=lambda x:
                              {0:"Asymptomatique",1:"Angine typique",
                               2:"Angine atypique",3:"Non-angineuse"}[x])
        fbs    = st.radio("Glycémie à jeun > 120", ["Non","Oui"])
        fbs    = 1 if fbs=="Oui" else 0
        exang  = st.radio("Angine à l'effort", ["Non","Oui"])
        exang  = 1 if exang=="Oui" else 0

    with c3:
        thalach = st.slider("FC max atteinte", 60, 220, 150)
        restecg = st.selectbox("ECG au repos", [0,1,2],
                               format_func=lambda x:
                               {0:"Normal",1:"Anomalie ST-T",2:"HVG probable"}[x])
        slope   = st.selectbox("Pente ST",  [0,1,2],
                               format_func=lambda x:
                               {0:"Descendante",1:"Plate",2:"Ascendante"}[x])
        thal    = st.selectbox("Thalassémie",  [3,6,7],
                               format_func=lambda x:
                               {3:"Normal",6:"Défaut fixe",7:"Défaut réversible"}[x])

    st.markdown("---")
    if st.button("🔍 Analyser ce patient"):
        input_arr = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                               thalach,exang,oldpeak,slope,ca,thal]])
        input_sc  = scaler.transform(input_arr)
        model, _, _ = trained[sel_model]
        pred  = model.predict(input_sc)[0]
        proba = model.predict_proba(input_sc)[0][1]

        col_res, col_gauge = st.columns([1,1])

        with col_res:
            if pred == 1:
                st.markdown(f"""
                <div class='risk-high'>
                    ⚠️ Risque élevé de maladie cardiaque<br>
                    <span style='font-size:.85rem;font-weight:400;'>
                        Probabilité estimée : <b>{proba*100:.1f}%</b>
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='risk-low'>
                    ✅ Faible risque de maladie cardiaque<br>
                    <span style='font-size:.85rem;font-weight:400;'>
                        Probabilité estimée : <b>{proba*100:.1f}%</b>
                    </span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            # Probabilités de tous les modèles
            all_probas = []
            for mname, (mdl,_,_) in trained.items():
                p = mdl.predict_proba(input_sc)[0][1]
                all_probas.append({"Modèle": mname, "Probabilité maladie": round(p,3)})
            df_ap = pd.DataFrame(all_probas).sort_values("Probabilité maladie", ascending=False)
            st.markdown("**Probabilités selon chaque modèle :**")
            st.dataframe(df_ap, use_container_width=True, hide_index=True)

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba*100,1),
                number={"suffix":"%","font":{"size":36,"color":"#f8fafc"}},
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor="#475569", tickfont=dict(color="#64748b")),
                    bar=dict(color="#ef4444" if proba>.5 else "#10b981", thickness=.25),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0,40],   color="#064e3b"),
                        dict(range=[40,60],  color="#713f12"),
                        dict(range=[60,100], color="#7f1d1d"),
                    ],
                    threshold=dict(line=dict(color="#f8fafc",width=3), thickness=.8, value=50),
                ),
                title={"text":"Score de risque","font":{"color":"#94a3b8","size":14}},
            ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor="rgba(0,0,0,0)",
                height=280, margin=dict(t=30,b=0,l=20,r=20),
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — À PROPOS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "À propos":
    st.markdown("## 📋 À propos du projet")
    st.markdown("---")

    st.markdown("""
    <div class='card'>
        <h4 style='color:#e2e8f0;'>📘 Contexte académique</h4>
        <p style='color:#94a3b8;'>
            Projet réalisé dans le cadre du cours <b>Intelligence Artificielle et Apprentissage Automatique</b>
            à l'<b>IFOAD — Université Joseph KI-ZERBO</b>, sous la direction du <b>Dr Arthur Sawadogo</b>.
        </p>
    </div>
    <div class='card'>
        <h4 style='color:#e2e8f0;'>🗄️ Jeu de données</h4>
        <p style='color:#94a3b8;'>
            <b>Heart Disease UCI</b> — UCI Machine Learning Repository<br>
            <a href='https://archive.ics.uci.edu/dataset/45/heart+disease' 
               style='color:#ef4444;'>archive.ics.uci.edu/dataset/45/heart+disease</a><br><br>
            303 patients · 13 features cliniques · Variable cible binaire (maladie : oui/non)
        </p>
    </div>
    <div class='card'>
        <h4 style='color:#e2e8f0;'>⚙️ Stack technique</h4>
        <p style='color:#94a3b8;'>
            Python · Streamlit · Scikit-learn · Pandas · NumPy<br>
            Plotly (visualisations interactives) · ucimlrepo (chargement données)
        </p>
    </div>
    <div class='card'>
        <h4 style='color:#e2e8f0;'>🎯 Métriques d'évaluation</h4>
        <p style='color:#94a3b8;'>
            Accuracy · Précision · Rappel (Recall) · F1-Score · AUC-ROC<br>
            Validation croisée 5-fold · Matrices de confusion interactives
        </p>
    </div>
    """, unsafe_allow_html=True)