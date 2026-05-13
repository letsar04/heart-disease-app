import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from ucimlrepo import fetch_ucirepo

# Charger les données
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

df = pd.concat([X, y], axis=1)
df.columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal','target']

# Binariser la cible (0 = pas malade, 1 = malade)
df['target'] = (df['target'] > 0).astype(int)

# Traitement des valeurs manquantes
df = df.dropna()

X = df.drop('target', axis=1)
y = df['target']

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Normalisation
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Définition des modèles
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN':                 KNeighborsClassifier(),
    'SVM':                 SVC(probability=True),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(random_state=42),
    'AdaBoost':            AdaBoostClassifier(random_state=42)
}

# Entraîner et évaluer
results = []
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:,1]

    results.append({
        'Modèle':    name,
        'Accuracy':  round(accuracy_score(y_test, y_pred), 3),
        'Précision': round(precision_score(y_test, y_pred), 3),
        'Rappel':    round(recall_score(y_test, y_pred), 3),
        'F1-Score':  round(f1_score(y_test, y_pred), 3),
        'AUC-ROC':   round(roc_auc_score(y_test, y_proba), 3),
    })

df_results = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)

# Sélectionner le meilleur modèle
best_model_name = df_results.iloc[0]['Modèle']
best_model = models[best_model_name]

st.title("🫀 Prédiction de maladie cardiaque")
st.sidebar.header("Paramètres du patient")

# Inputs utilisateur
age      = st.sidebar.slider("Âge", 20, 80, 50)
sex      = st.sidebar.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x==0 else "Homme")
cp       = st.sidebar.selectbox("Type douleur thoracique", [0,1,2,3])
trestbps = st.sidebar.slider("Pression artérielle (mm Hg)", 90, 200, 120)
chol     = st.sidebar.slider("Cholestérol (mg/dl)", 100, 600, 240)
fbs      = st.sidebar.selectbox("Glycémie à jeun > 120 mg/dl", [0, 1], format_func=lambda x: "Non" if x==0 else "Oui")
restecg  = st.sidebar.selectbox("Résultats ECG au repos", [0,1,2])
thalach  = st.sidebar.slider("Fréquence cardiaque max", 70, 200, 150)
exang    = st.sidebar.selectbox("Angine induite par l'exercice", [0, 1], format_func=lambda x: "Non" if x==0 else "Oui")
oldpeak  = st.sidebar.slider("Dépression ST", 0.0, 6.0, 1.0)
slope    = st.sidebar.selectbox("Pente du segment ST", [0,1,2])
ca       = st.sidebar.slider("Nombre de vaisseaux principaux", 0, 3, 0)
thal     = st.sidebar.selectbox("Thalassémie", [3,6,7], format_func=lambda x: "Normal" if x==3 else "Défaut fixe" if x==6 else "Défaut réversible")

# Prédiction
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
input_data_scaled = scaler.transform(input_data)
prediction = best_model.predict(input_data_scaled)[0]
proba      = best_model.predict_proba(input_data_scaled)[0][1]

if prediction == 1:
    st.error(f"⚠️ Risque de maladie cardiaque détecté ({proba:.1%})")
else:
    st.success(f"✅ Pas de maladie cardiaque détectée ({1-proba:.1%})")

# Graphiques comparatifs
st.subheader("Comparaison des modèles")
st.dataframe(df_results)
fig = px.bar(df_results, x='Modèle', y='AUC-ROC', color='Modèle')
st.plotly_chart(fig)