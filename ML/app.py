# titanic_app.py
import streamlit as st
import joblib
import pandas as pd

# Chargement du modèle entraîné
model = joblib.load("titanic_pipeline.pkl")

# Titre de l'application
st.title("Titanic Survival Prediction")

# Entrée utilisateur
st.header("Entrez les informations du passager :")
Pclass = st.selectbox("Classe (Pclass)", [1, 2, 3])
Sex = st.selectbox("Sexe", ["male", "female"])
Age = st.slider("Âge", 0, 100, 25)
Fare = st.slider("Tarif du billet (Fare)", 0.0, 500.0, 32.0)
Embarked = st.selectbox("Port d'embarquement (Embarked)", ["S", "C", "Q"])
SibSp = st.slider("Nombre de frères/soeurs / époux(ses) à bord (SibSp)", 0, 10, 0)
Parch = st.slider("Nombre de parents / enfants à bord (Parch)", 0, 10, 0)

# Prédiction
if st.button("Prédire la survie"):
    FamilySize = SibSp + Parch
    X_new = pd.DataFrame([[Pclass, Sex, Age, Fare, Embarked, FamilySize]],
                         columns=["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"])
    
    pred = model.predict(X_new)
    st.subheader("Résultat de la prédiction :")
    st.write("✅ Survécu" if pred[0] == 1 else "❌ N'a pas survécu")
