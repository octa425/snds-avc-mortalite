import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score
)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("donnees_ml/dataset_ml_avc.csv")
print(f"Shape : {df.shape}")
print(f"Taux readmission : {df['readmis_30j'].mean():.3f}")

le_sexe = LabelEncoder()
le_avc = LabelEncoder()
le_sor = LabelEncoder()

df["sexe_enc"] = le_sexe.fit_transform(df["sexe"])
df["type_avc_enc"] = le_avc.fit_transform(df["type_avc"])
df["sor_mod_enc"] = le_sor.fit_transform(df["sor_mod"])

FEATURES = [
    "age", "sexe_enc", "type_avc_enc",
    "hta", "diabete", "fibrillation",
    "nb_hospit", "duree_sejour", "sor_mod_enc"
]

X = df[FEATURES]
y = df["readmis_30j"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train : {len(X_train)} | Test : {len(X_test)}")

def evaluer_modele(nom, modele, X_train, X_test, y_train, y_test):
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)
    y_proba = modele.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cv_scores = cross_val_score(
        modele, X_train, y_train, cv=5, scoring="roc_auc"
    )
    print(f"\n{'='*50}")
    print(f"MODELE : {nom}")
    print(f"{'='*50}")
    print(f"AUC-ROC        : {auc:.3f}")
    print(f"PR-AUC         : {pr_auc:.3f}")
    print(f"CV AUC (5-fold): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"\n{classification_report(y_test, y_pred)}")
    return {
        "modele": nom,
        "auc_roc": round(auc, 3),
        "pr_auc": round(pr_auc, 3),
        "cv_auc_mean": round(cv_scores.mean(), 3),
        "cv_auc_std": round(cv_scores.std(), 3),
    }

resultats = []

print("\n1. LOGISTIC REGRESSION...")
lr = LogisticRegression(
    class_weight="balanced", max_iter=1000, random_state=42
)
resultats.append(evaluer_modele(
    "Logistic Regression", lr, X_train, X_test, y_train, y_test
))

print("\n2. RANDOM FOREST...")
rf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42
)
resultats.append(evaluer_modele(
    "Random Forest", rf, X_train, X_test, y_train, y_test
))

print("\n3. XGBOOST...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5,
    scale_pos_weight=3, random_state=42,
    eval_metric="logloss", verbosity=0
)
resultats.append(evaluer_modele(
    "XGBoost", xgb_model, X_train, X_test, y_train, y_test
))

print("\n4. LIGHTGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.1,
    class_weight="balanced", random_state=42, verbose=-1
)
resultats.append(evaluer_modele(
    "LightGBM", lgb_model, X_train, X_test, y_train, y_test
))

print("\n5. MLP (Reseau de neurones)...")
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=200,
    random_state=42
)
resultats.append(evaluer_modele(
    "MLP", mlp, X_train, X_test, y_train, y_test
))

print("\n" + "="*50)
print("COMPARAISON FINALE")
print("="*50)
df_resultats = pd.DataFrame(resultats)
df_resultats = df_resultats.sort_values("auc_roc", ascending=False)
print(df_resultats.to_string(index=False))

print("\n" + "="*50)
print("IMPORTANCE DES VARIABLES (Random Forest)")
print("="*50)
importances = pd.DataFrame({
    "variable": FEATURES,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
print(importances.to_string(index=False))

df_resultats.to_csv("donnees_ml/comparaison_modeles.csv", index=False)
print("\nResultats sauvegardes !")