import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

df = pd.read_csv("donnees_ml/dataset_ml_avc.csv")

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

modele = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=3,
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)
modele.fit(X_train, y_train)

joblib.dump(modele, "modele_xgb_avc.pkl")
joblib.dump(le_sexe, "encodeur_sexe.pkl")
joblib.dump(le_avc, "encodeur_type_avc.pkl")
joblib.dump(le_sor, "encodeur_sor_mod.pkl")

print("Modele XGBoost et encodeurs sauvegardes !")
print(f"Classes sexe    : {list(le_sexe.classes_)}")
print(f"Classes type_avc: {list(le_avc.classes_)}")
print(f"Classes sor_mod : {list(le_sor.classes_)}")
