from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

try:
    modele = joblib.load("modele_xgb_avc.pkl")
    le_sexe = joblib.load("encodeur_sexe.pkl")
    le_avc = joblib.load("encodeur_type_avc.pkl")
    le_sor = joblib.load("encodeur_sor_mod.pkl")
    print("Modele et encodeurs charges avec succes !")
except FileNotFoundError:
    modele = None
    print("AVERTISSEMENT : modele introuvable")

app = FastAPI(
    title="API Prediction Readmission Post-AVC",
    description="Predit le risque de readmission a 30 jours apres un AVC",
    version="1.0.0"
)

class DonneesPatient(BaseModel):
    age: int
    sexe: str
    type_avc: str
    hta: int
    diabete: int
    fibrillation: int
    nb_hospit: int
    duree_sejour: int
    sor_mod: int

@app.get("/")
def accueil():
    return {
        "message": "API Prediction Readmission Post-AVC",
        "version": "1.0.0",
        "modele": "XGBoost"
    }

@app.get("/health")
def health():
    return {
        "statut": "OK",
        "modele": "XGBoost",
        "projet": "SNDS AVC Readmission"
    }

@app.post("/predict")
def predire_readmission(patient: DonneesPatient):
    if modele is None:
        return {
            "erreur": "Modele non disponible",
            "solution": "Lancez sauvegarder_modele_avc.py"
        }

    if patient.sexe not in le_sexe.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Sexe invalide. Valides : {list(le_sexe.classes_)}"
        )

    if patient.type_avc not in le_avc.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Type AVC invalide. Valides : {list(le_avc.classes_)}"
        )

    classes_sor = [int(c) for c in le_sor.classes_]
    if patient.sor_mod not in classes_sor:
        raise HTTPException(
            status_code=400,
            detail=f"Mode sortie invalide. Valides : {classes_sor}"
        )

    sexe_enc = le_sexe.transform([patient.sexe])[0]
    avc_enc = le_avc.transform([patient.type_avc])[0]
    sor_enc = le_sor.transform([patient.sor_mod])[0]

    X = np.array([[
        patient.age, sexe_enc, avc_enc,
        patient.hta, patient.diabete, patient.fibrillation,
        patient.nb_hospit, patient.duree_sejour, sor_enc
    ]])

    proba = float(modele.predict_proba(X)[0][1])

    if proba >= 0.7:
        niveau_risque = "ELEVE"
    elif proba >= 0.4:
        niveau_risque = "MODERE"
    else:
        niveau_risque = "FAIBLE"

    return {
        "age": patient.age,
        "sexe": patient.sexe,
        "type_avc": patient.type_avc,
        "hta": patient.hta,
        "diabete": patient.diabete,
        "fibrillation": patient.fibrillation,
        "nb_hospit": patient.nb_hospit,
        "duree_sejour": patient.duree_sejour,
        "sor_mod": patient.sor_mod,
        "probabilite_readmission_30j": round(proba, 3),
        "niveau_risque": niveau_risque,
        "interpretation": f"Probabilite de readmission a 30 jours : {round(proba * 100, 1)}%"
    }
