
# ============================================================
# API de prediction de readmission post-AVC a 30 jours
# Modele : XGBoost entraine sur 10 000 patients simules SNDS
# ============================================================

# --- Imports ---
# FastAPI : framework pour creer l'API REST
# HTTPException : pour retourner des erreurs HTTP explicites
from fastapi import FastAPI, HTTPException

# BaseModel : classe de base pour valider les donnees d'entree
# Pydantic verifie automatiquement les types (int, str...)
from pydantic import BaseModel

# joblib : charge les fichiers .pkl (modele et encodeurs)
import joblib

# numpy : manipulation des tableaux numeriques
# Utilise pour construire le vecteur de features X
import numpy as np

# ============================================================
# CHARGEMENT DU MODELE ET DES ENCODEURS AU DEMARRAGE
# ============================================================
# On charge une seule fois au demarrage, pas a chaque requete
# Si les .pkl sont absents, l'API demarre en mode degrade
# (elle repond mais ne peut pas faire de prediction)
try:
    # Modele XGBoost entraine (434 KB)
    modele = joblib.load("modele_xgb_avc.pkl")
    
    # Encodeur pour la variable "sexe" (H → 0, F → 1)
    le_sexe = joblib.load("encodeur_sexe.pkl")
    
    # Encodeur pour la variable "type_avc"
    # (ait → 0, hemorragique → 1, ischemique → 2)
    le_avc = joblib.load("encodeur_type_avc.pkl")
    
    # Encodeur pour la variable "sor_mod"
    # (0, 6, 7, 8, 9 → valeurs numeriques ordonnees)
    le_sor = joblib.load("encodeur_sor_mod.pkl")
    print("Modele et encodeurs charges avec succes !")
    
except FileNotFoundError:
    
    # Mode degrade : l'API demarre sans modele
    # Elle retourne un message d'erreur explicite
    # au lieu de planter completement
    
    modele = None
    print("AVERTISSEMENT : modele introuvable")
    
# ============================================================
# CREATION DE L'APPLICATION FASTAPI
# ============================================================
# Le titre, la description et la version apparaissent
# automatiquement dans l'interface Swagger (/docs)

app = FastAPI(
    title="API Prediction Readmission Post-AVC",
    description="Predit le risque de readmission a 30 jours apres un AVC",
    version="1.0.0"
)

# ============================================================
# SCHEMA DES DONNEES D'ENTREE (validation automatique)
# ============================================================
# Pydantic valide automatiquement chaque champ.
# Si un champ est manquant ou a un mauvais type,
# FastAPI retourne une erreur 422 sans qu'on ecrive
# une seule ligne de validation manuelle.

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

# ============================================================
# ENDPOINT GET / — Page d'accueil
# ============================================================
# Retourne les informations de base sur l'API
# Accessible via : GET https://snds-avc-api.onrender.com/

@app.get("/")
def accueil():
    return {
        "message": "API Prediction Readmission Post-AVC",
        "version": "1.0.0",
        "modele": "XGBoost"
    }

# ============================================================
# ENDPOINT GET /health — Verification de sante
# ============================================================
# Utilise par Docker (healthcheck) et les outils de monitoring
# pour verifier que le service est operationnel.
# Un endpoint /health est indispensable en production.
# Accessible via : GET https://snds-avc-api.onrender.com/health

@app.get("/health")
def health():
    return {
        "statut": "OK",
        "modele": "XGBoost",
        "projet": "SNDS AVC Readmission"
    }
    
# ============================================================
# ENDPOINT POST /predict  Prediction de readmission
# ============================================================
# Recoit un objet JSON (DonneesPatient), valide les donnees,
# encode les variables categorielles, construit le vecteur
# de features et retourne la prediction du modele XGBoost.
# Accessible via : POST https://snds-avc-api.onrender.com/predict

@app.post("/predict")

# --- Verification mode degrade ---
    # Si le modele n'a pas pu etre charge au demarrage,
    # on retourne un message d'erreur explicite (pas un plantage)
    
def predire_readmission(patient: DonneesPatient):
    if modele is None:
        return {
            "erreur": "Modele non disponible",
            "solution": "Lancez sauvegarder_modele_avc.py"
        }
        
# --- Validation metier : sexe ---
    # Pydantic valide que sexe est une string (type)
    # mais pas que sa valeur est "H" ou "F" (metier).
    # On verifie manuellement contre les classes connues
    # par l'encodeur. Erreur 400 = erreur metier.
    
    if patient.sexe not in le_sexe.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Sexe invalide. Valides : {list(le_sexe.classes_)}"
        )
        
  # --- Validation metier : type_avc ---
    # Meme logique : on verifie que la valeur est
    # dans les classes connues par l'encodeur
    # ("ait", "hemorragique", "ischemique")
    
    if patient.type_avc not in le_avc.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Type AVC invalide. Valides : {list(le_avc.classes_)}"
        )
        
# --- Validation metier : sor_mod ---
    # L'encodeur stocke les classes comme np.int64
    # (ex: np.int64(8)) mais patient.sor_mod est un int Python.
    # On convertit les classes en int Python pour la comparaison.
    # Valides : 0, 6, 7, 8, 9
    
    classes_sor = [int(c) for c in le_sor.classes_]
    if patient.sor_mod not in classes_sor:
        raise HTTPException(
            status_code=400,
            detail=f"Mode sortie invalide. Valides : {classes_sor}"
        )
        
# --- Encodage des variables categorielles ---
    # Le modele XGBoost ne comprend que des nombres.
    # LabelEncoder transforme les valeurs texte en entiers :
    # "H" → 0, "F" → 1
    # "ait" → 0, "hemorragique" → 1, "ischemique" → 2
    # [0] recupere le premier (et seul) element du tableau
    
    sexe_enc = le_sexe.transform([patient.sexe])[0]
    avc_enc = le_avc.transform([patient.type_avc])[0]
    sor_enc = le_sor.transform([patient.sor_mod])[0]
    
# --- Construction du vecteur de features ---
    # Le modele attend exactement 9 features dans cet ordre :
    # age, sexe_enc, type_avc_enc, hta, diabete,
    # fibrillation, nb_hospit, duree_sejour, sor_mod_enc
    # np.array cree un tableau 2D de forme (1, 9)
    # car predict_proba attend un tableau 2D
    
    X = np.array([[
        patient.age, sexe_enc, avc_enc,
        patient.hta, patient.diabete, patient.fibrillation,
        patient.nb_hospit, patient.duree_sejour, sor_enc
    ]])
    
# --- Inference (prediction) ---
    # predict_proba retourne les probabilites pour chaque classe
    # [0] = premier patient (on en envoie un seul)
    # [1] = probabilite de la classe 1 (readmis = oui)
    # float() convertit numpy.float64 en float Python standard
    
    proba = float(modele.predict_proba(X)[0][1])
    
# --- Stratification du risque ---
    # Seuils empiriques definis pour la demonstration.
    # Ne constituent pas des seuils cliniques valides.
    
    if proba >= 0.7:
        niveau_risque = "ELEVE"
    elif proba >= 0.4:
        niveau_risque = "MODERE"
    else:
        niveau_risque = "FAIBLE"
        
# --- Retour de la reponse ---
    # On retourne toutes les donnees d'entree + la prediction
    # round(proba, 3) : arrondi a 3 decimales (ex: 0.894)
    # round(proba * 100, 1) : en pourcentage (ex: 89.4%)
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
