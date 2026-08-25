# Prediction de readmission post-AVC a 30 jours

## Contexte

Projet de Machine Learning applique aux donnees de sante,
inspire du modele SNDS (Systeme National des Donnees de Sante).

10 000 patients simules avec une structure causale controlee :
les relations entre facteurs de risque et readmission sont
introduites dans la simulation, puis verifiees par les modeles.

## Donnees simulees

Tables inspirees du modele SNDS :

| Table | Description | Lignes |
|-------|-------------|--------|
| T_MCO_B | Sejours hospitaliers | 10 000 |
| T_MCO_D | Comorbidites (CIM-10) | 10 729 |
| IR_BEN_R | Referentiel beneficiaires | 10 000 |
| ER_PRS_F | Prestations DCIR | 56 053 |
| ER_PHA_F | Pharmacie DCIR | 37 452 |

## Variables du modele

| Variable | Description |
|----------|-------------|
| age | Age du patient (18-95 ans) |
| sexe | H / F |
| type_avc | ischemique / hemorragique / ait |
| hta | Hypertension arterielle (0/1) |
| diabete | Diabete (0/1) |
| fibrillation | Fibrillation auriculaire (0/1) |
| nb_hospit | Nombre d hospitalisations precedentes (0-5) |
| duree_sejour | Duree du sejour en jours |
| sor_mod | Mode de sortie PMSI (0/6/7/8/9) |

### Codes mode de sortie (SOR_MOD)

| Code | Signification |
|------|---------------|
| 0 | Transfert pour acte |
| 6 | Mutation interne |
| 7 | Transfert etablissement |
| 8 | Domicile |
| 9 | Deces |

## Relations causales introduites

| Facteur | Impact sur le risque |
|---------|---------------------|
| age > 75 | +70% |
| AVC hemorragique | +60% |
| Fibrillation auriculaire | +40% |
| nb_hospit > 2 | +50% |
| duree_sejour > 14 jours | +30% |
| HTA | +35% |
| Diabete | +30% |
| Retour domicile (sor_mod=8) | +20% |
| Transfert (sor_mod=7) | -30% |

## Comparaison des modeles

| Modele | AUC-ROC | PR-AUC | CV AUC (5-fold) |
|--------|---------|--------|-----------------|
| XGBoost | 0.687 | 0.419 | 0.657 +/- 0.004 |
| LightGBM | 0.679 | 0.410 | 0.640 +/- 0.009 |
| MLP | 0.666 | 0.432 | 0.648 +/- 0.017 |
| Random Forest | 0.665 | 0.361 | 0.624 +/- 0.008 |
| Logistic Regression | 0.658 | 0.400 | 0.635 +/- 0.016 |

XGBoost est le modele retenu (meilleur AUC-ROC et CV stable).

## Importance des variables (Random Forest)

| Variable | Importance |
|----------|-----------|
| age | 37.0% |
| duree_sejour | 29.2% |
| sor_mod | 9.8% |
| nb_hospit | 9.0% |
| type_avc | 3.8% |
| sexe | 3.8% |
| diabete | 2.7% |
| hta | 2.4% |
| fibrillation | 2.3% |

## API deployee publiquement

URL : https://snds-avc-api.onrender.com/docs

### Tester une prediction

Patient a risque ELEVE :
```json
{
  "age": 78,
  "sexe": "H",
  "type_avc": "hemorragique",
  "hta": 1,
  "diabete": 1,
  "fibrillation": 1,
  "nb_hospit": 3,
  "duree_sejour": 20,
  "sor_mod": 8
}
```
Resultat : 89.4% — Risque ELEVE

Patient a risque FAIBLE :
```json
{
  "age": 35,
  "sexe": "F",
  "type_avc": "ait",
  "hta": 0,
  "diabete": 0,
  "fibrillation": 0,
  "nb_hospit": 0,
  "duree_sejour": 3,
  "sor_mod": 7
}
```
Resultat : 12.9% — Risque FAIBLE

## Seuils de stratification du risque

| Niveau | Probabilite |
|--------|------------|
| FAIBLE | < 40% |
| MODERE | 40% - 70% |
| ELEVE | >= 70% |

Ces seuils sont empiriques et definis pour
les besoins de la demonstration technique.
Ils ne constituent pas des seuils cliniques valides.

## Stack technique

- Python 3.11
- XGBoost, LightGBM, scikit-learn
- FastAPI + Uvicorn
- Docker
- GitHub Actions (CI/CD)
- Render (deploiement)

## Lancer le projet en local

```bash
# Generer les donnees
python3 generer_dataset_ml.py

# Entrainer et comparer les modeles
python3 modelisation_ml.py

# Sauvegarder le meilleur modele
python3 sauvegarder_modele_avc.py

# Lancer l'API
uvicorn api_prediction_avc:app --reload --port 8001

# Ou avec Docker
docker build -t snds-avc-api .
docker run -p 8000:8000 snds-avc-api
```

## Disclaimer

Toutes les donnees utilisees sont entierement simulees.
Aucune donnee patient reelle n'a ete utilisee.
Ce projet est un prototype a visee pedagogique et technique.
