import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("TEST DE ROBUSTESSE — Regression logistique")
print("Mortalite a J365 apres AVC — Cohorte SNDS Guyane 2019")
print("=" * 60)

# ──────────────────────────────────────────────
# CHARGEMENT
# ──────────────────────────────────────────────
path_cohorte = os.path.expanduser(
    "~/projets_ml/pmsi_prediction/donnees_snds/cohorte_avc_finale.csv"
)
path_mcod = os.path.expanduser(
    "~/projets_ml/pmsi_prediction/donnees_snds/T_MCO_D.csv"
)

df    = pd.read_csv(path_cohorte)
mco_d = pd.read_csv(path_mcod)
print(f"\n  {len(df)} patients AVC charges")

# ──────────────────────────────────────────────
# RECONSTRUCTION DES VARIABLES
# ──────────────────────────────────────────────
MAP_COMORBIDITES = {
    "I10": "HTA",
    "E11": "Diabete_T2",
    "I48": "FA",
    "I25": "Cardiopathie",
    "E78": "Dyslipidemie",
    "J44": "BPCO",
    "N18": "IRC",
    "I50": "IC",
    "E14": "Diabete_NP",
    "I20": "Angor",
}

avc_eta = df[["ETA_NUM","RSA_NUM","NIR_ANO_17"]].copy()
comorbidites = avc_eta.merge(mco_d, on=["ETA_NUM","RSA_NUM"], how="left")

for code, nom in MAP_COMORBIDITES.items():
    patients_avec = comorbidites.loc[
        comorbidites["ASS_DGN"].str[:3] == code, "NIR_ANO_17"
    ].unique()
    df[f"TOP_{code}"] = df["NIR_ANO_17"].isin(patients_avec).astype(int)

df["SEXE_NUM"] = (df["COD_SEX"].astype(str) == "1").astype(int)
df["AGE_NUM"]  = df["AGE_ANN"]
le = LabelEncoder()
df["DIAG_NUM"] = le.fit_transform(df["DIAGNOSTIC"].fillna("Inconnu"))

top_cols = [f"TOP_{c}" for c in MAP_COMORBIDITES.keys()]
features = ["AGE_NUM","SEXE_NUM","DIAG_NUM"] + top_cols
target   = "DECES_J365"

df_model = df[features + [target]].dropna()
X = df_model[features].values
y = df_model[target].values

print(f"  Deces J365 : {int(y.sum())} ({y.mean()*100:.1f}%)")
print(f"  Vivants    : {int((1-y).sum())} ({(1-y.mean())*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# TEST 1 — CROSS-VALIDATION 5 FOLDS
# ══════════════════════════════════════════════════════════════
# On divise les 400 patients en 5 groupes de 80
# A chaque iteration on entraine sur 4 groupes et on teste sur 1
# Si resultats stables -> modele robuste
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TEST 1 — Cross-validation stratifiee 5 folds")
print("="*60)

modele = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy" : "accuracy",
    "recall"   : "recall",
    "precision": "precision",
    "f1"       : "f1",
    "roc_auc"  : "roc_auc",
}

resultats_cv = cross_validate(
    modele, X, y,
    cv=cv,
    scoring=scoring,
    return_train_score=False
)

print(f"\n  {'Fold':<6} {'Accuracy':<10} {'Recall':<10} {'Precision':<12} {'F1':<8} {'AUC'}")
print("  " + "-"*56)

for i in range(5):
    print(f"  Fold {i+1:<3} "
          f"{resultats_cv['test_accuracy'][i]:.3f}      "
          f"{resultats_cv['test_recall'][i]:.3f}      "
          f"{resultats_cv['test_precision'][i]:.3f}         "
          f"{resultats_cv['test_f1'][i]:.3f}     "
          f"{resultats_cv['test_roc_auc'][i]:.3f}")

print("\n  Resume (moyenne +/- ecart-type) :")
for metric in ["accuracy","recall","precision","f1","roc_auc"]:
    vals = resultats_cv[f"test_{metric}"]
    print(f"  {metric:<12} : {vals.mean():.3f} +/- {vals.std():.3f}")

std_auc = resultats_cv['test_roc_auc'].std()
print("\n  Interpretation :")
if std_auc < 0.05:
    print("  OK  Modele ROBUSTE — ecart-type AUC < 0.05")
elif std_auc < 0.10:
    print("  /!\\ Modele MODEREMENT STABLE — ecart-type entre 0.05 et 0.10")
else:
    print("  !! Modele INSTABLE — ecart-type AUC > 0.10")

# ══════════════════════════════════════════════════════════════
# TEST 2 — STABILITE DES ODDS RATIO (100 bootstraps)
# ══════════════════════════════════════════════════════════════
# On tire aleatoirement 80% des patients 100 fois
# On calcule les OR a chaque iteration
# IC95% etroit -> facteur de risque confirme
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TEST 2 — Stabilite des Odds Ratio (100 bootstraps)")
print("="*60)

n_bootstrap = 100
n_sample    = int(len(X) * 0.8)
coefs_boot  = []

np.random.seed(42)
for i in range(n_bootstrap):
    idx    = np.random.choice(len(X), size=n_sample, replace=True)
    X_boot = X[idx]
    y_boot = y[idx]
    if y_boot.sum() < 5:
        continue
    m = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=i
    )
    m.fit(X_boot, y_boot)
    coefs_boot.append(m.coef_[0])

coefs_boot = np.array(coefs_boot)
or_boot    = np.exp(coefs_boot)

print(f"\n  {len(coefs_boot)} bootstraps realises\n")
print(f"  {'Variable':<15} {'OR moyen':<12} {'IC95% bas':<12} {'IC95% haut':<12} Stabilite")
print("  " + "-"*65)

for j, feat in enumerate(features):
    or_moy  = or_boot[:, j].mean()
    ic_bas  = np.percentile(or_boot[:, j], 2.5)
    ic_haut = np.percentile(or_boot[:, j], 97.5)
    stable  = "OK Stable" if (ic_bas > 1 or ic_haut < 1) else "Incertain"
    print(f"  {feat:<15} {or_moy:<12.3f} {ic_bas:<12.3f} {ic_haut:<12.3f} {stable}")

# ══════════════════════════════════════════════════════════════
# TEST 3 — AUC-ROC GLOBAL
# ══════════════════════════════════════════════════════════════
# AUC = capacite du modele a distinguer decedes vs vivants
# 0.5 = hasard | 0.7 = acceptable | 0.8 = bon | 0.9 = excellent
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("TEST 3 — AUC-ROC global")
print("="*60)

modele.fit(X, y)
y_proba = modele.predict_proba(X)[:, 1]
auc = roc_auc_score(y, y_proba)

print(f"\n  AUC-ROC global : {auc:.3f}")
if auc >= 0.8:
    print("  OK  BON — modele discriminant")
elif auc >= 0.7:
    print("  /!\\ ACCEPTABLE — modele moderement discriminant")
else:
    print("  !! FAIBLE — ajouter des variables explicatives")

# ══════════════════════════════════════════════════════════════
# RESUME FINAL
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("RESUME FINAL — Robustesse du modele")
print("="*60)
print(f"""
  Cross-validation (5 folds) :
    AUC moyen    : {resultats_cv['test_roc_auc'].mean():.3f} +/- {resultats_cv['test_roc_auc'].std():.3f}
    Recall moyen : {resultats_cv['test_recall'].mean():.3f} +/- {resultats_cv['test_recall'].std():.3f}
    F1 moyen     : {resultats_cv['test_f1'].mean():.3f} +/- {resultats_cv['test_f1'].std():.3f}

  Bootstrap (100 iterations) :
    OR calcules sur 80% des donnees — 100 repetitions
    IC95% calcule pour chaque variable

  AUC global : {auc:.3f}

  Limites :
    - Donnees simulees (400 patients)
    - Sur donnees reelles : n > 1000 recommande
    - Variables manquantes : delai prise en charge,
      score de severite, donnees biologiques
    - Validation externe recommandee (autre periode/region)
""")

# Sauvegarde
path_out = os.path.expanduser(
    "~/projets_ml/pmsi_prediction/donnees_snds/robustesse_cv.csv"
)
pd.DataFrame({
    "Fold"     : range(1, 6),
    "Accuracy" : resultats_cv["test_accuracy"],
    "Recall"   : resultats_cv["test_recall"],
    "Precision": resultats_cv["test_precision"],
    "F1"       : resultats_cv["test_f1"],
    "AUC_ROC"  : resultats_cv["test_roc_auc"],
}).to_csv(path_out, index=False)

print(f"  OK robustesse_cv.csv sauvegarde")
