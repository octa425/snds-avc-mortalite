import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

print("=" * 60)
print("ANALYSE MORTALITE AVC — SNDS PMSI")
print("Indicateurs : Survie à J30 et J365 après AVC")
print("=" * 60)

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

print("=" * 60)
print("ANALYSE MORTALITE AVC — SNDS PMSI")
print("Indicateurs : Survie a J30 et J365 apres AVC")
print("=" * 60)

print("\nChargement des tables SNDS...")
mco_b  = pd.read_csv("donnees_snds/T_MCO_B.csv")
mco_c  = pd.read_csv("donnees_snds/T_MCO_C.csv", parse_dates=["EXE_SOI_DTD","EXE_SOI_DTF"])
mco_d  = pd.read_csv("donnees_snds/T_MCO_D.csv")
mco_gv = pd.read_csv("donnees_snds/T_MCO_GV.csv")
ir_ben = pd.read_csv("donnees_snds/IR_BEN_R.csv", parse_dates=["BEN_DCD_DTE"])

print(f"  T_MCO_B  : {len(mco_b):>6} lignes")
print(f"  T_MCO_C  : {len(mco_c):>6} lignes")
print(f"  T_MCO_D  : {len(mco_d):>6} lignes")
print(f"  T_MCO_GV : {len(mco_gv):>6} lignes")
print(f"  IR_BEN_R : {len(ir_ben):>6} lignes")

FINESS_OUT = [
    '130780521','130783236','130783293','130784234','130804297','130784259',
    '600100101','750041543','750100018','750100042','750100075','750100083',
    '750100091','750100109','750100125','750100166','750100208','750100216',
    '750100232','750100273','750100299','750801441','750803447','750803454',
    '910100015','910100023','920100013','920100021','920100039','920100047',
    '920100054','920100062','930100011','930100037','930100045','940100027',
    '940100035','940100043','940100050','940100068','950100016','690783154',
    '690784137','690784152','690784178','690787478','830100558'
]
SEANCES_GHM = ["17M05","17M06","23M09","23M14","17K04","17K05","17K08","17K09","11K02"]

print("\n" + "="*60)
print("ETAPE 1 — Filtres qualite")
print("="*60)

df = mco_b.merge(mco_c, on=["ETA_NUM","RSA_NUM"], how="left")
df = df.merge(mco_gv, on=["ETA_NUM","RSA_NUM"], how="left")
print(f"  Avant filtres              : {len(df):>6} sejours")
df = df[df["BDI_DEP"] == "9C"]
print(f"  Apres BDI_DEP=9C           : {len(df):>6} sejours")
df = df[~df["ETA_NUM"].astype(str).isin(FINESS_OUT)]
print(f"  Apres exclusion FINESS     : {len(df):>6} sejours")
df = df[df["GHM_NUM"].str[:2] != "90"]
print(f"  Apres exclusion CMD90      : {len(df):>6} sejours")
df = df[df["SEJ_TYP"].isna() | (df["SEJ_TYP"] != "B")]
print(f"  Apres exclusion SEJ_TYP=B  : {len(df):>6} sejours")
df = df[df["GHM_NUM"].str[:2] != "28"]
df = df[~df["GHM_NUM"].str[:5].isin(SEANCES_GHM)]
print(f"  Apres exclusion CMD28      : {len(df):>6} sejours")
df = df[df["BDI_COD"].str[2:] != "999"]
print(f"  Apres exclusion BDI999     : {len(df):>6} sejours")
df = df[df["AGE_ANN"] >= 18]
print(f"  Apres AGE >= 18            : {len(df):>6} sejours")
df = df[~df["NIR_ANO_17"].isin(["xxxxxxxxxxxxxxxxx","XXXXXXXXXXXXXXXXS"])]
for col in ["NIR_RET","NAI_RET","SEX_RET","DAT_RET","SEJ_RET","FHO_RET","PMS_RET"]:
    df = df[df[col].astype(str) == "0"]
print(f"  Apres qualite NIR (7 ctrl) : {len(df):>6} sejours")
print(f"\n  FILTRESS973 : {len(df)} sejours MCO Guyane adultes valides")

print("\n" + "="*60)
print("ETAPE 2 — Selection cohorte AVC")
print("="*60)

CODES_AVC = ["I60","I61","I62","I63","I64"]
MAP_DIAG = {
    "I60":"Hemorragie meningee",
    "I61":"Hemorragie intracerebrales",
    "I62":"Autres hemorragies intracraniennes",
    "I63":"Infarctus cerebral",
    "I64":"AVC non precise",
}

def is_avc_dp(code):
    if pd.isna(code): return False
    return any(str(code).startswith(c) for c in CODES_AVC)

def is_g46_dp(code):
    if pd.isna(code): return False
    return str(code).startswith("G46")

da_avc = mco_d[mco_d["ASS_DGN"].str[:3].isin(CODES_AVC)][["ETA_NUM","RSA_NUM"]].drop_duplicates()

avc_dp = df[df["DGN_PAL"].apply(is_avc_dp)].copy()
avc_dp["DIAGNOSTIC"] = avc_dp["DGN_PAL"].str[:3].map(MAP_DIAG)

g46_dp = df[df["DGN_PAL"].apply(is_g46_dp)].copy()
g46_avec_da = g46_dp.merge(da_avc, on=["ETA_NUM","RSA_NUM"], how="inner")
g46_avec_da["DIAGNOSTIC"] = "Syndrome vasculaire cerebral"

cohorte_avc = pd.concat([avc_dp, g46_avec_da], ignore_index=True)
cohorte_avc = cohorte_avc.drop_duplicates(subset=["ETA_NUM","RSA_NUM"])
cohorte_avc["TOP_HOSPavc"] = 1

print(f"  AVC en DP (I60-I64)        : {len(avc_dp):>6} sejours")
print(f"  G46 + AVC en DA            : {len(g46_avec_da):>6} sejours")
print(f"  Cohorte AVC totale         : {len(cohorte_avc):>6} sejours")
print("\n  Distribution par type d'AVC :")
print(cohorte_avc["DIAGNOSTIC"].value_counts().to_string())

print("\n" + "="*60)
print("ETAPE 3 — Classes d age")
print("="*60)

def classe_age(age):
    if 18 <= age < 65:  return "1 — 18-64 ans"
    if 65 <= age < 85:  return "2 — 65-84 ans"
    if age >= 85:       return "3 — 85 ans et +"
    return None

cohorte_avc["CL_AGE"] = cohorte_avc["AGE_ANN"].apply(classe_age)
print(cohorte_avc["CL_AGE"].value_counts().sort_index().to_string())

print("\n" + "="*60)
print("ETAPE 4 — Jointure IR_BEN_R")
print("="*60)

cohorte_avc = cohorte_avc.merge(
    ir_ben[["NIR_ANO_17","BEN_DCD_DTE","BEN_SEX_COD","BEN_RES_DPT"]],
    on="NIR_ANO_17", how="left"
)
patients_avc = cohorte_avc.sort_values("EXE_SOI_DTD", ascending=True)
patients_avc = patients_avc.drop_duplicates(subset=["NIR_ANO_17"])
print(f"  Sejours AVC totaux         : {len(cohorte_avc):>6}")
print(f"  Patients uniques AVC       : {len(patients_avc):>6}")

print("\n" + "="*60)
print("ETAPE 5 — Indicateurs de survie J30 et J365")
print("="*60)

DATE_FIN_SUIVI = pd.Timestamp("2019-12-31")

def calcul_survie(row):
    date_avc = row["EXE_SOI_DTD"]
    if pd.isna(date_avc):
        return pd.Series({"DELAI_JOURS":0,"EVENT":0,"DECES_J30":0,"DECES_J365":0,"SURVIE_J30":1,"SURVIE_J365":1})
    if pd.isna(row["BEN_DCD_DTE"]):
        delai = (DATE_FIN_SUIVI - date_avc).days
        return pd.Series({"DELAI_JOURS":max(0,delai),"EVENT":0,"DECES_J30":0,"DECES_J365":0,"SURVIE_J30":1,"SURVIE_J365":1})
    else:
        delai = max(0,(row["BEN_DCD_DTE"] - date_avc).days)
        d30  = 1 if delai <= 30  else 0
        d365 = 1 if delai <= 365 else 0
        return pd.Series({"DELAI_JOURS":delai,"EVENT":1,"DECES_J30":d30,"DECES_J365":d365,"SURVIE_J30":1-d30,"SURVIE_J365":1-d365})

patients_avc = patients_avc.copy()
patients_avc[["DELAI_JOURS","EVENT","DECES_J30","DECES_J365","SURVIE_J30","SURVIE_J365"]] = patients_avc.apply(calcul_survie, axis=1)

n_tot  = len(patients_avc)
n_j30  = int(patients_avc["DECES_J30"].sum())
n_j365 = int(patients_avc["DECES_J365"].sum())
s_j30  = int(patients_avc["SURVIE_J30"].sum())
s_j365 = int(patients_avc["SURVIE_J365"].sum())

print(f"\n  Patients AVC               : {n_tot:>6}")
print(f"\n  MORTALITE")
print(f"  Deces a J30               : {n_j30:>6} ({n_j30/n_tot*100:.1f}%)")
print(f"  Deces a J365              : {n_j365:>6} ({n_j365/n_tot*100:.1f}%)")
print(f"\n  SURVIE (criteres de jugement)")
print(f"  Vivants a J30             : {s_j30:>6} ({s_j30/n_tot*100:.1f}%)")
print(f"  Vivants a J365            : {s_j365:>6} ({s_j365/n_tot*100:.1f}%)")

print("\n" + "="*60)
print("ETAPE 6 — Kaplan-Meier")
print("="*60)

kmf = KaplanMeierFitter()
kmf.fit(durations=patients_avc["DELAI_JOURS"], event_observed=patients_avc["EVENT"], label="Survie apres AVC")
print(f"\n  Probabilite survie J30     : {float(kmf.predict(30)):.1%}")
print(f"  Probabilite survie J365    : {float(kmf.predict(365)):.1%}")
print(f"  Mediane de survie          : {kmf.median_survival_time_} jours")

print("\n  Survie par sexe :")
for sexe, label in [("1","Homme"),("2","Femme")]:
    mask = patients_avc["COD_SEX"].astype(str) == sexe
    if mask.sum() > 0:
        kmf_s = KaplanMeierFitter()
        kmf_s.fit(durations=patients_avc.loc[mask,"DELAI_JOURS"], event_observed=patients_avc.loc[mask,"EVENT"])
        print(f"  {label:8} J30: {float(kmf_s.predict(30)):.1%}  J365: {float(kmf_s.predict(365)):.1%}")

mask_h = patients_avc["COD_SEX"].astype(str) == "1"
mask_f = patients_avc["COD_SEX"].astype(str) == "2"
if mask_h.sum() > 0 and mask_f.sum() > 0:
    lr = logrank_test(
        patients_avc.loc[mask_h,"DELAI_JOURS"],
        patients_avc.loc[mask_f,"DELAI_JOURS"],
        event_observed_A=patients_avc.loc[mask_h,"EVENT"],
        event_observed_B=patients_avc.loc[mask_f,"EVENT"]
    )
    print(f"\n  Log-rank test H vs F       : p-value = {lr.p_value:.4f}")
    if lr.p_value < 0.05:
        print("  Difference significative entre hommes et femmes")
    else:
        print("  Pas de difference significative (p >= 0.05)")

print("\n  Survie par classe d age :")
for cl in ["1 — 18-64 ans","2 — 65-84 ans","3 — 85 ans et +"]:
    mask = patients_avc["CL_AGE"] == cl
    if mask.sum() > 0:
        kmf_a = KaplanMeierFitter()
        kmf_a.fit(durations=patients_avc.loc[mask,"DELAI_JOURS"], event_observed=patients_avc.loc[mask,"EVENT"])
        print(f"  {cl:20} J30: {float(kmf_a.predict(30)):.1%}  J365: {float(kmf_a.predict(365)):.1%}  (n={mask.sum()})")

print("\n" + "="*60)
print("ETAPE 7 — Comorbidites")
print("="*60)

MAP_COMORBIDITES = {
    "I10":"HTA","E11":"Diabete type 2","I48":"Fibrillation auriculaire",
    "I25":"Cardiopathie ischemique","E78":"Dyslipidemie","J44":"BPCO",
    "N18":"Insuffisance renale","I50":"Insuffisance cardiaque",
    "E14":"Diabete non precise","I20":"Angor",
}

avc_eta_rsa = cohorte_avc[["ETA_NUM","RSA_NUM","NIR_ANO_17"]].copy()
comorbidites_avc = avc_eta_rsa.merge(mco_d, on=["ETA_NUM","RSA_NUM"], how="left")

for code, nom in MAP_COMORBIDITES.items():
    patients_avec = comorbidites_avc.loc[
        comorbidites_avc["ASS_DGN"].str[:3] == code, "NIR_ANO_17"
    ].unique()
    patients_avc[f"TOP_{code}"] = patients_avc["NIR_ANO_17"].isin(patients_avec).astype(int)
    n_c = patients_avc[f"TOP_{code}"].sum()
    print(f"  {nom:35} ({code}) : {n_c:>4} patients ({n_c/n_tot*100:.1f}%)")

print("\n" + "="*60)
print("ETAPE 8 — Regression logistique (mortalite J365)")
print("="*60)

top_cols = [f"TOP_{c}" for c in MAP_COMORBIDITES.keys()]
df_reg = patients_avc.copy()
df_reg["SEXE_NUM"] = (df_reg["COD_SEX"].astype(str) == "1").astype(int)
df_reg["AGE_NUM"]  = df_reg["AGE_ANN"]
le_diag = LabelEncoder()
df_reg["DIAG_NUM"] = le_diag.fit_transform(df_reg["DIAGNOSTIC"].fillna("Inconnu"))

features = ["AGE_NUM","SEXE_NUM","DIAG_NUM"] + top_cols
target   = "DECES_J365"
df_model = df_reg[features + [target]].dropna()
X = df_model[features]
y = df_model[target]

print(f"\n  Dataset modele : {len(df_model)} patients")
print(f"  Deces J365     : {int(y.sum())} ({y.mean()*100:.1f}%)")
print(f"  Vivants J365   : {int((1-y).sum())} ({(1-y.mean())*100:.1f}%)")

modele_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
modele_lr.fit(X, y)

coefs = pd.DataFrame({
    "Variable"   : features,
    "Coefficient": modele_lr.coef_[0],
    "Odds Ratio" : np.exp(modele_lr.coef_[0])
}).sort_values("Odds Ratio", ascending=False)

print("\n  Coefficients (facteurs associes a la mortalite J365) :")
print(coefs.to_string(index=False))

y_pred = modele_lr.predict(X)
print("\n  Performance du modele :")
print(classification_report(y, y_pred, target_names=["Vivant","Decede"]))

patients_avc.to_csv("donnees_snds/cohorte_avc_finale.csv", index=False)
coefs.to_csv("donnees_snds/regression_logistique_j365.csv", index=False)

print("\n" + "="*60)
print("RESUME FINAL")
print("="*60)
print(f"  Sejours MCO Guyane analyses : {len(df)}")
print(f"  Patients AVC identifies     : {n_tot}")
print(f"  Survie a J30                : {s_j30/n_tot*100:.1f}%")
print(f"  Survie a J365               : {s_j365/n_tot*100:.1f}%")
print(f"  Mortalite a J30             : {n_j30/n_tot*100:.1f}%")
print(f"  Mortalite a J365            : {n_j365/n_tot*100:.1f}%")
print(f"\n  Cohorte finale    : donnees_snds/cohorte_avc_finale.csv")
print(f"  Regression J365   : donnees_snds/regression_logistique_j365.csv")
