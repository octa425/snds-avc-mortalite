import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

N_PATIENTS = 10000
DATE_DEB = datetime(2023, 1, 1)
DATE_FIN = datetime(2023, 12, 31)

FINESS = ["973000032", "973000040", "973000057",
          "973000065", "973000073", "973000081"]

GHM_AVC = ["01C021", "01C022", "01C023", "01M101",
           "01M102", "01M111", "01M112", "01M131"]

def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

def random_nir():
    return ''.join([str(random.randint(0,9)) for _ in range(15)]) + "A"

def generate_patient(i):
    # 1. Demographie
    age = int(np.random.normal(68, 15))
    age = max(18, min(95, age))
    sexe = np.random.choice(["H", "F"], p=[0.55, 0.45])

    # 2. Comorbidites liees a l'age
    p_hta = 0.70 if age > 65 else 0.35
    hta = np.random.binomial(1, p_hta)

    p_diabete = 0.35 if age > 65 else 0.15
    diabete = np.random.binomial(1, p_diabete)

    p_fa = 0.40 if age > 65 else 0.10
    fibrillation = np.random.binomial(1, p_fa)

    # 3. Type AVC et sejour
    type_avc = np.random.choice(
        ["ischemique", "hemorragique", "ait"],
        p=[0.70, 0.20, 0.10]
    )
    nb_hospit = min(5, np.random.poisson(0.8))
    duree_sejour = int(np.random.lognormal(2.1, 0.6))
    duree_sejour = max(1, min(60, duree_sejour))

    # 4. Mode de sortie PMSI
    # 0=transfert acte, 6=mutation, 7=transfert, 8=domicile, 9=deces
    sor_mod = np.random.choice(
        ["0", "6", "7", "8", "9"],
        p=[0.05, 0.10, 0.20, 0.60, 0.05]
    )

    # 5. Score de risque logit
    if sor_mod == "9":
        readmis_30j = 0  # deces = pas de readmission
    else:
        score = -2.0
        if age > 75:                                score += 0.7
        if type_avc == "hemorragique":              score += 0.6
        if hta == 1:                                score += 0.35
        if diabete == 1:                            score += 0.3
        if fibrillation == 1:                       score += 0.4
        if nb_hospit > 2:                           score += 0.5
        if duree_sejour > 14:                       score += 0.3
        if sor_mod == "8":                          score += 0.2
        if sor_mod == "7":                          score -= 0.3
        if sor_mod == "6":                          score -= 0.2
        if age > 75 and type_avc == "hemorragique": score += 0.4
        prob = 1 / (1 + np.exp(-score))
        readmis_30j = np.random.binomial(1, prob)

    # 6. Identifiants
    nir = random_nir()
    rsa = f"RSA{str(i).zfill(7)}"
    eta = random.choice(FINESS)
    date_entree = random_date(DATE_DEB, DATE_FIN)
    date_sortie = date_entree + timedelta(days=duree_sejour)

    # 7. Code diagnostic selon type AVC
    if type_avc == "ischemique":
        dgn = f"I63{random.randint(0,9)}"
    elif type_avc == "hemorragique":
        dgn = f"I61{random.randint(0,9)}"
    else:
        dgn = "G460"

    return {
        "patient": {
            "age": age, "sexe": sexe, "type_avc": type_avc,
            "hta": hta, "diabete": diabete, "fibrillation": fibrillation,
            "nb_hospit": nb_hospit, "duree_sejour": duree_sejour,
            "sor_mod": sor_mod, "readmis_30j": readmis_30j,
            "nir": nir, "rsa": rsa, "eta": eta,
            "date_entree": date_entree, "date_sortie": date_sortie,
            "dgn": dgn
        }
    }

# Generation des patients
print("Generation des 10 000 patients...")
patients = [generate_patient(i)["patient"] for i in range(N_PATIENTS)]
df = pd.DataFrame(patients)

# ══════════════════════════════════════════
# T_MCO_B — Sejours hospitaliers
# ══════════════════════════════════════════
print("Generation T_MCO_B...")
t_mco_b = pd.DataFrame({
    "ETA_NUM"    : df["eta"],
    "RSA_NUM"    : df["rsa"],
    "NIR_ANO_17" : df["nir"],
    "DGN_PAL"    : df["dgn"],
    "AGE_ANN"    : df["age"],
    "COD_SEX"    : df["sexe"].map({"H": "1", "F": "2"}),
    "EXE_SOI_DTD": df["date_entree"],
    "EXE_SOI_DTF": df["date_sortie"],
    "DUR_SEJ"    : df["duree_sejour"],
    "SOR_MOD"    : df["sor_mod"],
    "GHM_NUM"    : [random.choice(GHM_AVC) for _ in range(N_PATIENTS)],
    "NIR_RET"    : ["0"] * N_PATIENTS,
})
print(f"  T_MCO_B : {len(t_mco_b)} lignes")

# ══════════════════════════════════════════
# T_MCO_D — Comorbidites
# ══════════════════════════════════════════
print("Generation T_MCO_D...")
rows_d = []
for _, row in df.iterrows():
    if row["hta"] == 1:
        rows_d.append({"RSA_NUM": row["rsa"], "ASS_DGN": "I10"})
    if row["diabete"] == 1:
        rows_d.append({"RSA_NUM": row["rsa"], "ASS_DGN": "E119"})
    if row["fibrillation"] == 1:
        rows_d.append({"RSA_NUM": row["rsa"], "ASS_DGN": "I48"})
t_mco_d = pd.DataFrame(rows_d)
print(f"  T_MCO_D : {len(t_mco_d)} lignes")

# ══════════════════════════════════════════
# IR_BEN_R — Referentiel beneficiaires
# ══════════════════════════════════════════
print("Generation IR_BEN_R...")
dcd_dates = []
for _, row in df.iterrows():
    if row["sor_mod"] == "9":
        dcd_dates.append(
            row["date_entree"] + timedelta(days=random.randint(1, 30))
        )
    else:
        dcd_dates.append(None)

ir_ben_r = pd.DataFrame({
    "NIR_ANO_17" : df["nir"],
    "BEN_SEX_COD": df["sexe"].map({"H": "1", "F": "2"}),
    "AGE_ANN"    : df["age"],
    "BEN_DCD_DTE": dcd_dates,
    "BEN_RES_DPT": ["973"] * N_PATIENTS,
    "NIR_RET"    : ["0"] * N_PATIENTS,
})
print(f"  IR_BEN_R : {len(ir_ben_r)} lignes")

# ══════════════════════════════════════════
# ER_PRS_F — Prestations DCIR
# ══════════════════════════════════════════
print("Generation ER_PRS_F...")
rows_prs = []
for _, row in df.iterrows():
    n_consult = 2
    if row["hta"] == 1:          n_consult += 2
    if row["diabete"] == 1:      n_consult += 2
    if row["fibrillation"] == 1: n_consult += 3
    if row["age"] > 75:          n_consult += 2
    if row["sor_mod"] == "8":    n_consult += 1  # domicile = suivi externe

    for _ in range(n_consult):
        date_acte = row["date_sortie"] + timedelta(
            days=random.randint(1, 90)
        )
        rows_prs.append({
            "NIR_ANO_17" : row["nir"],
            "EXE_SOI_DTD": date_acte,
            "PRS_NAT_REF": random.choice(["C", "CS", "V", "VS"]),
            "PSE_SPE_COD": random.choice(["01", "04", "12", "30"]),
            "BSE_REM_MNT": round(random.uniform(23, 50), 2),
            "SOR_MOD"    : row["sor_mod"],
        })
t_mco_prs = pd.DataFrame(rows_prs)
print(f"  ER_PRS_F : {len(t_mco_prs)} lignes")

# ══════════════════════════════════════════
# ER_PHA_F — Pharmacie DCIR
# ══════════════════════════════════════════
print("Generation ER_PHA_F...")
rows_pha = []
MEDICAMENTS = {
    "hta"        : [("C09AA01", "Ramipril", 15.50),
                    ("C07AB02", "Metoprolol", 12.30)],
    "diabete"    : [("A10BA02", "Metformine", 8.20),
                    ("A10BB01", "Glibenclamide", 11.40)],
    "fibrillation": [("B01AF01", "Rivaroxaban", 85.00),
                     ("B01AA03", "Warfarine", 4.50)],
}
for _, row in df.iterrows():
    for comorbidite, meds in MEDICAMENTS.items():
        if row[comorbidite] == 1:
            med = random.choice(meds)
            n_delivrances = random.randint(1, 6)
            for _ in range(n_delivrances):
                date_deliv = row["date_sortie"] + timedelta(
                    days=random.randint(1, 180)
                )
                rows_pha.append({
                    "NIR_ANO_17" : row["nir"],
                    "EXE_SOI_DTD": date_deliv,
                    "PHA_ATC_COD": med[0],
                    "PHA_NOM"    : med[1],
                    "BSE_REM_MNT": med[2],
                })
t_mco_pha = pd.DataFrame(rows_pha)
print(f"  ER_PHA_F : {len(t_mco_pha)} lignes")

# ══════════════════════════════════════════
# SAUVEGARDE
# ══════════════════════════════════════════
os.makedirs("donnees_ml", exist_ok=True)

df_ml = df[[
    "age", "sexe", "type_avc", "hta", "diabete",
    "fibrillation", "nb_hospit", "duree_sejour",
    "sor_mod", "readmis_30j"
]]

df_ml.to_csv("donnees_ml/dataset_ml_avc.csv", index=False)
t_mco_b.to_csv("donnees_ml/T_MCO_B.csv", index=False)
t_mco_d.to_csv("donnees_ml/T_MCO_D.csv", index=False)
ir_ben_r.to_csv("donnees_ml/IR_BEN_R.csv", index=False)
t_mco_prs.to_csv("donnees_ml/ER_PRS_F.csv", index=False)
t_mco_pha.to_csv("donnees_ml/ER_PHA_F.csv", index=False)

print("\n TABLES GENEREES :")
print(f"  dataset_ml_avc.csv : {len(df_ml):>6} patients")
print(f"  T_MCO_B            : {len(t_mco_b):>6} lignes")
print(f"  T_MCO_D            : {len(t_mco_d):>6} lignes")
print(f"  IR_BEN_R           : {len(ir_ben_r):>6} lignes")
print(f"  ER_PRS_F           : {len(t_mco_prs):>6} lignes")
print(f"  ER_PHA_F           : {len(t_mco_pha):>6} lignes")

print(f"\nTaux readmission global :")
print(df_ml["readmis_30j"].value_counts(normalize=True).round(3))
print(f"\nReadmission par type AVC :")
print(df_ml.groupby("type_avc")["readmis_30j"].mean().round(3))
print(f"\nReadmission age > 75 vs <= 75 :")
print(df_ml.groupby(df_ml["age"] > 75)["readmis_30j"].mean().round(3))
print(f"\nReadmission par HTA :")
print(df_ml.groupby("hta")["readmis_30j"].mean().round(3))
print(f"\nReadmission par mode de sortie :")
print(df_ml.groupby("sor_mod")["readmis_30j"].mean().round(3))
