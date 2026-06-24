import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("ANALYSE DES DEPENSES POST-AVC — DCIR")
print("Pharmacie · Medecins · Hospitalisation")
print("=" * 60)

# ══════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════
print("\nChargement des tables...")
cohorte = pd.read_csv("donnees_snds/cohorte_avc_finale.csv",
                       parse_dates=["EXE_SOI_DTD","EXE_SOI_DTF"])
er_prs  = pd.read_csv("donnees_snds/ER_PRS_F.csv",
                       parse_dates=["EXE_SOI_DTD"])
er_pha  = pd.read_csv("donnees_snds/ER_PHA_F.csv")

print(f"  Cohorte AVC : {len(cohorte)} patients")
print(f"  ER_PRS_F    : {len(er_prs)} prestations")
print(f"  ER_PHA_F    : {len(er_pha)} delivrances")

# ══════════════════════════════════════════
# JOINTURE COHORTE + PRESTATIONS
# ══════════════════════════════════════════
print("\n" + "="*60)
print("ETAPE 1 — Jointure cohorte AVC + ER_PRS_F")
print("="*60)

# Filtres DCIR obligatoires
er_prs = er_prs[er_prs["DPN_QLF"] != 71]
er_prs = er_prs[er_prs["CPL_MAJ_TOP"] < 2]

# Jointure sur NIR
df = cohorte[[
    "NIR_ANO_17","DIAGNOSTIC","AGE_ANN","COD_SEX",
    "CL_AGE","EXE_SOI_DTD","DECES_J30","DECES_J365",
    "SURVIE_J30","SURVIE_J365"
]].merge(
    er_prs.rename(columns={
        "BEN_NIR_PSA" : "NIR_ANO_17",
        "EXE_SOI_DTD" : "DATE_PRESTA"
    }),
    on="NIR_ANO_17",
    how="left"
)

print(f"  Prestations apres jointure : {len(df)}")
print(f"  Patients avec prestations  : {df['NIR_ANO_17'].nunique()}")

# ══════════════════════════════════════════
# ETAPE 2 — CATEGORIES DE DEPENSES
# ══════════════════════════════════════════
print("\n" + "="*60)
print("ETAPE 2 — Categories de depenses")
print("="*60)

def categoriser(prs_nat):
    if str(prs_nat).startswith("1"):
        return "Medecins"
    elif str(prs_nat).startswith("3"):
        return "Pharmacie"
    elif str(prs_nat).startswith("4"):
        return "Kinesitherapie"
    elif str(prs_nat).startswith("5"):
        return "Hospitalisation"
    else:
        return "Autre"

df["CATEGORIE"] = df["PRS_NAT_REF"].astype(str).apply(categoriser)

recap_cat = df.groupby("CATEGORIE").agg(
    N_Prestations=("PRS_PAI_MNT","count"),
    Total_Euros=("PRS_PAI_MNT","sum"),
    Moyenne_Euros=("PRS_PAI_MNT","mean")
).round(2).sort_values("Total_Euros", ascending=False)

total_global = df["PRS_PAI_MNT"].sum()
recap_cat["Part_%"] = (recap_cat["Total_Euros"] / total_global * 100).round(1)

print(recap_cat.to_string())
print(f"\n  Total global : {total_global:,.2f} euros")

# ══════════════════════════════════════════
# ETAPE 3 — DEPENSES PAR PATIENT
# ══════════════════════════════════════════
print("\n" + "="*60)
print("ETAPE 3 — Depenses par patient")
print("="*60)

dep_patient = df.groupby("NIR_ANO_17").agg(
    Total_Patient=("PRS_PAI_MNT","sum"),
    N_Actes=("PRS_PAI_MNT","count"),
    Rembourse=("BSE_REM_MNT","sum"),
).round(2)

dep_patient["Reste_Charge"] = (
    dep_patient["Total_Patient"] - dep_patient["Rembourse"]
).round(2)

print(f"  Depense moyenne/patient    : {dep_patient['Total_Patient'].mean():>10,.2f} euros")
print(f"  Depense mediane/patient    : {dep_patient['Total_Patient'].median():>10,.2f} euros")
print(f"  Depense min/patient        : {dep_patient['Total_Patient'].min():>10,.2f} euros")
print(f"  Depense max/patient        : {dep_patient['Total_Patient'].max():>10,.2f} euros")
print(f"  Remboursement moyen        : {dep_patient['Rembourse'].mean():>10,.2f} euros")
print(f"  Reste a charge moyen       : {dep_patient['Reste_Charge'].mean():>10,.2f} euros")

# ══════════════════════════════════════════
# ETAPE 4 — PHARMACIE DETAILLEE
# ══════════════════════════════════════════
print("\n" + "="*60)
print("ETAPE 4 — Pharmacie detaillee (ER_PHA_F)")
print("="*60)

recap_pha = er_pha.groupby(["PHA_NOM","PHA_ATC_C07"]).agg(
    N_Delivrances=("PHA_ACT_QSN","count"),
    Total_Boites=("PHA_ACT_QSN","sum"),
    Prix_Moyen=("PHA_PRI_BRU","mean"),
    Total_Euros=("PHA_PRI_BRU","sum")
).round(2).sort_values("Total_Euros", ascending=False)

print(recap_pha.to_string())

# ══════════════════════════════════════════
# ETAPE 5 — DEPENSES PAR TYPE AVC
# ══════════════════════════════════════════
print("\n" + "="*60)
print("ETAPE 5 — Depenses par type d'AVC")
print("="*60)

dep_diag = df.groupby("DIAGNOSTIC")["PRS_PAI_MNT"].agg(
    N_Patients="count",
    Total="sum",
    Moyenne="mean"
).round(2).sort_values("Moyenne", ascending=False)

print(dep_diag.to_string())

# ══════════════════════════════════════════
# ETAPE 6 — DEPENSES PAR CLASSE D'AGE
# ══════════════════════════════════════════
print("\n" + "="*60)
print("ETAPE 6 — Depenses par classe d'age")
print("="*60)

dep_age = df.groupby("CL_AGE")["PRS_PAI_MNT"].agg(
    N_Actes="count",
    Total="sum",
    Moyenne="mean"
).round(2)

print(dep_age.to_string())

# ══════════════════════════════════════════
# ETAPE 7 — DEPENSES VIVANTS vs DECEDES
# ══════════════════════════════════════════
print("\n" + "="*60)
print("ETAPE 7 — Depenses Vivants vs Decedes a J365")
print("="*60)

dep_survie = df.groupby("SURVIE_J365")["PRS_PAI_MNT"].agg(
    N_Actes="count",
    Total="sum",
    Moyenne="mean"
).round(2)
dep_survie.index = dep_survie.index.map({0:"Decede J365", 1:"Vivant J365"})

print(dep_survie.to_string())

# ══════════════════════════════════════════
# SAUVEGARDE
# ══════════════════════════════════════════
df.to_csv("donnees_snds/depenses_avc_complete.csv", index=False)
dep_patient.to_csv("donnees_snds/depenses_par_patient.csv")

print("\n" + "="*60)
print("RESUME FINAL — DEPENSES POST-AVC")
print("="*60)
print(f"  Patients AVC analyses      : {len(cohorte)}")
print(f"  Patients avec prestations  : {df['NIR_ANO_17'].nunique()}")
print(f"  Prestations totales        : {len(er_prs)}")
print(f"  Depense totale post-AVC    : {total_global:,.2f} euros")
print(f"  Depense moyenne/patient    : {dep_patient['Total_Patient'].mean():,.2f} euros")
print(f"\n  depenses_avc_complete.csv")
print(f"  depenses_par_patient.csv")