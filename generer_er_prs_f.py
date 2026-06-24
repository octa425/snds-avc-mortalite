import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

print("Chargement de la cohorte AVC...")
cohorte = pd.read_csv("donnees_snds/cohorte_avc_finale.csv",
                       parse_dates=["EXE_SOI_DTD","EXE_SOI_DTF"])
print(f"  {len(cohorte)} patients AVC charges")

nirs_avc = cohorte["NIR_ANO_17"].tolist()
dates_avc = dict(zip(cohorte["NIR_ANO_17"],
                     pd.to_datetime(cohorte["EXE_SOI_DTD"])))

PRS_NAT = {
    "1111": "Consultation medecin generaliste",
    "1112": "Visite medecin generaliste",
    "1127": "Consultation medecin specialiste",
    "1191": "Acte de biologie",
    "3221": "Medicament rembourse",
    "3222": "Medicament anticoagulant",
    "3223": "Medicament antihypertenseur",
    "3224": "Medicament antiagregeant",
    "4211": "Seance kinesitherapie",
    "5111": "Hospitalisation MCO",
    "5112": "Hospitalisation SSR",
    "3141": "Acte infirmier domicile",
}

SPE_COD = {
    "1111": "01",
    "1112": "01",
    "1127": "13",
    "1191": "00",
}

MONTANTS = {
    "1111": (25, 5),
    "1112": (30, 5),
    "1127": (50, 10),
    "1191": (35, 15),
    "3221": (20, 8),
    "3222": (45, 10),
    "3223": (15, 5),
    "3224": (12, 4),
    "4211": (18, 3),
    "5111": (1500, 500),
    "5112": (800, 200),
    "3141": (22, 4),
}

MEDICAMENTS = {
    "3400933226558": {"nom": "Kardegic (aspirine)", "atc": "B01AC06", "prix": 5.20},
    "3400934744198": {"nom": "Xarelto (rivaroxaban)", "atc": "B01AF01", "prix": 45.80},
    "3400926939939": {"nom": "Plavix (clopidogrel)", "atc": "B01AC04", "prix": 12.50},
    "3400936543210": {"nom": "Amlodipine", "atc": "C08CA01", "prix": 8.30},
    "3400921234567": {"nom": "Atorvastatine", "atc": "C10AA05", "prix": 6.90},
    "3400987654321": {"nom": "Metformine", "atc": "A10BA02", "prix": 4.20},
}

print("\nGeneration ER_PRS_F...")

rows = []
rows_pha = []
flux_id = 0
DATE_FIN = datetime(2019, 12, 31)

for nir in nirs_avc:
    date_avc = dates_avc[nir]
    if pd.isna(date_avc):
        continue
    date_avc = pd.Timestamp(date_avc).to_pydatetime()
    jours_dispo = (DATE_FIN - date_avc).days
    if jours_dispo < 1:
        continue
    n_prestations = random.randint(5, 30)
    for _ in range(n_prestations):
        jours_apres = random.randint(1, jours_dispo)
        date_presta = date_avc + timedelta(days=jours_apres)
        prs_nat = random.choices(
            list(PRS_NAT.keys()),
            weights=[20, 10, 15, 10, 15, 8, 8, 8, 5, 3, 3, 5],
            k=1
        )[0]
        moy, std = MONTANTS[prs_nat]
        montant = max(0, round(np.random.normal(moy, std), 2))
        remboursement = round(montant * random.uniform(0.6, 1.0), 2)
        rows.append({
            "BEN_NIR_PSA" : nir,
            "BEN_RNG_GEM" : "01",
            "PRS_NAT_REF" : prs_nat,
            "PRS_NAT_LIB" : PRS_NAT[prs_nat],
            "EXE_SOI_DTD" : date_presta,
            "EXE_SOI_DTF" : date_presta + timedelta(days=random.randint(0,1)),
            "PRS_PAI_MNT" : montant,
            "BSE_REM_MNT" : remboursement,
            "BEN_RES_DPT" : "973",
            "CPL_MAJ_TOP" : 1,
            "DPN_QLF"     : 0,
            "PSE_SPE_COD" : SPE_COD.get(prs_nat, "00"),
            "DCT_ORD_NUM" : flux_id,
            "FLX_DIS_DTD" : date_presta,
            "FLX_EMT_NUM" : random.randint(1, 999),
            "FLX_EMT_ORD" : random.randint(1, 99),
            "FLX_EMT_TYP" : random.randint(1, 3),
            "FLX_TRT_DTD" : date_presta + timedelta(days=random.randint(1,30)),
            "ORG_CLE_NUM" : "01",
            "PRS_ORD_NUM" : flux_id,
            "REM_TYP_AFF" : "1",
        })
        if prs_nat in ["3221","3222","3223","3224"]:
            cip13 = random.choice(list(MEDICAMENTS.keys()))
            med = MEDICAMENTS[cip13]
            rows_pha.append({
                "BEN_NIR_PSA" : nir,
                "DCT_ORD_NUM" : flux_id,
                "FLX_DIS_DTD" : date_presta,
                "PHA_PRS_C13" : cip13,
                "PHA_PRS_IDE" : cip13[:7],
                "PHA_NOM"     : med["nom"],
                "PHA_ATC_C07" : med["atc"],
                "PHA_ACT_QSN" : random.randint(1, 3),
                "PHA_PRI_BRU" : med["prix"],
            })
        flux_id += 1

er_prs_f = pd.DataFrame(rows)
er_prs_f = er_prs_f[er_prs_f["DPN_QLF"] != 71]
er_prs_f = er_prs_f[er_prs_f["CPL_MAJ_TOP"] < 2]

er_pha_f = pd.DataFrame(rows_pha)

print(f"  ER_PRS_F : {len(er_prs_f)} prestations")
print(f"  ER_PHA_F : {len(er_pha_f)} delivrances medicaments")
print(f"  Patients couverts : {er_prs_f['BEN_NIR_PSA'].nunique()}")

er_prs_f.to_csv("donnees_snds/ER_PRS_F.csv", index=False)
er_pha_f.to_csv("donnees_snds/ER_PHA_F.csv", index=False)

print(f"\n✅ TABLES GENEREES :")
print(f"  ER_PRS_F : {len(er_prs_f):>6} lignes → ER_PRS_F.csv")
print(f"  ER_PHA_F : {len(er_pha_f):>6} lignes → ER_PHA_F.csv")

print("\nRepartition des depenses par type :")
recap = er_prs_f.groupby("PRS_NAT_LIB").agg(
    N=("PRS_PAI_MNT","count"),
    Total=("PRS_PAI_MNT","sum"),
    Moyenne=("PRS_PAI_MNT","mean")
).round(2).sort_values("Total", ascending=False)
print(recap.to_string())

total = er_prs_f["PRS_PAI_MNT"].sum()
moy_pat = er_prs_f.groupby("BEN_NIR_PSA")["PRS_PAI_MNT"].sum().mean()
print(f"\nDepense totale post-AVC  : {total:,.2f} euros")
print(f"Depense moyenne/patient  : {moy_pat:,.2f} euros")
