import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# ══════════════════════════════════════════
# PARAMETRES
# ══════════════════════════════════════════
N_SEJOURS = 5000        # Total séjours MCO Guyane
N_AVC     = 450         # Dont séjours AVC
DATE_DEB  = datetime(2019, 1, 1)
DATE_FIN  = datetime(2019, 12, 31)

# Codes GHM valides (hors erreurs CMD90, hors séances CMD28)
GHM_AVC   = ["01C021", "01C022", "01C023", "01M101", "01M102",
              "01M111", "01M112", "01M131", "01M132", "01M133"]
GHM_AUTRES = ["05M092", "05M091", "04M051", "06M071", "07M091",
               "08M071", "10M051", "11M091", "12M051", "14M051"]

# Codes diagnostics AVC
CODES_AVC = {
    "I60": "Hémorragie méningée",
    "I61": "Hémorragie intracérébrale",
    "I62": "Autres hémorragies",
    "I63": "Infarctus cérébral",
    "I64": "AVC non précisé",
    "G46": "Syndrome vasculaire cérébral"
}

# Comorbidités fréquentes
COMORBIDITES = ["I10", "E11", "I48", "I25", "E78",
                "J44", "N18", "I50", "E14", "I20"]

# Codes communes Guyane (BDI_COD)
COMMUNES_973 = ["9C010", "9C020", "9C050", "9C051",
                "9C054", "9C070", "9CC01", "9CC02"]

# FINESS établissements Guyane (pas APHP/APHM/HCL)
FINESS_973 = ["973000032", "973000040", "973000057",
              "973000065", "973000073", "973000081"]

def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

def random_nir():
    return ''.join([str(random.randint(0, 9)) for _ in range(15)]) + "A"

# ══════════════════════════════════════════
# T_MCO_B — Résumés de sortie
# ══════════════════════════════════════════
print("Génération T_MCO_B...")

eta_nums = [random.choice(FINESS_973) for _ in range(N_SEJOURS)]
rsa_nums = [f"RSA{str(i).zfill(7)}" for i in range(N_SEJOURS)]

# Diagnostics principaux — 450 AVC + 4550 autres
dgn_pal_list = []
for i in range(N_SEJOURS):
    if i < N_AVC:
        code = random.choice(list(CODES_AVC.keys()))
        # Sous-codes précis
        if code == "I60": dgn_pal_list.append(f"I60{random.randint(0,9)}")
        elif code == "I61": dgn_pal_list.append(f"I61{random.randint(0,9)}")
        elif code == "I62": dgn_pal_list.append(f"I620")
        elif code == "I63": dgn_pal_list.append(f"I63{random.randint(0,9)}")
        elif code == "I64": dgn_pal_list.append("I64")
        else: dgn_pal_list.append(f"G46{random.randint(0,8)}")
    else:
        dgn_pal_list.append(f"Z{random.randint(10,99)}{random.randint(0,9)}")

ages = []
for i in range(N_SEJOURS):
    if i < N_AVC:
        # AVC : distribution réaliste 18-95 ans
        ages.append(int(np.random.normal(68, 15)))
    else:
        ages.append(random.randint(18, 95))
ages = [max(18, min(95, a)) for a in ages]

dates_entree = [random_date(DATE_DEB, DATE_FIN) for _ in range(N_SEJOURS)]
durees = [random.randint(1, 30) for _ in range(N_SEJOURS)]
dates_sortie = [dates_entree[i] + timedelta(days=durees[i])
                for i in range(N_SEJOURS)]

t_mco_b = pd.DataFrame({
    "ETA_NUM"  : eta_nums,
    "RSA_NUM"  : rsa_nums,
    "DGN_PAL"  : dgn_pal_list,
    "DGN_REL"  : ["" for _ in range(N_SEJOURS)],
    "AGE_ANN"  : ages,
    "COD_SEX"  : [random.choice(["1", "2"]) for _ in range(N_SEJOURS)],
    "BDI_DEP"  : ["9C"] * N_SEJOURS,
    "BDI_COD"  : [random.choice(COMMUNES_973) for _ in range(N_SEJOURS)],
    "ENT_MOD"  : [random.choice(["6", "7", "8"]) for _ in range(N_SEJOURS)],
    "ENT_PRV"  : [random.choice(["", "MCO", "SSR"]) for _ in range(N_SEJOURS)],
    "SOR_MOD"  : [random.choice(["6", "7", "8", "9"]) for _ in range(N_SEJOURS)],
    "SOR_DES"  : [random.choice(["", "1", "2", "6"]) for _ in range(N_SEJOURS)],
    "SEQ_RUM"  : ["01"] * N_SEJOURS,
    "SEJ_TYP"  : [random.choice(["", "A", "C"]) for _ in range(N_SEJOURS)],
})

print(f"  T_MCO_B : {len(t_mco_b)} séjours dont {N_AVC} AVC")

# ══════════════════════════════════════════
# T_MCO_C — Table de chaînage
# ══════════════════════════════════════════
print("Génération T_MCO_C...")

nirs = [random_nir() for _ in range(N_SEJOURS)]

t_mco_c = pd.DataFrame({
    "ETA_NUM"     : eta_nums,
    "RSA_NUM"     : rsa_nums,
    "NIR_ANO_17"  : nirs,
    "EXE_SOI_DTD" : dates_entree,
    "EXE_SOI_DTF" : dates_sortie,
    "SEJ_NUM"     : [f"SEJ{str(i).zfill(7)}" for i in range(N_SEJOURS)],
    "NIR_RET"     : ["0"] * N_SEJOURS,
    "NAI_RET"     : ["0"] * N_SEJOURS,
    "SEX_RET"     : ["0"] * N_SEJOURS,
    "DAT_RET"     : ["0"] * N_SEJOURS,
    "SEJ_RET"     : ["0"] * N_SEJOURS,
    "FHO_RET"     : ["0"] * N_SEJOURS,
    "PMS_RET"     : ["0"] * N_SEJOURS,
})

print(f"  T_MCO_C : {len(t_mco_c)} lignes")

# ══════════════════════════════════════════
# T_MCO_D — Diagnostics associés (comorbidités)
# ══════════════════════════════════════════
print("Génération T_MCO_D...")

rows_d = []
for i in range(N_SEJOURS):
    n_comorbidites = random.randint(0, 4)
    comorbidites_patient = random.sample(COMORBIDITES, n_comorbidites)
    for c in comorbidites_patient:
        rows_d.append({
            "ETA_NUM" : eta_nums[i],
            "RSA_NUM" : rsa_nums[i],
            "ASS_DGN" : c,
        })

t_mco_d = pd.DataFrame(rows_d)
print(f"  T_MCO_D : {len(t_mco_d)} diagnostics associés")

# ══════════════════════════════════════════
# T_MCO_GV — Valorisation GHM
# ══════════════════════════════════════════
print("Génération T_MCO_GV...")

ghm_list = []
for i in range(N_SEJOURS):
    if i < N_AVC:
        ghm_list.append(random.choice(GHM_AVC))
    else:
        ghm_list.append(random.choice(GHM_AUTRES))

t_mco_gv = pd.DataFrame({
    "ETA_NUM" : eta_nums,
    "RSA_NUM" : rsa_nums,
    "GHM_NUM" : ghm_list,
})

print(f"  T_MCO_GV : {len(t_mco_gv)} lignes")

# ══════════════════════════════════════════
# IR_BEN_R — Référentiel bénéficiaires + décès
# ══════════════════════════════════════════
print("Génération IR_BEN_R...")

# Patients uniques
patients_nirs = list(set(nirs))
N_PATIENTS = len(patients_nirs)

# Mortalité réaliste : ~15% à J30, ~25% à J365
# On crée d'abord la date de décès pour les patients AVC
nir_avc = nirs[:N_AVC]
nir_autres = nirs[N_AVC:]

dcd_dates = []
for nir in patients_nirs:
    if nir in nir_avc:
        idx = nir_avc.index(nir)
        date_avc = dates_entree[idx]
        rand = random.random()
        if rand < 0.12:
            # Décès à J30
            dcd_dates.append(date_avc + timedelta(days=random.randint(1, 30)))
        elif rand < 0.22:
            # Décès entre J31 et J365
            dcd_dates.append(date_avc + timedelta(days=random.randint(31, 365)))
        else:
            # Vivant
            dcd_dates.append(None)
    else:
        # Non AVC : très peu de décès
        if random.random() < 0.02:
            dcd_dates.append(random_date(DATE_DEB, DATE_FIN))
        else:
            dcd_dates.append(None)

ir_ben_r = pd.DataFrame({
    "NIR_ANO_17"  : patients_nirs,
    "BEN_NIR_PSA" : patients_nirs,
    "BEN_RNG_GEM" : ["01"] * N_PATIENTS,
    "BEN_DCD_DTE" : dcd_dates,
    "BEN_NAI_MOI" : [str(random.randint(1, 12)).zfill(2)
                     for _ in range(N_PATIENTS)],
    "BEN_SEX_COD" : [random.choice(["1", "2"])
                     for _ in range(N_PATIENTS)],
    "BEN_RES_DPT" : ["973"] * N_PATIENTS,
    "BEN_DTE_MAJ" : [datetime(2019, 12, 31)] * N_PATIENTS,
})

n_deces = ir_ben_r["BEN_DCD_DTE"].notna().sum()
print(f"  IR_BEN_R : {N_PATIENTS} patients dont {n_deces} décès")

# ══════════════════════════════════════════
# SAUVEGARDE
# ══════════════════════════════════════════
print("\nSauvegarde des tables...")

os.makedirs("donnees_snds", exist_ok=True)

t_mco_b.to_csv("donnees_snds/T_MCO_B.csv", index=False)
t_mco_c.to_csv("donnees_snds/T_MCO_C.csv", index=False)
t_mco_d.to_csv("donnees_snds/T_MCO_D.csv", index=False)
t_mco_gv.to_csv("donnees_snds/T_MCO_GV.csv", index=False)
ir_ben_r.to_csv("donnees_snds/IR_BEN_R.csv", index=False)

print("\n✅ TABLES GÉNÉRÉES :")
print(f"  T_MCO_B  : {len(t_mco_b):>6} lignes → donnees_snds/T_MCO_B.csv")
print(f"  T_MCO_C  : {len(t_mco_c):>6} lignes → donnees_snds/T_MCO_C.csv")
print(f"  T_MCO_D  : {len(t_mco_d):>6} lignes → donnees_snds/T_MCO_D.csv")
print(f"  T_MCO_GV : {len(t_mco_gv):>6} lignes → donnees_snds/T_MCO_GV.csv")
print(f"  IR_BEN_R : {len(ir_ben_r):>6} lignes → donnees_snds/IR_BEN_R.csv")
print(f"\n  Séjours AVC simulés     : {N_AVC}")
print(f"  Patients uniques        : {N_PATIENTS}")
print(f"  Décès simulés           : {n_deces}")