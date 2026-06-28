import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

print("=" * 60)
print("GENERATION DES DONNEES SIMULEES — DMS CCH AP-HP")
print("=" * 60)

# ══════════════════════════════════════════
# 1 — FICHIER EXHAUSTIVITE (UMA)
# ══════════════════════════════════════════
print("\nGeneration Exhaustivite_hupc.xlsx...")

uma_data = [
    # DMU 503 — Chirurgie
    (503, "CCH", "HC",  "103 REA CHIR OLLIER"),
    (503, "CCH", "HC",  "104 USC OLLIER"),
    (503, "CCH", "HDJ", "260J HDJ BILAN PRE OP"),
    (503, "CCH", "HDJ", "500C HDJ UCA RADIOFREQ"),
    (503, "CCH", "HDJ", "501J Douleur periop HDJ"),

    # DMU 504 — Urgences
    (504, "HTD", "HC",  "010 Urgences medico-chirurgicales UHTCD"),
    (504, "HTD", "HC",  "011 Unite MPA post UHCD"),
    (504, "HTD", "HC",  "013 CUSCO Urgences medico-judiciaires"),
    (504, "CCH", "HC",  "340 Medecine Intensive Reanimation"),
    (504, "CCH", "HC",  "341 SC Medicale"),
    (504, "CCH", "HC",  "540 Accueil Urgences"),
    (504, "CCH", "HC",  "543 Unite Med Patho Aigu Urgences"),
    (504, "CCH", "HDJ", "384J Addictologie HDJ"),

    # DMU 505 — Pneumologie
    (505, "CCH", "HC",  "037 Chirurgie thoraco-pulmonaire Cochin"),
    (505, "CCH", "HC",  "037S Chirurgie thoraco-pulmonaire Cochin HCS"),
    (505, "CCH", "HC",  "060 Pneumologie"),
    (505, "CCH", "HC",  "060S Pneumologie HCS"),
    (505, "HTD", "HC",  "150 Centre du sommeil"),
    (505, "CCH", "HC",  "393 SC Pneumologie"),
    (505, "CCH", "HDJ", "037J Chirurgie thoraco-pulmonaire Cochin HDJ"),
    (505, "CCH", "HDJ", "060J Pneumologie HDJ"),
    (505, "CCH", "HDJ", "061J Pneumologie diagnostic HDJ"),
    (505, "CCH", "HDJ", "062C Pneumologie interventionnelle HDJ"),
    (505, "CCH", "HDJ", "062J HDJ ONCOTHORAX"),
    (505, "CCH", "HDJ", "063J Suivi post-greffe HDJ"),
    (505, "HTD", "HDJ", "150J Centre du sommeil nuit HDJ"),
    (505, "HTD", "HDJ", "151J Centre du sommeil jour HDJ"),
    (505, "HTD", "HDJ", "152J HDJ PATHOLOGIES PROFESSIONNELLES ENV"),
    (505, "HTD", "HDJ", "230J Rehabilitation par le sport HDJ"),

    # DMU 506 — Geriatrie
    (506, "BRC", "HC",  "018 Court-sejour geriatrique G1"),
    (506, "BRC", "HC",  "029 Court-sejour geriatrique G2"),
    (506, "BRC", "HDJ", "029J Court-sejour geriatrique G2 HDJ"),

    # DMU 507 — Radiotherapie
    (507, "CCH", "HC",  "022S Iratherapie HCS"),
    (507, "CCH", "HDJ", "022J Radiotherapie Interne Vectorisee HDJ"),
    (507, "CCH", "HDJ", "150C HDJ UCA POSE DE PAC"),
    (507, "CCH", "HDJ", "150J Radio A HDJ"),
    (507, "CCH", "HDJ", "320J Radio B HDJ"),

    # DMU 513 — Maison des Adolescents
    (513, "CCH", "HC",  "600 Medecine interne Maison des Adolescents"),
    (513, "CCH", "HDJ", "600J Medecine interne ambulatoire Maison Adolescents HDJ"),

    # DMU 514 — Medecine interne
    (514, "CCH", "HC",  "010 Medecine interne"),
    (514, "CCH", "HC",  "010S Medecine interne HCS"),
    (514, "CCH", "HDJ", "010J Medecine interne HDJ"),
    (514, "HTD", "HDJ", "221J HTA et PCV HDJ"),
    (514, "HTD", "HDJ", "222J Medecine specialisee HDJ"),
    (514, "HTD", "HDJ", "227J HDJ ESJ"),
    (514, "HTD", "HDJ", "500J HDJ CASPER"),

    # DMU 515 — Oncologie / Gastro / Hemato / Urologie
    (515, "CCH", "HC",  "011 Oncologie"),
    (515, "CCH", "HC",  "040 Gastro-enterologie"),
    (515, "CCH", "HC",  "040S Gastro-enterologie HCS"),
    (515, "CCH", "HC",  "040X HDJ CHIMIOS GASTRO"),
    (515, "CCH", "HC",  "041 Chirurgie digestive et endocrinienne"),
    (515, "CCH", "HC",  "270 Urologie"),
    (515, "CCH", "HC",  "270S Urologie HCS"),
    (515, "CCH", "HC",  "451 SI Hematologie"),
    (515, "CCH", "HC",  "670 Hepatologie medicale"),
    (515, "CCH", "HC",  "670 Hematologie medicale"),
    (515, "CCH", "HC",  "086 Biotherapie medicale"),
    (515, "CCH", "HDJ", "011J Oncologie HDJ"),
    (515, "CCH", "HDJ", "040J Gastro-enterologie HDJ"),
    (515, "CCH", "HDJ", "040J Biotherapie HDJ"),
    (515, "CCH", "HDJ", "271C UCA Urologie HDJ"),
    (515, "CCH", "HDJ", "410C UCA Chirurgie digestive HDJ"),
    (515, "CCH", "HDJ", "450J Hematologie HDJ"),
    (515, "CCH", "HDJ", "670J Hepatologie medicale HDJ"),

    # DMU 516 — Gynecologie / Obstetrique / Neonatologie
    (516, "CCH", "HC",  "091 Reanination neonatale Port-Royal"),
    (516, "CCH", "HC",  "092 SI Neonatologie Port-Royal"),
    (516, "CCH", "HC",  "130 Obstetrique Port-Royal Suite de couche"),
    (516, "CCH", "HC",  "140 Gynecologie chirurgicale"),
    (516, "CCH", "HC",  "140S Gynecologie chirurgicale HCS"),
    (516, "CCH", "HC",  "1405 Gynecologie chirurgicale HCS"),
    (516, "CCH", "HC",  "591 Ophtalmologie"),
    (516, "CCH", "HDJ", "090J Neonatologie Port-Royal HDJ"),
    (516, "CCH", "HDJ", "130J Obstetrique Port-Royal HDJ NN vulnerables"),
    (516, "CCH", "HDJ", "1301 Obstetrique Port-Royal HDJ"),
    (516, "CCH", "HDJ", "140C UCA Gynecologie chirurgicale HDJ"),
    (516, "CCH", "HDJ", "141J IVG B HDJ"),
    (516, "CCH", "HDJ", "142J IVG B HDJ"),
    (516, "CCH", "HDJ", "143J Gynecologie medicale HDJ"),
    (516, "CCH", "HDJ", "146J HDJ AMP PF"),
    (516, "CCH", "HDJ", "146J UCA Gynecologie chirurgicale HDJ"),
    (516, "CCH", "HDJ", "561C UCA chir plastie et rec"),
    (516, "CCH", "HDJ", "591C Ophtalmologie HDJ"),

    # DMU 517 — Dermatologie / Diabetologie / Endocrinologie
    (517, "CCH", "HC",  "160 Dermatologie"),
    (517, "CCH", "HC",  "070 Endocrinologie"),
    (517, "CCH", "HC",  "071 Diabetologie Cochin"),
    (517, "CCH", "HC",  "071S Diabetologie Cochin HCS"),
    (517, "CCH", "HC",  "0715 Diabetologie Cochin HCS"),
    (517, "CCH", "HDJ", "070J Endocrinologie HDJ"),
    (517, "CCH", "HDJ", "071J Diabetologie HDJ"),
    (517, "CCH", "HDJ", "072J Gynecologie Endocrinologie HDJ"),
    (517, "CCH", "HDJ", "073J Diabetologie Pied Cochin HDJ"),
    (517, "CCH", "HDJ", "161J Dermatologie medicale Tamier HDJ"),
    (517, "CCH", "HDJ", "162C UCA Dermatologie HDJ"),
]

uma_df = pd.DataFrame(uma_data, columns=["DMU","Site","Type","UMA"])
uma_df.to_excel("donnees_dms/Exhaustivite_hupc.xlsx", index=False)
print(f"  Exhaustivite_hupc.xlsx : {len(uma_df)} UMA")

# ══════════════════════════════════════════
# 2 — FICHIER CCH_2024.CSV (Sejours MCO)
# ══════════════════════════════════════════
print("\nGeneration CCH_2024.csv...")

# Services HC et HCS uniquement (pas HDJ)
services_hc = [row for row in uma_data
               if row[2] in ("HC",) and row[1] == "CCH"
               and not any(c in row[3] for c in ["HDJ","HCS"])]

# DMS reelles par specialite (en jours)
DMS_PAR_SERVICE = {
    "011 Oncologie"                          : (8,  3),
    "040 Gastro-enterologie"                 : (5,  2),
    "041 Chirurgie digestive et endocrinienne": (7,  3),
    "270 Urologie"                           : (4,  2),
    "451 SI Hematologie"                     : (12, 4),
    "670 Hepatologie medicale"               : (6,  2),
    "670 Hematologie medicale"               : (10, 3),
    "086 Biotherapie medicale"               : (3,  1),
    "037 Chirurgie thoraco-pulmonaire Cochin": (6,  3),
    "060 Pneumologie"                        : (7,  3),
    "150 Centre du sommeil"                  : (2,  1),
    "340 Medecine Intensive Reanimation"     : (5,  3),
    "341 SC Medicale"                        : (3,  1),
    "540 Accueil Urgences"                   : (1,  0),
    "543 Unite Med Patho Aigu Urgences"      : (2,  1),
    "018 Court-sejour geriatrique G1"        : (8,  3),
    "029 Court-sejour geriatrique G2"        : (10, 4),
    "010 Medecine interne"                   : (6,  3),
    "600 Medecine interne Maison des Adolescents": (5, 2),
    "160 Dermatologie"                       : (4,  2),
    "070 Endocrinologie"                     : (5,  2),
    "071 Diabetologie Cochin"                : (4,  2),
    "140 Gynecologie chirurgicale"           : (3,  1),
    "591 Ophtalmologie"                      : (1,  1),
    "091 Reanination neonatale Port-Royal"   : (15, 5),
    "092 SI Neonatologie Port-Royal"         : (10, 4),
    "130 Obstetrique Port-Royal Suite de couche": (3, 1),
    "103 REA CHIR OLLIER"                    : (7,  3),
    "104 USC OLLIER"                         : (3,  1),
}

# Nb sejours par service
NB_SEJOURS = {
    "011 Oncologie"                          : 236,
    "040 Gastro-enterologie"                 : 83,
    "041 Chirurgie digestive et endocrinienne": 424,
    "270 Urologie"                           : 296,
    "451 SI Hematologie"                     : 107,
    "670 Hepatologie medicale"               : 86,
    "670 Hematologie medicale"               : 48,
    "086 Biotherapie medicale"               : 1360,
    "037 Chirurgie thoraco-pulmonaire Cochin": 259,
    "060 Pneumologie"                        : 296,
    "150 Centre du sommeil"                  : 162,
    "340 Medecine Intensive Reanimation"     : 185,
    "341 SC Medicale"                        : 20,
    "540 Accueil Urgences"                   : 1459,
    "543 Unite Med Patho Aigu Urgences"      : 828,
    "018 Court-sejour geriatrique G1"        : 95,
    "029 Court-sejour geriatrique G2"        : 128,
    "010 Medecine interne"                   : 236,
    "600 Medecine interne Maison des Adolescents": 33,
    "160 Dermatologie"                       : 143,
    "070 Endocrinologie"                     : 85,
    "071 Diabetologie Cochin"                : 115,
    "140 Gynecologie chirurgicale"           : 346,
    "591 Ophtalmologie"                      : 151,
    "091 Reanination neonatale Port-Royal"   : 85,
    "092 SI Neonatologie Port-Royal"         : 93,
    "130 Obstetrique Port-Royal Suite de couche": 2041,
    "103 REA CHIR OLLIER"                    : 36,
    "104 USC OLLIER"                         : 19,
}

def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

DATE_DEB = datetime(2024, 1, 1)
DATE_FIN = datetime(2024, 12, 31)

rows = []
nip_counter = 1000000
nda_counter = 2024000000

for uma_nom, nb_sejours in NB_SEJOURS.items():
    # Extraire code URMP
    urmp = uma_nom.split(" ")[0]
    dms_moy, dms_std = DMS_PAR_SERVICE.get(uma_nom, (5, 2))

    for _ in range(nb_sejours):
        nip = str(nip_counter + random.randint(0, nb_sejours * 3))
        nda = str(nda_counter)
        nda_counter += 1

        # Duree sejour
        duree = max(1, int(np.random.normal(dms_moy, dms_std)))

        # Dates
        date_entree = random_date(DATE_DEB, DATE_FIN - timedelta(days=duree))
        date_sortie = date_entree + timedelta(days=duree)

        # NAS = NDA + date_entree (identifiant HDJ)
        nas = f"{nda}_{date_entree.strftime('%d%m%Y')}"

        rows.append({
            "NIP"                  : nip,
            "NDA"                  : nda,
            "NAS"                  : nas,
            "URMP"                 : urmp,
            "Date entree resume"   : date_entree.strftime("%d/%m/%Y %H:%M"),
            "Date sortie resume"   : date_sortie.strftime("%d/%m/%Y %H:%M"),
            "Duree sejour"         : duree,
            "Duree Resume"         : duree,
        })

    # Ajouter version HCS si elle existe
    hcs_services = [r for r in uma_data
                    if r[2] == "HC" and r[1] == "CCH"
                    and r[3].endswith("HCS")
                    and r[3].split(" ")[0] == urmp + "S"]

    if hcs_services:
        urmp_s = urmp + "S"
        nb_hcs = max(5, nb_sejours // 4)
        for _ in range(nb_hcs):
            nip = str(nip_counter + random.randint(0, nb_sejours * 3))
            nda = str(nda_counter)
            nda_counter += 1
            duree = max(2, int(np.random.normal(dms_moy - 1, dms_std)))
            date_entree = random_date(DATE_DEB, DATE_FIN - timedelta(days=duree))
            date_sortie = date_entree + timedelta(days=duree)
            nas = f"{nda}_{date_entree.strftime('%d%m%Y')}"
            rows.append({
                "NIP"                : nip,
                "NDA"                : nda,
                "NAS"                : nas,
                "URMP"               : urmp_s,
                "Date entree resume" : date_entree.strftime("%d/%m/%Y %H:%M"),
                "Date sortie resume" : date_sortie.strftime("%d/%m/%Y %H:%M"),
                "Duree sejour"       : duree,
                "Duree Resume"       : duree,
            })

df_cch = pd.DataFrame(rows)

# Ajouter quelques HDJ pour tester l'exclusion
hdj_rows = []
for i in range(500):
    nda = str(nda_counter)
    nda_counter += 1
    urmp_hdj = random.choice(["011J","040J","060J","140C","010J"])
    date_entree = random_date(DATE_DEB, DATE_FIN)
    nas = f"{nda}_{date_entree.strftime('%d%m%Y')}"
    hdj_rows.append({
        "NIP"                : str(random.randint(1000000, 9999999)),
        "NDA"                : nda,
        "NAS"                : nas,
        "URMP"               : urmp_hdj,
        "Date entree resume" : date_entree.strftime("%d/%m/%Y %H:%M"),
        "Date sortie resume" : date_entree.strftime("%d/%m/%Y %H:%M"),
        "Duree sejour"       : 0,
        "Duree Resume"       : 0,
    })

df_cch = pd.concat([df_cch, pd.DataFrame(hdj_rows)], ignore_index=True)
df_cch = df_cch.sample(frac=1, random_state=42).reset_index(drop=True)

df_cch.to_csv("donnees_dms/CCH_2024.csv", sep=";",
              encoding="windows-1252", index=False)

print(f"  CCH_2024.csv : {len(df_cch)} lignes")
print(f"  Sejours HC   : {len(df_cch[~df_cch['URMP'].str.endswith('J')])}")
print(f"  Sejours HDJ  : {len(df_cch[df_cch['URMP'].str.endswith('J')])}")

print(f"\n  Apercu par service :")
recap = df_cch[~df_cch['URMP'].str.endswith('J')].groupby("URMP").agg(
    N=("NDA","count"),
    DMS_moy=("Duree sejour","mean")
).round(1).sort_values("N", ascending=False).head(10)
print(recap.to_string())

print("\n✅ TABLES GENEREES :")
print("  donnees_dms/CCH_2024.csv")
print("  donnees_dms/Exhaustivite_hupc.xlsx")
EOF