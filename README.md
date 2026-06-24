# SNDS — Analyse de la mortalité post-AVC

Analyse épidémiologique de la mortalité à J30 et J365
après Accident Vasculaire Cérébral (AVC) à partir des
données SNDS (PMSI-MCO, DCIR, CépiDc).

## ⚠️ Avertissement
Ce projet est une reproduction en Python d'une étude réalisée en SAS à l'Observatoire Régional de Santé de Guyane (ORS).
Les données utilisées sont **entièrement simulées** (400 patients).
Aucune donnée patient réelle n'a été utilisée à aucune étape.
Les résultats présentés (survie, mortalité, dépenses) sont obtenus sur ce jeu simulé.
Ils ne reflètent pas les données réelles de la Guyane.

## Contexte
- **Étude** : mortalité post-AVC à J30 et J365
- **Source** : ORS Guyane (reproduction Python)
- **Données** : SNDS (PMSI-MCO, DCIR, CépiDc)
- **Statut** : données simulées, aucune donnée réelle

## Tables SNDS utilisées
| Table | Source | Contenu |
|-------|--------|---------|
| T_MCO_B | PMSI | Résumés de sortie MCO |
| T_MCO_C | PMSI | Table de chaînage NIR |
| T_MCO_D | PMSI | Diagnostics associés |
| T_MCO_GV | PMSI | Valorisation GHM |
| IR_BEN_R | SNDS | Référentiel des bénéficiaires |
| ER_PRS_F | DCIR | Prestations de soins de ville |
| ER_PHA_F | DCIR | Pharmacie |

## Cohorte
- **400 patients AVC** identifiés (I60-I64 + G46)
- **Période** : 2019
- **Territoire** : Guyane (BDI_DEP = 9C)
- **Population** : Adultes ≥ 18 ans
- **Exclusions** : APHP, APHM, HCL (doublons inter-régionaux)
  — liste des FINESS explicite dans le code

## Indicateurs de survie
- **J30** : qualité de la phase aiguë (prise en charge initiale)
- **J365** : efficacité du parcours de soins (rééducation, suivi)

## Résultats (sur données simulées)
| Indicateur | Résultat |
|---|---|
| Survie à J30 | 86.5% |
| Survie à J365 | 76.8% |
| Mortalité à J30 | 13.5% |
| Mortalité à J365 | 23.2% |
| Dépense moyenne / patient | 1 571 € |

## Pipeline
Le pipeline est orchestré en Python avec **Pandas**, **Lifelines** et **Scikit-learn** :
1. Lecture des tables SNDS (PMSI, DCIR, CépiDc)
2. Filtres qualité (NIR_RET = 0, NIR_ANO = 0, âge ≥ 18)
3. Sélection de la cohorte AVC (I60-I64 + G46)
4. Enrichissement avec IR_BEN_R (âge, sexe, décès)
5. Calcul de la survie à J30 et J365
6. Courbes de survie (Kaplan-Meier) par sexe et âge
7. Identification des comorbidités (HTA, diabète, BPCO, etc.)
8. Régression logistique (facteurs associés à la mortalité à J365)

## Avertissement méthodologique
La définition des comorbidités repose sur les diagnostics associés (ASS_DGN) du PMSI,
avec une recherche sur les 3 premiers caractères.
Les données étant simulées, les résultats ne sont pas interprétables en tant que données réelles.

## Statut
Projet terminé: code reproductible, résultats documentés.

## Auteur
Octavien YAMESSE:  github.com/octa425
