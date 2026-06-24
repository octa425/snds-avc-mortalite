# ══════════════════════════════════════════════════════════════
# VISUALISATION DES COURBES DE SURVIE KAPLAN-MEIER
# Projet : Mortalité post-AVC — SNDS PMSI
# Auteur : Octavien YAMESSE
# ══════════════════════════════════════════════════════════════
# Kaplan-Meier = méthode statistique qui estime la probabilité
# de survie à chaque instant T, en tenant compte des patients
# "censurés" (perdus de vue ou encore vivants en fin de suivi)
# ══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CHARGEMENT DE LA COHORTE FINALE
# ──────────────────────────────────────────────
print("Chargement de la cohorte AVC...")
df = pd.read_csv("donnees_snds/cohorte_avc_finale.csv",
                  parse_dates=["EXE_SOI_DTD","EXE_SOI_DTF"])
print(f"  {len(df)} patients AVC charges")

# Vérification des colonnes nécessaires
# DELAI_JOURS = nombre de jours entre l'AVC et le décès (ou fin de suivi)
# EVENT       = 1 si décédé, 0 si censuré (vivant en fin de suivi)
print(f"  Deces observes : {int(df['EVENT'].sum())}")
print(f"  Censures       : {int((df['EVENT']==0).sum())}")

# ──────────────────────────────────────────────
# CONFIGURATION GRAPHIQUE GENERALE
# ──────────────────────────────────────────────
# On définit un style propre et lisible pour tous les graphiques
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2E75B6","#E24B4A","#375623","#BF8F00","#7F77DD","#1D9E75"]

# ══════════════════════════════════════════════════════════════
# GRAPHIQUE 1 — COURBE DE SURVIE GLOBALE
# ══════════════════════════════════════════════════════════════
# Cette courbe montre la probabilité de survie pour l'ensemble
# des patients AVC au fil du temps
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))

kmf = KaplanMeierFitter()
kmf.fit(
    durations       = df["DELAI_JOURS"],   # durée de suivi
    event_observed  = df["EVENT"],          # 1=décédé, 0=censuré
    label           = f"Tous patients AVC (n={len(df)})"
)

# Tracer la courbe avec intervalle de confiance à 95%
kmf.plot_survival_function(
    ax=ax,
    ci_show=True,          # afficher l'intervalle de confiance
    color=COLORS[0],
    linewidth=2
)

# Lignes verticales aux dates clés J30 et J365
ax.axvline(x=30,  color="gray", linestyle="--", alpha=0.7, linewidth=1)
ax.axvline(x=365, color="gray", linestyle="--", alpha=0.7, linewidth=1)

# Annotations des probabilités à J30 et J365
prob_j30  = float(kmf.predict(30))
prob_j365 = float(kmf.predict(365))

ax.annotate(f"J30 : {prob_j30:.1%}",
            xy=(30, prob_j30),
            xytext=(50, prob_j30 + 0.05),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="gray"),
            color="gray")

ax.annotate(f"J365 : {prob_j365:.1%}",
            xy=(365, prob_j365),
            xytext=(300, prob_j365 + 0.08),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="gray"),
            color="gray")

ax.set_title("Courbe de survie globale après AVC\nCohorte SNDS — Guyane 2019",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Délai depuis l'AVC (jours)", fontsize=12)
ax.set_ylabel("Probabilité de survie", fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_xlim(0)

plt.tight_layout()
plt.savefig("donnees_snds/courbe_survie_globale.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ courbe_survie_globale.png")

# ══════════════════════════════════════════════════════════════
# GRAPHIQUE 2 — SURVIE PAR SEXE
# ══════════════════════════════════════════════════════════════
# Compare la survie entre hommes et femmes
# Test log-rank = test statistique qui compare les courbes
# H0 : pas de différence de survie entre les deux groupes
# p < 0.05 = différence significative
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))

groupes_sexe = {
    "Homme" : df[df["COD_SEX"].astype(str) == "1"],
    "Femme" : df[df["COD_SEX"].astype(str) == "2"],
}

for i, (label, groupe) in enumerate(groupes_sexe.items()):
    if len(groupe) == 0:
        continue
    kmf_s = KaplanMeierFitter()
    kmf_s.fit(
        durations      = groupe["DELAI_JOURS"],
        event_observed = groupe["EVENT"],
        label          = f"{label} (n={len(groupe)})"
    )
    kmf_s.plot_survival_function(
        ax=ax, ci_show=True,
        color=COLORS[i], linewidth=2
    )

# Test log-rank hommes vs femmes
mask_h = df["COD_SEX"].astype(str) == "1"
mask_f = df["COD_SEX"].astype(str) == "2"
lr = logrank_test(
    df.loc[mask_h, "DELAI_JOURS"],
    df.loc[mask_f, "DELAI_JOURS"],
    event_observed_A=df.loc[mask_h, "EVENT"],
    event_observed_B=df.loc[mask_f, "EVENT"]
)

# Afficher la p-value sur le graphique
significance = "p < 0.05 ✓ (significatif)" if lr.p_value < 0.05 \
               else "p ≥ 0.05 (non significatif)"
ax.text(0.05, 0.10,
        f"Log-rank test : p = {lr.p_value:.4f}\n{significance}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

ax.axvline(x=30,  color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.axvline(x=365, color="gray", linestyle="--", alpha=0.5, linewidth=1)

ax.set_title("Courbe de survie après AVC par sexe\nCohorte SNDS — Guyane 2019",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Délai depuis l'AVC (jours)", fontsize=12)
ax.set_ylabel("Probabilité de survie", fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_xlim(0)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig("donnees_snds/courbe_survie_sexe.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ courbe_survie_sexe.png")

# ══════════════════════════════════════════════════════════════
# GRAPHIQUE 3 — SURVIE PAR CLASSE D'AGE
# ══════════════════════════════════════════════════════════════
# L'âge est un facteur majeur de mortalité post-AVC
# On compare les 3 classes : 18-64 ans, 65-84 ans, 85 ans et +
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))

classes_age = [
    ("1 — 18-64 ans", "18-64 ans"),
    ("2 — 65-84 ans", "65-84 ans"),
    ("3 — 85 ans et +", "85 ans et +"),
]

for i, (cl_code, cl_label) in enumerate(classes_age):
    groupe = df[df["CL_AGE"] == cl_code]
    if len(groupe) == 0:
        continue
    kmf_a = KaplanMeierFitter()
    kmf_a.fit(
        durations      = groupe["DELAI_JOURS"],
        event_observed = groupe["EVENT"],
        label          = f"{cl_label} (n={len(groupe)})"
    )
    kmf_a.plot_survival_function(
        ax=ax, ci_show=True,
        color=COLORS[i], linewidth=2
    )
    # Probabilité de survie à J365 pour chaque classe
    print(f"  {cl_label:15} → J30: {float(kmf_a.predict(30)):.1%}"
          f"  J365: {float(kmf_a.predict(365)):.1%}"
          f"  (n={len(groupe)})")

ax.axvline(x=30,  color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.axvline(x=365, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.text(32,  0.02, "J30",  fontsize=9, color="gray")
ax.text(367, 0.02, "J365", fontsize=9, color="gray")

ax.set_title("Courbe de survie après AVC par classe d'âge\nCohorte SNDS — Guyane 2019",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Délai depuis l'AVC (jours)", fontsize=12)
ax.set_ylabel("Probabilité de survie", fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_xlim(0)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig("donnees_snds/courbe_survie_age.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ courbe_survie_age.png")

# ══════════════════════════════════════════════════════════════
# GRAPHIQUE 4 — SURVIE PAR TYPE D'AVC
# ══════════════════════════════════════════════════════════════
# Compare la survie selon le diagnostic principal
# I60=hémorragie méningée, I63=infarctus, G46=syndrome vasculaire
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 7))

types_avc = df["DIAGNOSTIC"].dropna().unique()

for i, diag in enumerate(types_avc):
    groupe = df[df["DIAGNOSTIC"] == diag]
    if len(groupe) < 10:   # exclure les groupes trop petits
        continue
    kmf_d = KaplanMeierFitter()
    kmf_d.fit(
        durations      = groupe["DELAI_JOURS"],
        event_observed = groupe["EVENT"],
        label          = f"{diag} (n={len(groupe)})"
    )
    kmf_d.plot_survival_function(
        ax=ax, ci_show=False,   # pas d'IC pour garder lisible
        color=COLORS[i % len(COLORS)],
        linewidth=2
    )

ax.axvline(x=30,  color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.axvline(x=365, color="gray", linestyle="--", alpha=0.5, linewidth=1)

ax.set_title("Courbe de survie après AVC par type de diagnostic\nCohorte SNDS — Guyane 2019",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Délai depuis l'AVC (jours)", fontsize=12)
ax.set_ylabel("Probabilité de survie", fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_xlim(0)
ax.legend(fontsize=9, loc="lower left")

plt.tight_layout()
plt.savefig("donnees_snds/courbe_survie_type_avc.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ courbe_survie_type_avc.png")

# ══════════════════════════════════════════════════════════════
# GRAPHIQUE 5 — TABLEAU DE BORD COMPLET (4 graphiques en 1)
# ══════════════════════════════════════════════════════════════
# Un seul fichier avec les 4 courbes — parfait pour le rapport
# ══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

axes = [fig.add_subplot(gs[0,0]),
        fig.add_subplot(gs[0,1]),
        fig.add_subplot(gs[1,0]),
        fig.add_subplot(gs[1,1])]

# ── Graphique A : Survie globale ──
kmf_g = KaplanMeierFitter()
kmf_g.fit(df["DELAI_JOURS"], df["EVENT"], label="Tous AVC")
kmf_g.plot_survival_function(ax=axes[0], ci_show=True,
                              color=COLORS[0], linewidth=2)
axes[0].axvline(x=30,  color="gray", linestyle="--", alpha=0.5)
axes[0].axvline(x=365, color="gray", linestyle="--", alpha=0.5)
axes[0].set_title("Survie globale", fontweight="bold")
axes[0].set_xlabel("Jours")
axes[0].set_ylabel("P(survie)")
axes[0].set_ylim(0, 1.05)

# ── Graphique B : Survie par sexe ──
for i, (sexe, label) in enumerate([("1","Homme"),("2","Femme")]):
    g = df[df["COD_SEX"].astype(str) == sexe]
    if len(g) > 0:
        kmf_sx = KaplanMeierFitter()
        kmf_sx.fit(g["DELAI_JOURS"], g["EVENT"], label=f"{label} (n={len(g)})")
        kmf_sx.plot_survival_function(ax=axes[1], ci_show=False,
                                       color=COLORS[i], linewidth=2)
axes[1].axvline(x=30,  color="gray", linestyle="--", alpha=0.5)
axes[1].axvline(x=365, color="gray", linestyle="--", alpha=0.5)
axes[1].set_title("Survie par sexe", fontweight="bold")
axes[1].set_xlabel("Jours")
axes[1].set_ylabel("P(survie)")
axes[1].set_ylim(0, 1.05)
axes[1].text(0.05, 0.08,
             f"Log-rank p={lr.p_value:.3f}",
             transform=axes[1].transAxes, fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# ── Graphique C : Survie par classe d'âge ──
for i, (cl_code, cl_label) in enumerate(classes_age):
    g = df[df["CL_AGE"] == cl_code]
    if len(g) > 0:
        kmf_ag = KaplanMeierFitter()
        kmf_ag.fit(g["DELAI_JOURS"], g["EVENT"],
                   label=f"{cl_label} (n={len(g)})")
        kmf_ag.plot_survival_function(ax=axes[2], ci_show=False,
                                       color=COLORS[i], linewidth=2)
axes[2].axvline(x=30,  color="gray", linestyle="--", alpha=0.5)
axes[2].axvline(x=365, color="gray", linestyle="--", alpha=0.5)
axes[2].set_title("Survie par classe d'âge", fontweight="bold")
axes[2].set_xlabel("Jours")
axes[2].set_ylabel("P(survie)")
axes[2].set_ylim(0, 1.05)

# ── Graphique D : OR de la régression logistique ──
# On charge les résultats de la régression
try:
    coefs = pd.read_csv("donnees_snds/regression_logistique_j365.csv")
    coefs = coefs[coefs["Variable"].str.startswith("TOP_")]
    coefs = coefs.sort_values("Odds Ratio", ascending=True)

    colors_bar = ["#E24B4A" if or_ > 1 else "#2E75B6"
                  for or_ in coefs["Odds Ratio"]]

    axes[3].barh(coefs["Variable"].str.replace("TOP_",""),
                 coefs["Odds Ratio"],
                 color=colors_bar, alpha=0.8)
    axes[3].axvline(x=1, color="black", linestyle="--",
                    linewidth=1, alpha=0.7)
    axes[3].set_title("Odds Ratio — Mortalité J365\n(rouge=risque, bleu=protecteur)",
                      fontweight="bold")
    axes[3].set_xlabel("Odds Ratio")
    axes[3].set_ylabel("Comorbidité")
except Exception as e:
    axes[3].text(0.5, 0.5, "Regression non disponible",
                 ha="center", va="center",
                 transform=axes[3].transAxes)

fig.suptitle("Tableau de bord — Mortalité post-AVC\nCohorte SNDS Guyane 2019",
             fontsize=16, fontweight="bold", y=1.01)

plt.savefig("donnees_snds/tableau_bord_avc.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ tableau_bord_avc.png")

print("\n" + "="*60)
print("VISUALISATIONS GENEREES :")
print("="*60)
print("  courbe_survie_globale.png  — Survie globale tous AVC")
print("  courbe_survie_sexe.png     — Survie Homme vs Femme")
print("  courbe_survie_age.png      — Survie par classe d'age")
print("  courbe_survie_type_avc.png — Survie par type AVC")
print("  tableau_bord_avc.png       — Tableau de bord complet")
print("\n  → Ouvrir dans donnees_snds/ pour visualiser")