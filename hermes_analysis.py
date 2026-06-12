"""
Hermès Sales Performance Analysis 2022–2024
Data Analyst: Fatima Aagour
Données entièrement simulées à des fins analytiques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────
# 1. GÉNÉRATION DES DONNÉES SIMULÉES
# ─────────────────────────────────────────

regions = {
    "Chine":        0.222,
    "Europe":       0.200,
    "Japon":        0.155,
    "Amériques":    0.145,
    "Asie-Pacifique": 0.138,
    "Moyen-Orient": 0.080,
    "Autres":       0.060,
}

categories = {
    "Maroquinerie":     0.42,
    "Soie & Textiles":  0.18,
    "Prêt-à-Porter":    0.15,
    "Parfums & Beauté": 0.10,
    "Horlogerie":       0.09,
    "Joaillerie":       0.06,
}

products = [
    "Birkin 35",
    "Kelly 28",
    "Constance 24",
    "Picotin Lock 18",
    "Garden Party 36",
    "Evelyne III 29",
    "Herbag Zip 31",
    "Bolide 27",
]

seasons = {"Printemps": 0.23, "Été": 0.22, "Automne": 0.28, "Hiver": 0.27}

TARGET_CA = 85_000_000
N_CLIENTS = 2982
N_TRANSACTIONS = 15_000

dates = pd.date_range("2022-01-01", "2024-12-31", periods=N_TRANSACTIONS)
dates = pd.to_datetime(np.sort(np.random.choice(dates, N_TRANSACTIONS, replace=False)))

region_list   = np.random.choice(list(regions.keys()),   N_TRANSACTIONS, p=list(regions.values()))
category_list = np.random.choice(list(categories.keys()), N_TRANSACTIONS, p=list(categories.values()))
product_list  = np.random.choice(products, N_TRANSACTIONS)
client_ids    = np.random.randint(1, N_CLIENTS + 1, N_TRANSACTIONS)

base_prices = {
    "Birkin 35": 11000, "Kelly 28": 9500, "Constance 24": 8000,
    "Picotin Lock 18": 3200, "Garden Party 36": 2800,
    "Evelyne III 29": 2400, "Herbag Zip 31": 3000, "Bolide 27": 6500,
}
amounts = np.array([base_prices[p] * np.random.uniform(0.85, 1.15) for p in product_list])
scale_factor = TARGET_CA / amounts.sum()
amounts = amounts * scale_factor

df = pd.DataFrame({
    "date": dates,
    "client_id": client_ids,
    "region": region_list,
    "category": category_list,
    "product": product_list,
    "amount": amounts,
    "year": dates.year,
    "month": dates.month,
    "month_label": dates.strftime("%Y-%m"),
})

def get_season(month):
    if month in [3,4,5]:   return "Printemps"
    elif month in [6,7,8]: return "Été"
    elif month in [9,10,11]: return "Automne"
    else: return "Hiver"

df["season"] = df["month"].apply(get_season)

# Save data
df.to_csv("/home/claude/hermes_project/data/hermes_sales.csv", index=False)
print(f"✅ Dataset: {len(df):,} transactions | CA total: {df['amount'].sum()/1e6:.1f}M€ | Clients: {df['client_id'].nunique():,}")

# ─────────────────────────────────────────
# 2. PALETTE & STYLE
# ─────────────────────────────────────────
GOLD    = "#C9A84C"
DARK    = "#1A1A1A"
CREAM   = "#F5F0E8"
GRAY    = "#888888"
LIGHT   = "#E8E0D0"

colors_regions = ["#C9A84C","#2C2C2C","#8B7355","#D4B896","#A0856A","#6B5B45","#E8D5B0"]
colors_cat     = ["#C9A84C","#2C2C2C","#8B7355","#D4B896","#A0856A","#E8D5B0"]
colors_seasons = ["#C9A84C","#8B7355","#2C2C2C","#D4B896"]

plt.rcParams.update({
    "font.family": "serif",
    "axes.facecolor": CREAM,
    "figure.facecolor": CREAM,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": LIGHT,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "text.color": DARK,
})

# ─────────────────────────────────────────
# 3. VIZ 1 — CA MENSUEL PAR ANNÉE
# ─────────────────────────────────────────
monthly = df.groupby(["year","month"])["amount"].sum().reset_index()
fig, ax = plt.subplots(figsize=(12,5), facecolor=CREAM)
ax.set_facecolor(CREAM)

year_colors = {2022: "#8B7355", 2023: "#C9A84C", 2024: "#2C2C2C"}
for year, color in year_colors.items():
    d = monthly[monthly["year"]==year].sort_values("month")
    ax.plot(d["month"], d["amount"]/1e6, marker="o", linewidth=2.5,
            markersize=6, color=color, label=str(year))
    ax.fill_between(d["month"], d["amount"]/1e6, alpha=0.08, color=color)

ax.set_title("Évolution mensuelle du CA — Par année", fontsize=14, fontweight="bold",
             color=DARK, pad=15)
ax.set_xlabel("Mois", fontsize=10, color=GRAY)
ax.set_ylabel("CA (M€)", fontsize=10, color=GRAY)
ax.set_xticks(range(1,13))
ax.set_xticklabels(["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M€"))
ax.legend(title="Année", frameon=False)
ax.spines["bottom"].set_color(GOLD)
ax.spines["bottom"].set_linewidth(1.5)
plt.tight_layout()
plt.savefig("/home/claude/hermes_project/images/01_ca_mensuel.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Viz 1 — CA mensuel")

# ─────────────────────────────────────────
# 4. VIZ 2 — CA PAR RÉGION
# ─────────────────────────────────────────
region_ca = df.groupby("region")["amount"].sum().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10,6), facecolor=CREAM)
ax.set_facecolor(CREAM)

bars = ax.barh(region_ca.index, region_ca.values/1e6, color=colors_regions[::-1], 
               edgecolor="white", linewidth=0.5, height=0.6)

for bar, val in zip(bars, region_ca.values/1e6):
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}M€", va="center", ha="left", fontsize=10, color=DARK)

ax.set_title("CA par région — Classement décroissant", fontsize=14, fontweight="bold",
             color=DARK, pad=15)
ax.set_xlabel("Chiffre d'affaires (M€)", fontsize=10, color=GRAY)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M€"))
ax.spines["bottom"].set_color(GOLD)
plt.tight_layout()
plt.savefig("/home/claude/hermes_project/images/02_ca_region.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Viz 2 — CA par région")

# ─────────────────────────────────────────
# 5. VIZ 3 — RÉPARTITION PAR CATÉGORIE
# ─────────────────────────────────────────
cat_ca = df.groupby("category")["amount"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8,8), facecolor=CREAM)
ax.set_facecolor(CREAM)

wedges, texts, autotexts = ax.pie(
    cat_ca.values,
    labels=cat_ca.index,
    colors=colors_cat,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.75,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
for text in texts: text.set_fontsize(10)
for autotext in autotexts:
    autotext.set_fontsize(9)
    autotext.set_color("white")
    autotext.set_fontweight("bold")

ax.set_title("Répartition par catégorie — Part du CA total", fontsize=14,
             fontweight="bold", color=DARK, pad=20)
plt.tight_layout()
plt.savefig("/home/claude/hermes_project/images/03_categories.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Viz 3 — Catégories")

# ─────────────────────────────────────────
# 6. VIZ 4 — SAISONNALITÉ
# ─────────────────────────────────────────
season_ca = df.groupby("season")["amount"].sum()
season_order = ["Printemps","Été","Automne","Hiver"]
season_ca = season_ca.reindex(season_order)

fig, ax = plt.subplots(figsize=(8,5), facecolor=CREAM)
ax.set_facecolor(CREAM)

bars = ax.bar(season_ca.index, season_ca.values/1e6, color=colors_seasons,
              edgecolor="white", linewidth=1, width=0.6)
for bar, val in zip(bars, season_ca.values/1e6):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.1f}M€", ha="center", va="bottom", fontsize=11, color=DARK, fontweight="bold")

ax.set_title("Saisonnalité — CA par saison", fontsize=14, fontweight="bold", color=DARK, pad=15)
ax.set_ylabel("CA (M€)", fontsize=10, color=GRAY)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M€"))
ax.spines["bottom"].set_color(GOLD)
plt.tight_layout()
plt.savefig("/home/claude/hermes_project/images/04_saisonnalite.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Viz 4 — Saisonnalité")

# ─────────────────────────────────────────
# 7. VIZ 5 — TOP 8 PRODUITS
# ─────────────────────────────────────────
top_products = df.groupby("product")["amount"].sum().sort_values(ascending=False).head(8)
fig, ax = plt.subplots(figsize=(10,6), facecolor=CREAM)
ax.set_facecolor(CREAM)

bar_colors = [GOLD if i == 0 else DARK if i == 1 else "#8B7355" if i == 2 else LIGHT
              for i in range(len(top_products))]
bars = ax.barh(top_products.index[::-1], top_products.values[::-1]/1e6,
               color=bar_colors[::-1], edgecolor="white", linewidth=0.5, height=0.6)

for bar, val in zip(bars, top_products.values[::-1]/1e6):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}M€", va="center", ha="left", fontsize=10, color=DARK)

ax.set_title("Top 8 produits — Par chiffre d'affaires", fontsize=14, fontweight="bold",
             color=DARK, pad=15)
ax.set_xlabel("CA (M€)", fontsize=10, color=GRAY)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M€"))
ax.spines["bottom"].set_color(GOLD)
plt.tight_layout()
plt.savefig("/home/claude/hermes_project/images/05_top_produits.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Viz 5 — Top produits")

# ─────────────────────────────────────────
# 8. VIZ 6 — SEGMENTATION RFM + K-MEANS
# ─────────────────────────────────────────
snapshot = df["date"].max()
rfm = df.groupby("client_id").agg(
    recency   =("date", lambda x: (snapshot - x.max()).days),
    frequency =("client_id", "count"),
    monetary  =("amount", "sum")
).reset_index()

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["recency","frequency","monetary"]])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

cluster_labels = {
    rfm.groupby("cluster")["monetary"].mean().idxmax(): "Champions",
}
sorted_clusters = rfm.groupby("cluster")["monetary"].mean().sort_values(ascending=False).index
label_names = ["Champions", "Clients fidèles", "Clients à risque", "Inactifs"]
cluster_labels = {c: l for c, l in zip(sorted_clusters, label_names)}
rfm["segment"] = rfm["cluster"].map(cluster_labels)

segment_colors = {"Champions": GOLD, "Clients fidèles": DARK,
                  "Clients à risque": "#8B7355", "Inactifs": LIGHT}

fig, axes = plt.subplots(1, 2, figsize=(14,6), facecolor=CREAM)
for ax in axes: ax.set_facecolor(CREAM)

# Scatter RFM
for seg, color in segment_colors.items():
    mask = rfm["segment"] == seg
    axes[0].scatter(rfm[mask]["recency"], rfm[mask]["monetary"]/1000,
                    alpha=0.6, s=25, color=color, label=seg)
axes[0].set_xlabel("Récence (jours)", fontsize=10, color=GRAY)
axes[0].set_ylabel("Valeur client (k€)", fontsize=10, color=GRAY)
axes[0].set_title("Segmentation RFM — 4 clusters K-Means", fontsize=12, fontweight="bold", color=DARK)
axes[0].legend(frameon=False, fontsize=9)
axes[0].spines["bottom"].set_color(GOLD)

# Répartition segments
seg_counts = rfm["segment"].value_counts().reindex(["Champions","Clients fidèles","Clients à risque","Inactifs"])
bars = axes[1].bar(seg_counts.index, seg_counts.values,
                   color=[segment_colors[s] for s in seg_counts.index],
                   edgecolor="white", linewidth=1, width=0.6)
for bar, val in zip(bars, seg_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(val), ha="center", va="bottom", fontsize=11, fontweight="bold", color=DARK)
axes[1].set_title("Répartition des segments clients", fontsize=12, fontweight="bold", color=DARK)
axes[1].set_ylabel("Nombre de clients", fontsize=10, color=GRAY)
axes[1].spines["bottom"].set_color(GOLD)

plt.suptitle("Segmentation clients — RFM + K-Means (4 segments)", fontsize=14,
             fontweight="bold", color=DARK, y=1.02)
plt.tight_layout()
plt.savefig("/home/claude/hermes_project/images/06_rfm_kmeans.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Viz 6 — Segmentation RFM")

# ─────────────────────────────────────────
# 9. RÉSUMÉ MÉTRIQUES
# ─────────────────────────────────────────
print("\n" + "="*50)
print("📊 MÉTRIQUES CLÉS")
print("="*50)
print(f"CA total      : {df['amount'].sum()/1e6:.1f}M€")
print(f"Panier moyen  : {df['amount'].mean():,.0f}€")
print(f"Clients actifs: {df['client_id'].nunique():,}")
print(f"Top région    : {df.groupby('region')['amount'].sum().idxmax()}")
print(f"Segments RFM  : {rfm['segment'].value_counts().to_dict()}")
print("="*50)
print("\n✅ Toutes les visualisations générées dans /images/")
