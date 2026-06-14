"""
Hermès Sales Performance Dashboard — Plotly/Dash
Data Analyst: Fatima Aagour
Données entièrement simulées à des fins analytiques.
Pour lancer : python3 dashboard.py
Puis ouvrir : http://127.0.0.1:8050
"""

import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

GOLD  = "#C9A84C"
DARK  = "#1A1A1A"
CREAM = "#F5F0E8"
LIGHT = "#E8E0D0"
GRAY  = "#888888"
WHITE = "#FFFFFF"
COLORS_CAT = [GOLD, DARK, "#8B7355", "#D4B896", "#A0856A", "#E8D5B0"]
COLORS_REG = [GOLD, DARK, "#8B7355", "#D4B896", "#A0856A", "#6B5B45", "#E8D5B0"]
SEG_COLORS = {"Champions": GOLD, "Clients fidèles": DARK,
              "Clients à risque": "#8B7355", "Inactifs": LIGHT}

np.random.seed(42)
TARGET_CA = 85_000_000
N_CLIENTS = 2982
N_TRANSACTIONS = 15_000

regions    = {"Chine":0.222,"Europe":0.200,"Japon":0.155,"Amériques":0.145,
              "Asie-Pacifique":0.138,"Moyen-Orient":0.080,"Autres":0.060}
categories = {"Maroquinerie":0.42,"Soie & Textiles":0.18,"Prêt-à-Porter":0.15,
              "Parfums & Beauté":0.10,"Horlogerie":0.09,"Joaillerie":0.06}
products   = ["Birkin 35","Kelly 28","Constance 24","Picotin Lock 18",
              "Garden Party 36","Evelyne III 29","Herbag Zip 31","Bolide 27"]
base_prices = {"Birkin 35":11000,"Kelly 28":9500,"Constance 24":8000,
               "Picotin Lock 18":3200,"Garden Party 36":2800,
               "Evelyne III 29":2400,"Herbag Zip 31":3000,"Bolide 27":6500}

dates     = pd.date_range("2022-01-01","2024-12-31",periods=N_TRANSACTIONS)
dates     = pd.to_datetime(np.sort(np.random.choice(dates,N_TRANSACTIONS,replace=False)))
region_l  = np.random.choice(list(regions.keys()),N_TRANSACTIONS,p=list(regions.values()))
cat_l     = np.random.choice(list(categories.keys()),N_TRANSACTIONS,p=list(categories.values()))
prod_l    = np.random.choice(products,N_TRANSACTIONS)
client_ids= np.random.randint(1,N_CLIENTS+1,N_TRANSACTIONS)
amounts   = np.array([base_prices[p]*np.random.uniform(0.85,1.15) for p in prod_l])
amounts   = amounts*(TARGET_CA/amounts.sum())

df = pd.DataFrame({"date":dates,"client_id":client_ids,"region":region_l,
                   "category":cat_l,"product":prod_l,"amount":amounts,
                   "year":dates.year,"month":dates.month})
df["season"] = df["month"].apply(
    lambda m: "Printemps" if m in [3,4,5] else "Été" if m in [6,7,8]
              else "Automne" if m in [9,10,11] else "Hiver")

snapshot = df["date"].max()
rfm = df.groupby("client_id").agg(
    recency=("date",lambda x:(snapshot-x.max()).days),
    frequency=("client_id","count"),monetary=("amount","sum")).reset_index()
rfm_scaled = StandardScaler().fit_transform(rfm[["recency","frequency","monetary"]])
rfm["cluster"] = KMeans(n_clusters=4,random_state=42,n_init=10).fit_predict(rfm_scaled)
sorted_c = rfm.groupby("cluster")["monetary"].mean().sort_values(ascending=False).index
labels   = ["Champions","Clients fidèles","Clients à risque","Inactifs"]
rfm["segment"] = rfm["cluster"].map({c:l for c,l in zip(sorted_c,labels)})

app = Dash(__name__)
app.title = "Hermès Sales Dashboard"

CARD = {"background":WHITE,"borderRadius":"12px","padding":"20px",
        "boxShadow":"0 2px 12px rgba(0,0,0,0.06)","border":f"1px solid {LIGHT}"}
METRIC = {**CARD,"textAlign":"center","flex":"1","margin":"8px"}

def base_layout(title):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CREAM,
        font=dict(family="Georgia, serif", color=DARK),
        margin=dict(l=50,r=30,t=50,b=40),
        title=dict(text=title, font=dict(size=13)),
    )

app.layout = html.Div(style={"background":CREAM,"minHeight":"100vh",
                              "fontFamily":"Georgia, serif"},children=[
    html.Div(style={"background":DARK,"padding":"32px 40px",
                    "borderBottom":f"3px solid {GOLD}"},children=[
        html.H1("Sales Performance Dashboard",
                style={"color":WHITE,"margin":0,"fontSize":"28px",
                       "fontWeight":"300","letterSpacing":"2px"}),
        html.P("Hermès · Données simulées · 2022–2024",
               style={"color":GOLD,"margin":"6px 0 0","fontSize":"13px"}),
        html.P("Fatima Aagour · Data Analyst",
               style={"color":GRAY,"margin":"4px 0 0","fontSize":"12px"}),
    ]),
    html.Div(style={"maxWidth":"1400px","margin":"0 auto","padding":"30px 20px"},children=[
        html.Div(style={**CARD,"marginBottom":"24px","display":"flex",
                        "gap":"20px","alignItems":"center","flexWrap":"wrap"},children=[
            html.Div([
                html.Label("Année",style={"color":GRAY,"fontSize":"11px",
                                          "letterSpacing":"1px","display":"block","marginBottom":"6px"}),
                dcc.Dropdown(id="year-filter",
                    options=[{"label":"Toutes les années","value":"all"}]+
                            [{"label":str(y),"value":y} for y in [2022,2023,2024]],
                    value="all",clearable=False,style={"width":"200px","fontSize":"13px"}),
            ]),
            html.Div([
                html.Label("Région",style={"color":GRAY,"fontSize":"11px",
                                           "letterSpacing":"1px","display":"block","marginBottom":"6px"}),
                dcc.Dropdown(id="region-filter",
                    options=[{"label":"Toutes les régions","value":"all"}]+
                            [{"label":r,"value":r} for r in regions.keys()],
                    value="all",clearable=False,style={"width":"220px","fontSize":"13px"}),
            ]),
        ]),
        html.Div(id="kpi-row",style={"display":"flex","flexWrap":"wrap","marginBottom":"24px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"2fr 1fr",
                        "gap":"20px","marginBottom":"20px"},children=[
            html.Div(dcc.Graph(id="monthly-chart"),style=CARD),
            html.Div(dcc.Graph(id="category-chart"),style=CARD),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr",
                        "gap":"20px","marginBottom":"20px"},children=[
            html.Div(dcc.Graph(id="region-chart"),style=CARD),
            html.Div(dcc.Graph(id="season-chart"),style=CARD),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr",
                        "gap":"20px","marginBottom":"20px"},children=[
            html.Div(dcc.Graph(id="products-chart"),style=CARD),
            html.Div(dcc.Graph(id="rfm-chart"),style=CARD),
        ]),
        html.Div(style={"textAlign":"center","padding":"20px","color":GRAY,"fontSize":"11px"},children=[
            html.P("Données simulées à des fins analytiques · Fatima Aagour · aagour.fati@gmail.com"),
        ]),
    ]),
])

@app.callback(
    [Output("kpi-row","children"),
     Output("monthly-chart","figure"),
     Output("category-chart","figure"),
     Output("region-chart","figure"),
     Output("season-chart","figure"),
     Output("products-chart","figure"),
     Output("rfm-chart","figure")],
    [Input("year-filter","value"),Input("region-filter","value")]
)
def update_all(year,region):
    d = df.copy()
    if year != "all":   d = d[d["year"]==int(year)]
    if region != "all": d = d[d["region"]==region]

    ca    = d["amount"].sum()
    panier= d["amount"].mean() if len(d) else 0
    n_cli = d["client_id"].nunique()
    top_r = d.groupby("region")["amount"].sum().idxmax() if len(d) else "—"
    top_ca= d.groupby("region")["amount"].sum().max() if len(d) else 0

    def kpi(val,label,sub=""):
        return html.Div(style=METRIC,children=[
            html.P(label,style={"color":GRAY,"fontSize":"11px","letterSpacing":"1px","margin":"0 0 8px"}),
            html.P(val,  style={"color":DARK,"fontSize":"26px","fontWeight":"600","margin":0}),
            html.P(sub,  style={"color":GOLD,"fontSize":"11px","margin":"4px 0 0"}),
        ])

    kpis = [kpi(f"{ca/1e6:.1f}M€","Chiffre d'affaires","2022–2024"),
            kpi(f"{panier:,.0f}€","Panier moyen","par transaction"),
            kpi(f"{n_cli:,}","Clients actifs","sur la période"),
            kpi(top_r,"Top région",f"{top_ca/1e6:.1f}M€")]

    # Monthly
    monthly = d.groupby(["year","month"])["amount"].sum().reset_index()
    fig_m = go.Figure()
    for y,col in {2022:"#8B7355",2023:GOLD,2024:DARK}.items():
        md = monthly[monthly["year"]==y].sort_values("month")
        if len(md):
            fig_m.add_trace(go.Scatter(
                x=md["month"],y=md["amount"]/1e6,mode="lines+markers",name=str(y),
                line=dict(color=col,width=2.5),marker=dict(size=6),
                fill="tozeroy",fillcolor="rgba(201,168,76,0.06)"))
    fig_m.update_layout(**base_layout("Évolution mensuelle du CA"),
        xaxis=dict(tickvals=list(range(1,13)),showgrid=False,linecolor=LIGHT,
                   ticktext=["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]),
        yaxis=dict(ticksuffix="M€",showgrid=True,gridcolor=LIGHT,linecolor=LIGHT),
        legend=dict(orientation="h",y=1.1))

    # Category
    cat_ca = d.groupby("category")["amount"].sum().sort_values(ascending=False)
    fig_c = go.Figure(go.Pie(
        labels=cat_ca.index,values=cat_ca.values,
        marker=dict(colors=COLORS_CAT,line=dict(color=WHITE,width=2)),
        textinfo="percent",hole=0.3,
        hovertemplate="%{label}<br>%{value:,.0f}€<extra></extra>"))
    fig_c.update_layout(**base_layout("Répartition par catégorie"),
        legend=dict(font=dict(size=10)))

    # Region
    reg_ca = d.groupby("region")["amount"].sum().sort_values(ascending=True)
    fig_r = go.Figure(go.Bar(
        x=reg_ca.values/1e6,y=reg_ca.index,orientation="h",
        marker=dict(color=COLORS_REG[::-1],line=dict(color=WHITE,width=0.5)),
        text=[f"{v:.1f}M€" for v in reg_ca.values/1e6],textposition="outside",
        hovertemplate="%{y}<br>%{x:.1f}M€<extra></extra>"))
    fig_r.update_layout(**base_layout("CA par région"),
        xaxis=dict(ticksuffix="M€",showgrid=True,gridcolor=LIGHT,linecolor=LIGHT),
        yaxis=dict(showgrid=False,linecolor=LIGHT))

    # Season
    sea_ca = d.groupby("season")["amount"].sum().reindex(
        ["Printemps","Été","Automne","Hiver"]).fillna(0)
    fig_s = go.Figure(go.Bar(
        x=sea_ca.index,y=sea_ca.values/1e6,
        marker=dict(color=[GOLD,"#8B7355",DARK,LIGHT],line=dict(color=WHITE,width=1)),
        text=[f"{v:.1f}M€" for v in sea_ca.values/1e6],textposition="outside"))
    fig_s.update_layout(**base_layout("Saisonnalité"),
        xaxis=dict(showgrid=False,linecolor=LIGHT),
        yaxis=dict(ticksuffix="M€",showgrid=True,gridcolor=LIGHT,linecolor=LIGHT))

    # Products
    top_p = d.groupby("product")["amount"].sum().sort_values(ascending=False).head(8)
    bar_colors = [GOLD if i==0 else DARK if i==1 else "#8B7355" if i==2
                  else LIGHT for i in range(len(top_p))]
    fig_p = go.Figure(go.Bar(
        x=top_p.values/1e6,y=top_p.index,orientation="h",
        marker=dict(color=bar_colors,line=dict(color=WHITE,width=0.5)),
        text=[f"{v:.1f}M€" for v in top_p.values/1e6],textposition="outside"))
    fig_p.update_layout(**base_layout("Top 8 produits"),
        xaxis=dict(ticksuffix="M€",showgrid=True,gridcolor=LIGHT,linecolor=LIGHT),
        yaxis=dict(showgrid=False,linecolor=LIGHT,autorange="reversed"))

    # RFM
    fig_rfm = go.Figure()
    for seg,col in SEG_COLORS.items():
        mask = rfm["segment"]==seg
        fig_rfm.add_trace(go.Scatter(
            x=rfm[mask]["recency"],y=rfm[mask]["monetary"]/1000,
            mode="markers",name=seg,
            marker=dict(color=col,size=5,opacity=0.7),
            hovertemplate=f"<b>{seg}</b><br>Récence: %{{x}}j<br>Valeur: %{{y:.1f}}k€<extra></extra>"))
    fig_rfm.update_layout(**base_layout("Segmentation RFM — K-Means"),
        xaxis=dict(title="Récence (jours)",showgrid=False,linecolor=LIGHT),
        yaxis=dict(title="Valeur (k€)",ticksuffix="k€",showgrid=True,
                   gridcolor=LIGHT,linecolor=LIGHT),
        legend=dict(orientation="h",y=1.1))

    return kpis,fig_m,fig_c,fig_r,fig_s,fig_p,fig_rfm

if __name__ == "__main__":
    print("🚀 Dashboard lancé sur http://127.0.0.1:8050")
    app.run(debug=True)
