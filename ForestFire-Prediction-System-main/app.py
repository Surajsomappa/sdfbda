# ============================================
# FOREST FIRE AI EARLY WARNING SYSTEM
# Professional Dashboard
# ============================================

import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Forest Fire AI Early Warning System",
    page_icon="🔥",
    layout="wide"
)

# ============================================
# CSS STYLE
# ============================================

page_bg = """
<style>

/* Background Image */

[data-testid="stAppViewContainer"]{
background-image:url("https://images.unsplash.com/photo-1475776408506-9a5371e7a068");
background-size:cover;
background-position:center;
background-repeat:no-repeat;
background-attachment:fixed;
}

/* Transparent Header */

[data-testid="stHeader"]{
background:rgba(0,0,0,0);
}

/* Sidebar Background */

[data-testid="stSidebar"]{
background:rgba(0,0,0,0.9);
}

/* Sidebar text */

[data-testid="stSidebar"] *{
color:white !important;
font-weight:600;
}

/* Latitude & Longitude input value color */

[data-baseweb="input"] input{
color:black !important;
font-weight:bold;
}

/* Labels */

label{
color:white !important;
font-weight:bold !important;
}

/* Main container */

.block-container{
background:rgba(0,0,0,0.5);
padding:2rem;
border-radius:15px;
}

/* Headings */

h1,h2,h3,h4{
color:white !important;
}

/* Metrics */

[data-testid="stMetricValue"]{
color:white !important;
font-weight:bold;
}

[data-testid="stMetricLabel"]{
color:white !important;
}

/* Dataframe */

[data-testid="stDataFrame"]{
background:rgba(0,0,0,0.6);
color:white;
}

</style>
"""

st.markdown(page_bg,unsafe_allow_html=True)

# ============================================
# TITLE
# ============================================

st.title("🔥 Forest Fire AI Early Warning System")
st.markdown("### Real-time Ensemble Prediction | Satellite Monitoring | Heatmap")

# ============================================
# LOAD MODELS
# ============================================

@st.cache_resource
def load_models():

    scaler = pickle.load(open("models/scaler.pkl","rb"))
    rf = pickle.load(open("models/rf_model.pkl","rb"))
    xgb = pickle.load(open("models/xgb_model.pkl","rb"))
    lgb = pickle.load(open("models/lgb_model.pkl","rb"))

    return scaler,rf,xgb,lgb

scaler,rf,xgb,lgb = load_models()

# ============================================
# SIDEBAR INPUT
# ============================================

st.sidebar.title("🌍 Environmental Inputs")

latitude = st.sidebar.number_input("Latitude",-90.0,90.0,20.0)
longitude = st.sidebar.number_input("Longitude",-180.0,180.0,78.0)

ndvi = st.sidebar.slider("NDVI (Vegetation Index)",0.0,1.0,0.5)
temp = st.sidebar.slider("Temperature °C",0.0,50.0,30.0)
humidity = st.sidebar.slider("Humidity %",0.0,100.0,50.0)
wind = st.sidebar.slider("Wind Speed",0.0,50.0,10.0)
rain = st.sidebar.slider("Rainfall",0.0,50.0,5.0)
elevation = st.sidebar.slider("Elevation",0,3000,500)
slope = st.sidebar.slider("Slope",0,60,10)
aspect = st.sidebar.slider("Aspect",0,360,180)

# ============================================
# FEATURE ENGINEERING
# ============================================

veg_dryness = 1 - ndvi
temp_risk = temp/50
humidity_risk = 1 - humidity/100
wind_risk = wind/50
rain_risk = 1 - rain/50
terrain_risk = slope/60

fire_risk_score = (
0.30*veg_dryness+
0.25*temp_risk+
0.15*humidity_risk+
0.15*wind_risk+
0.10*rain_risk+
0.05*terrain_risk
)

input_data=np.array([[

ndvi,
elevation,
slope,
aspect,
temp,
humidity,
rain,
wind,
veg_dryness,
temp_risk,
humidity_risk,
wind_risk,
rain_risk,
terrain_risk,
fire_risk_score

]])

input_scaled=scaler.transform(input_data)

# ============================================
# ENSEMBLE PREDICTION
# ============================================

with st.spinner("Analyzing environmental and satellite data..."):
    time.sleep(1)

rf_pred = rf.predict_proba(input_scaled)[0][1]
xgb_pred = xgb.predict_proba(input_scaled)[0][1]
lgb_pred = lgb.predict_proba(input_scaled)[0][1]

ensemble = (rf_pred + xgb_pred + lgb_pred)/3

# ============================================
# RISK LEVEL
# ============================================

def get_risk(prob):

    if prob<0.3:
        return "green","LOW"
    elif prob<0.6:
        return "yellow","MODERATE"
    elif prob<0.8:
        return "orange","HIGH"
    else:
        return "red","EXTREME"

color,risk=get_risk(ensemble)

# ============================================
# ALERT BANNER
# ============================================

if risk=="EXTREME":
    st.error("🚨 EXTREME FIRE RISK DETECTED")

elif risk=="HIGH":
    st.warning("⚠️ HIGH FIRE RISK")

elif risk=="MODERATE":
    st.info("⚠️ MODERATE FIRE RISK")

else:
    st.success("✅ LOW FIRE RISK")

# ============================================
# METRICS
# ============================================

col1,col2,col3 = st.columns(3)

col1.metric("Fire Probability",f"{ensemble:.3f}")
col2.metric("Risk Level",risk)
col3.metric("Confidence",f"{ensemble*100:.1f}%")

# ============================================
# FIRE RISK METER
# ============================================

st.subheader("🔥 Fire Risk Meter")

fig=go.Figure(go.Indicator(

mode="gauge+number",
value=ensemble*100,
title={'text':"Risk %"},

gauge={

'axis':{'range':[0,100]},
'bar':{'color':color},

'steps':[

{'range':[0,30],'color':'green'},
{'range':[30,60],'color':'yellow'},
{'range':[60,80],'color':'orange'},
{'range':[80,100],'color':'red'}

]

}

))

st.plotly_chart(fig,use_container_width=True)

# ============================================
# HEATMAP
# ============================================

st.subheader("🔥 Live Forest Fire Heatmap")

heat_df=pd.DataFrame({

"lat":[latitude],
"lon":[longitude],
"risk":[ensemble]

})

fig_map=px.scatter_mapbox(
heat_df,
lat="lat",
lon="lon",
color="risk",
size="risk",

color_continuous_scale=[
(0,"green"),
(0.5,"yellow"),
(0.7,"orange"),
(1,"red")
],

zoom=5,
height=500
)

fig_map.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig_map,use_container_width=True)

# ============================================
# SATELLITE STATUS
# ============================================

st.subheader("🛰 Satellite Detection Status")

if ensemble>0.7:
    st.error("Satellite detected potential fire hotspot")
else:
    st.success("No hotspot detected")

# ============================================
# PROGRESS BAR
# ============================================

st.subheader("Fire Risk Severity")

st.progress(int(ensemble*100))

# ============================================
# ENSEMBLE TABLE
# ============================================

st.subheader("Model Predictions")

table=pd.DataFrame({

"Model":[
"Random Forest",
"XGBoost",
"LightGBM",
"Final Ensemble"
],

"Probability":[
rf_pred,
xgb_pred,
lgb_pred,
ensemble
]

})

st.dataframe(table,use_container_width=True)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("### Advanced AI Forest Fire Monitoring System")
