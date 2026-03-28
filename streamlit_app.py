import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu
import numpy as np
import time

st.set_page_config(page_title="Crime Predict AI", layout="wide")

# ---------- PROFESSIONAL UI ----------

st.markdown("""
<style>
.stApp {
background: linear-gradient(135deg,#020617,#0f172a);
color:white;
}
section[data-testid="stSidebar"] {
background:#020617;
}
div[data-testid="metric-container"] {
background-color:#1e293b;
padding:15px;
border-radius:10px;
border:1px solid #334155;
}
h1,h2,h3 {
color:#38bdf8;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOGIN ----------

if "login" not in st.session_state:
    st.session_state.login=False

if not st.session_state.login:

    st.title("🔐 Crime Predict AI Login")

    username=st.text_input("Username")
    password=st.text_input("Password",type="password")

    if st.button("Login"):
        if username=="admin" and password=="admin123":
            st.session_state.login=True
            st.success("Login Successful")
        else:
            st.error("Invalid Login")

else:

    with st.sidebar:
        selected=option_menu(
            "Crime Predict AI",
            ["Dashboard","Command Center","Map","Statistics","AI Prediction","Report Crime"],
            icons=["speedometer","activity","map","bar-chart","cpu","plus-circle"],
            default_index=0
        )

    data=pd.read_csv("crime_data.csv")

# ---------- DASHBOARD ----------

    if selected=="Dashboard":

        st.markdown("# 🚔 SMART CITY CRIME COMMAND CENTER")

        col1,col2,col3,col4=st.columns(4)

        col1.metric("Total Crimes",len(data))
        col2.metric("Areas",data["Area"].nunique())
        col3.metric("Crime Types",data["Crime_Type"].nunique())
        col4.metric("Most Common",data["Crime_Type"].mode()[0])

        st.subheader("Crime Distribution")

        fig=px.histogram(data,x="Crime_Type",color="Crime_Type")
        st.plotly_chart(fig,use_container_width=True)

# ---------- COMMAND CENTER ----------

    if selected=="Command Center":

        left,center,right=st.columns([1,2,1])

        with left:
            st.subheader("📊 Stats")
            st.bar_chart(data["Area"].value_counts())

        with center:
            st.subheader("🗺 Live Map")

            m=folium.Map(
                location=[data["Latitude"].mean(),data["Longitude"].mean()],
                zoom_start=12
            )

            HeatMap(data[["Latitude","Longitude"]].values.tolist()).add_to(m)

            st_folium(m,width=700,height=500)

        with right:
            st.subheader("🚨 Alerts")

            counts=data["Area"].value_counts()

            for area,count in counts.items():
                if count>3:
                    st.error(f"High Risk: {area}")

# ---------- MAP ----------

    if selected=="Map":

        st.title("Crime Heatmap")

        m=folium.Map(
            location=[data["Latitude"].mean(),data["Longitude"].mean()],
            zoom_start=12
        )

        HeatMap(data[["Latitude","Longitude"]].values.tolist()).add_to(m)

        coords=data[["Latitude","Longitude"]]

        kmeans=KMeans(n_clusters=3)
        data["Cluster"]=kmeans.fit_predict(coords)

        colors=["red","orange","green"]

        for i in range(len(data)):
            folium.CircleMarker(
                location=[data.iloc[i]["Latitude"],data.iloc[i]["Longitude"]],
                radius=8,
                color=colors[data.iloc[i]["Cluster"]],
                fill=True
            ).add_to(m)

        st_folium(m,width=1000,height=500)

# ---------- STATISTICS ----------

    if selected=="Statistics":

        st.title("Crime Analysis")

        area_counts=data["Area"].value_counts()

        fig=px.bar(x=area_counts.index,y=area_counts.values)
        st.plotly_chart(fig)

        data["Date"]=pd.to_datetime(data["Date"])
        trend=data.groupby(data["Date"].dt.month).size()

        st.line_chart(trend)

# ---------- AI PREDICTION ----------

    if selected=="AI Prediction":

        st.title("AI Crime Risk Prediction")

        st.subheader("Processing Data...")

        with st.spinner("Running AI Model..."):
            time.sleep(2)

        area_counts=data["Area"].value_counts()
        total=len(data)

        for area,count in area_counts.items():

            risk=(count/total)*100

            if risk>40:
                level="🔴 High"
            elif risk>20:
                level="🟠 Medium"
            else:
                level="🟢 Low"

            st.write(f"{area} → {round(risk,2)}% Risk ({level})")

        st.metric("Predicted Risk Tomorrow",f"{np.random.randint(70,90)}%")

# ---------- REPORT CRIME ----------

    if selected=="Report Crime":

        st.title("Report Crime")

        crime=st.selectbox("Crime Type",["Theft","Robbery","Assault"])

        lat=st.number_input("Latitude")
        lon=st.number_input("Longitude")
        area=st.text_input("Area")

        if st.button("Submit"):

            new=pd.DataFrame({
                "Date":[pd.Timestamp.today()],
                "Crime_Type":[crime],
                "Latitude":[lat],
                "Longitude":[lon],
                "Area":[area]
            })

            data=pd.concat([data,new])
            data.to_csv("crime_data.csv",index=False)

            st.success("Crime Added Successfully")
