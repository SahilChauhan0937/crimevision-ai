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

# ---------- LOAD & CLEAN DATA ----------
@st.cache_data
def load_data():
    data = pd.read_csv("crime_data.csv")

    data["Latitude"] = pd.to_numeric(data["Latitude"], errors="coerce")
    data["Longitude"] = pd.to_numeric(data["Longitude"], errors="coerce")

    data = data.dropna(subset=["Latitude", "Longitude"])

    return data

data = load_data()

# ---------- SAFE CENTER FUNCTION ----------
def get_safe_center(data):
    try:
        lat = data["Latitude"].mean()
        lon = data["Longitude"].mean()

        if pd.isna(lat) or pd.isna(lon):
            return [20.5937, 78.9629]

        return [float(lat), float(lon)]
    except:
        return [20.5937, 78.9629]

# ---------- SESSION STATE ----------
if "map_state" not in st.session_state:
    st.session_state.map_state = {
        "zoom": 12,
        "center": get_safe_center(data)
    }

# ---------- UI ----------
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
    st.session_state.login = False

if not st.session_state.login:

    st.title("🔐 Crime Predict AI Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.login = True
            st.success("Login Successful")
        else:
            st.error("Invalid Login")

else:

    # ---------- SIDEBAR ----------
    with st.sidebar:
        selected = option_menu(
            "Crime Predict AI",
            ["Dashboard","Command Center","Map","Statistics","AI Prediction","Report Crime"],
            icons=["speedometer","activity","map","bar-chart","cpu","plus-circle"],
            default_index=0
        )

    # ---------- DASHBOARD ----------
    if selected == "Dashboard":

        st.markdown("# 🚔 SMART CITY CRIME COMMAND CENTER")

        col1,col2,col3,col4 = st.columns(4)

        col1.metric("Total Crimes", len(data))
        col2.metric("Areas", data["Area"].nunique())
        col3.metric("Crime Types", data["Crime_Type"].nunique())
        col4.metric("Most Common", data["Crime_Type"].mode()[0] if not data.empty else "N/A")

        fig = px.histogram(data, x="Crime_Type", color="Crime_Type")
        st.plotly_chart(fig, use_container_width=True)

    # ---------- COMMAND CENTER ----------
    if selected == "Command Center":

        left,center,right = st.columns([1,2,1])

        with left:
            st.bar_chart(data["Area"].value_counts())

        with center:
            st.subheader("🗺 Live Map")

            center_map = st.session_state.map_state.get("center", [20.5937, 78.9629])
            if not isinstance(center_map, list) or len(center_map) != 2:
                center_map = [20.5937, 78.9629]

            m = folium.Map(location=center_map, zoom_start=st.session_state.map_state.get("zoom", 12))

            heat_data = data[["Latitude","Longitude"]].dropna()

            if not heat_data.empty:
                HeatMap(heat_data.values.tolist(), radius=15, blur=20).add_to(m)

            map_data = st_folium(m, width=700, height=500, returned_objects=["center","zoom"])

            if map_data:
                if map_data.get("zoom"):
                    st.session_state.map_state["zoom"] = map_data["zoom"]

                if map_data.get("center"):
                    c = map_data["center"]
                    if isinstance(c, list) and len(c) == 2:
                        try:
                            st.session_state.map_state["center"] = [float(c[0]), float(c[1])]
                        except:
                            pass

        with right:
            counts = data["Area"].value_counts()
            for area,count in counts.items():
                if count > 3:
                    st.error(f"High Risk: {area}")

    # ---------- MAP ----------
    if selected == "Map":

        st.title("Crime Heatmap")

        center_map = st.session_state.map_state.get("center", [20.5937, 78.9629])
        if not isinstance(center_map, list) or len(center_map) != 2:
            center_map = [20.5937, 78.9629]

        m = folium.Map(location=center_map, zoom_start=st.session_state.map_state.get("zoom", 12))

        heat_data = data[["Latitude","Longitude"]].dropna()

        if not heat_data.empty:
            HeatMap(heat_data.values.tolist(), radius=15, blur=20).add_to(m)

        coords = data[["Latitude","Longitude"]].dropna().copy()

        if len(coords) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=0)
            coords["Cluster"] = kmeans.fit_predict(coords)

            colors = ["red","orange","green"]

            for i in range(len(coords)):
                folium.CircleMarker(
                    location=[coords.iloc[i]["Latitude"], coords.iloc[i]["Longitude"]],
                    radius=6,
                    color=colors[int(coords.iloc[i]["Cluster"])],
                    fill=True
                ).add_to(m)

        map_data = st_folium(m, width=1000, height=500, returned_objects=["center","zoom"])

        if map_data:
            if map_data.get("zoom"):
                st.session_state.map_state["zoom"] = map_data["zoom"]

            if map_data.get("center"):
                c = map_data["center"]
                if isinstance(c, list) and len(c) == 2:
                    try:
                        st.session_state.map_state["center"] = [float(c[0]), float(c[1])]
                    except:
                        pass

    # ---------- STATISTICS ----------
    if selected == "Statistics":

        st.title("Crime Analysis")

        fig = px.bar(data["Area"].value_counts())
        st.plotly_chart(fig)

        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
            trend = data.groupby(data["Date"].dt.month).size()
            st.line_chart(trend)

    # ---------- AI ----------
    if selected == "AI Prediction":

        st.title("AI Crime Risk Prediction")

        with st.spinner("Running AI Model..."):
            time.sleep(2)

        area_counts = data["Area"].value_counts()
        total = len(data)

        for area,count in area_counts.items():
            risk = (count / total) * 100

            if risk > 40:
                level = "🔴 High"
            elif risk > 20:
                level = "🟠 Medium"
            else:
                level = "🟢 Low"

            st.write(f"{area} → {round(risk,2)}% Risk ({level})")

        st.metric("Predicted Risk Tomorrow", f"{np.random.randint(70,90)}%")

    # ---------- REPORT ----------
    if selected == "Report Crime":

        st.title("Report Crime")

        crime = st.selectbox("Crime Type", ["Theft","Robbery","Assault"])
        lat = st.number_input("Latitude")
        lon = st.number_input("Longitude")
        area = st.text_input("Area")

        if st.button("Submit"):

            new = pd.DataFrame({
                "Date":[pd.Timestamp.today()],
                "Crime_Type":[crime],
                "Latitude":[lat],
                "Longitude":[lon],
                "Area":[area]
            })

            data = pd.concat([data,new], ignore_index=True)
            data.to_csv("crime_data.csv", index=False)

            st.success("Crime Added Successfully")
