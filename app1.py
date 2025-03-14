import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static, st_folium
import networkx as nx
from folium.plugins import Draw
from folium.plugins import HeatMap, Draw
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the page config first
st.set_page_config(layout="wide")

# Streamlit App Title
st.title("🌍 Disaster Relief Dashboard")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file with disaster locations", type=["csv"])

if uploaded_file is not None:
    try:
        # Load CSV Data
        df = pd.read_csv(uploaded_file)

        # Display Dataset
        st.subheader("📊 Data Preview")
        st.write(df.head())

        # Required Columns
        required_columns = {"Location", "Latitude", "Longitude", "Disaster Type", "Supplies Needed", "Priority", "Intensity"}
        if not required_columns.issubset(df.columns):
            st.error(f"CSV must contain these columns: {required_columns}")
        else:
            # Initialize Map at the Dataset Center
            disaster_map = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=6)

            # Marker Colors for Each Disaster Type
            color_map = {
                "Flood": "blue",
                "Earthquake": "red",
                "Wildfire": "orange",
                "Hurricane": "purple",
                "Landslide": "brown",
                "Tornado": "darkpurple",
                "Drought": "lightred",
                "Tsunami": "cadetblue",
                "Volcano": "darkred",
                "Heatwave": "pink",
                "Other": "gray",
            }

            # Add Markers for Each Disaster
            for _, row in df.iterrows():
                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=(f"<b>Location:</b> {row['Location']}<br>"
                           f"<b>Disaster Type:</b> {row['Disaster Type']}<br>"
                           f"<b>Supplies Needed:</b> {row['Supplies Needed']}<br>"
                           f"<b>Priority:</b> {row['Priority']}"),
                    icon=folium.Icon(color=color_map.get(row["Disaster Type"], "gray"), icon="info-sign"),
                ).add_to(disaster_map)

            # Heatmap for Disaster Intensity
            heat_data = df[["Latitude", "Longitude", "Intensity"]].values.tolist()
            HeatMap(heat_data, min_opacity=0.3, max_zoom=10, radius=15).add_to(disaster_map)

            # Enable Drawing Tool for Custom Paths
            Draw(export=True, show_geometry_on_click=True).add_to(disaster_map)

            # Create Graph for Route Optimization
            G = nx.Graph()
            locations = df[["Location", "Latitude", "Longitude"]].values.tolist()

            # Add Nodes
            for loc in locations:
                G.add_node(loc[0], pos=(loc[1], loc[2]))

            # Add Edges Based on Distance
            for i in range(len(locations)):
                for j in range(i + 1, len(locations)):
                    lat1, lon1 = locations[i][1], locations[i][2]
                    lat2, lon2 = locations[j][1], locations[j][2]
                    distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5  # Euclidean distance
                    G.add_edge(locations[i][0], locations[j][0], weight=distance)

            # Choose a Starting Location (First Row)
            start_location = locations[0][0]

            # Compute AI Shortest Paths
            try:
                shortest_paths = nx.single_source_dijkstra_path(G, source=start_location)

                # Convert Paths to Coordinates
                path_coordinates = []
                for destination, path in shortest_paths.items():
                    route_coords = [[df.loc[df["Location"] == loc, "Latitude"].values[0],
                                     df.loc[df["Location"] == loc, "Longitude"].values[0]] for loc in path]
                    path_coordinates.append(route_coords)

                # Draw AI-Optimized Routes on Map
                for route in path_coordinates:
                    folium.PolyLine(
                        locations=route, color="blue", weight=3, opacity=0.7
                    ).add_to(disaster_map)

            except nx.NetworkXNoPath:
                st.error("⚠️ No valid paths found. Try adding more locations.")

            # Display the map with all features added
            st.subheader("🗺️ Disaster Map")
            map_data = st_folium(disaster_map, width=800, height=500)

            # Show Drawn Data (User-Added Features)
            if map_data and "all_drawings" in map_data:
                st.write("✏️ User-Drawn Features:")
                st.json(map_data["all_drawings"])

            # Display AI Routes in Text
            st.subheader("🚛 AI-Optimized Routes")
            for destination, path in shortest_paths.items():
                st.write(f"🚚 Route to {destination}: {' → '.join(path)}")

    except Exception as e:
        st.error(f"Error loading CSV: {e}")


# Dashboard for Financial and Casualty Analysis
st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
else:
    st.sidebar.warning("Please upload a CSV file.")
    st.stop()

# Data Processing & Cleaning
data = {}

# Infrastructure Damage
infra_damage = df.iloc[10:15, :3].copy()
infra_damage.columns = ["Category", "Damage Details", "Estimated Cost (USD)"]
infra_damage["Estimated Cost (USD)"] = (
    infra_damage["Estimated Cost (USD)"]
    .astype(str)
    .str.replace(r'[^\d.]', '', regex=True)
    .pipe(pd.to_numeric, errors='coerce')
)
data["Infrastructure Damage"] = infra_damage.dropna()

# Region-wise Impact
province_impact = df.iloc[22:26, :4].copy()
province_impact.columns = ["Region", "Deaths", "Houses Damaged", "Cropland Affected"]
for col in ["Deaths", "Houses Damaged"]:
    province_impact[col] = (
        province_impact[col]
        .astype(str)
        .str.replace(r'[^\d.]', '', regex=True)
        .pipe(pd.to_numeric, errors='coerce')
    )
data["Region-Wise Impact"] = province_impact.dropna()

# Damage Analysis
damage_loss = df.iloc[44:48, :7].copy()
damage_loss.columns = [
    "Region", "Damage (PKR Billion)", "Damage (USD Million)",
    "Loss (PKR Billion)", "Loss (USD Million)", "Needs (PKR Billion)", "Needs (USD Million)"
]
for col in damage_loss.columns[1:]:
    damage_loss[col] = (
        damage_loss[col]
        .astype(str)
        .str.replace(r'[^\d.]', '', regex=True)
        .pipe(pd.to_numeric, errors='coerce')
    )
data["Damage Analysis"] = damage_loss.dropna(how='all')

# Tabs for Data Section
tabs = st.tabs(["[💰 Damage Analysis]", "[🏥 Infrastructure]", "[📋 Full Data]"])
with tabs[0]:
    if "Damage Analysis" in data:
        st.subheader("Financial Impact Assessment")
        st.dataframe(
            data["Damage Analysis"].style.format({
                'Damage (PKR Billion)': '{:,.1f}B',
                'Damage (USD Million)': '${:,.1f}M',
                'Loss (PKR Billion)': '{:,.1f}B',
                'Loss (USD Million)': '${:,.1f}M'
            }),
            use_container_width=True
        )
with tabs[1]:
    if "Infrastructure Damage" in data:
        st.subheader("Critical Infrastructure Damage")
        st.dataframe(data["Infrastructure Damage"], use_container_width=True)
with tabs[2]:
    st.subheader("Complete Dataset Overview")
    st.dataframe(df, use_container_width=True)

# Emergency Calculator
st.sidebar.header("🚨 Emergency Calculator")
population = st.sidebar.number_input("Affected population size:", min_value=1000, value=10000, step=1000)
st.sidebar.write(f"💧 Water: **{(population * 15):,} liters**")
st.sidebar.write(f"🍲 Food: **{(population * 2.1):,} kg**")
st.sidebar.write(f"🏥 Medical Kits: **{np.ceil(population/1000):.0f} units**")
st.sidebar.write(f"🛏️ Shelter: **{np.ceil(population/5):,} family tents**")
