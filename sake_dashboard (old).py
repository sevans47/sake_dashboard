# import dependencies
import pandas as pd
import streamlit as st
import json
import folium
from streamlit_folium import folium_static

# load data
#@st.cache
def load_data():
    # getting region data in geojson
    map_pref_region = pd.read_csv('data/wikidf_pref.csv').drop(columns="Unnamed: 0")[['Prefecture.1', 'Region']].set_index('Prefecture.1')
    geojson_pref = 'data/jp_prefs.geojson'
    with open(geojson_pref, "r") as file:
        jsonData = json.load(file)
    for pref in jsonData['features']:
        pref['properties']['REGION'] = map_pref_region.loc[pref['properties']['NAME_JP'], 'Region']

    # prefecture data
    df_prefecture = pd.read_csv('data/prefecture_list_no_outliers.csv')
    dict_pref_names = {}
    for pref in jsonData['features']:
        dict_pref_names[pref['properties']['NAME_JP']] = pref['properties']['NAME']
    df_prefecture['prefecture_eng2'] = df_prefecture.prefecture.map(dict_pref_names)

    # region data
    df_region = df_prefecture.groupby('region').agg(
        num_sake = ('num_sake', 'sum'),
        num_company = ('num_company', 'sum'),
        abv_avg = ('abv_avg', 'mean'),
        abv_count = ('abv_count', 'sum'),
        abv_std = ('abv_avg', 'std'),
        acidity_avg = ('acidity_avg', 'mean'),
        acidity_count = ('acidity_count', 'sum'),
        acidity_std = ('acidity_avg', 'std'),
        gravity_avg = ('gravity_avg', 'mean'),
        gravity_count = ('gravity_count', 'sum'),
        gravity_std = ('gravity_avg', 'std'),
    )
    dict_region_acidity = pd.Series(df_region['acidity_avg'].values, index=df_region.index).to_dict()
    dict_region_gravity = pd.Series(df_region['gravity_avg'].values, index=df_region.index).to_dict()

    # adding regional acidity and gravity to prefecture data
    df_prefecture['average_acidity'] = df_prefecture.region.map(dict_region_acidity)
    df_prefecture['average_gravity'] = df_prefecture.region.map(dict_region_gravity)

    return df_prefecture, jsonData


df_prefecture, jsonData = load_data()

# create sidebar
# add_sidebar = st.sidebar.selectbox("select map", ("acidity", "gravity"))

# create maps
tile_light_gray = 'https://server.arcgisonline.com/arcgis/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
attr_light_gray = 'Esri, HERE, Garmin, (c) OpenStreetMap contributors, and the GIS user community'
test_coord = [39, 140]

def acidity_map():
    m = folium.Map(
        location=test_coord,
        zoom_start=5,
        tiles=tile_light_gray,
        attr=attr_light_gray
    )

    cp = folium.Choropleth(
        geo_data=jsonData,
        name="choropleth",
        data=df_prefecture,
        columns=["region", "average_acidity"],
        key_on="properties.REGION",
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="acidity",
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # creating a state indexed version of the dataframe so we can lookup values
    pref_data_indexed = df_prefecture.set_index('region')

    # looping thru the geojson object and adding a new property(acidity) and assigning a value from our dataframe
    for s in cp.geojson.data['features']:
        acidity = pref_data_indexed.loc[s['properties']['REGION'], 'average_acidity']
        if isinstance(acidity, float) == False:
            acidity = float(acidity[0])
        s['properties']['average_acidity'] = round(acidity, 3)

    # and finally adding a tooltip/hover to the choropleth's geojson
    folium.GeoJsonTooltip(['REGION', 'average_acidity']).add_to(cp.geojson)

    return m

def gravity_map():
    m = folium.Map(
        location=test_coord,
        zoom_start=5,
        tiles=tile_light_gray,
        attr=attr_light_gray
    )

    cp = folium.Choropleth(
        geo_data=jsonData,
        name="choropleth",
        data=df_prefecture,
        columns=["region", "average_gravity"],
        key_on="properties.REGION",
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="gravity",
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # creating a state indexed version of the dataframe so we can lookup values
    pref_data_indexed = df_prefecture.set_index('region')

    # looping thru the geojson object and adding a new property(acidity) and assigning a value from our dataframe
    for s in cp.geojson.data['features']:
        acidity = pref_data_indexed.loc[s['properties']['REGION'], 'average_gravity']
        if isinstance(acidity, float) == False:
            acidity = float(acidity[0])
        s['properties']['average_gravity'] = round(acidity, 3)

    # and finally adding a tooltip/hover to the choropleth's geojson
    folium.GeoJsonTooltip(['REGION', 'average_gravity']).add_to(cp.geojson)

    return m

st.title("Sake Analysis")
st.subheader("comparing acidity and gravity of sake made in the different regions of Japan")
st.text("Sake with higher acidity tastes stronger and masks sweetness.")
st.text("Sake with higher gravity has less sugar, and thus tastes drier.")
st.text("I collected data on more than 10,000 bottles of sake to compare these elements")

st.header("Acidity by region")
m1 = acidity_map()
folium_static(m1)
text = st.markdown("""
                    - Sake from the Chugoku region in the west tends to be more acidic, and thus will have a stronger taste
                    - Besides Hokkaido and Chugoku, acidity levels are very similar between regions
                    """)

st.header("Gravity by region")
m2 = gravity_map()
folium_static(m2)
st.markdown("""
            - Sake from the Tohoku region in the north tends to have less gravity, meaning it is sweeter
            - Sake from the island of Shikoku has the highest gravity values, meaning their sake is less sweet.
            - Chugoku's sake has high gravity and acidity, meaning it tends to have a stronger, drier taste.
            """)


# if __name__ == "__main__":
#     df_prefecture, jsonData = load_data()
#     print(df_prefecture.tail(3))
#     m = acidity_map()
#     m.save('test_map.html')
