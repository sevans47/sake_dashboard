# import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
import geojson
import folium
from streamlit_folium import folium_static
from folium.features import DivIcon

st.set_page_config(
    page_title="Sake Map Explorer: Introduction",
    page_icon='🍶',
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None
)

# load data
# @st.experimental_memo
def load_data():
    # prefecture data
    map_pref_region = pd.read_csv('data/map_pref_region.csv', index_col='Prefecture.1')

    # prefecture geojson
    geojson_pref = 'data/prefs_final.geojson'
    with open(geojson_pref, 'r') as file:
        jsonData = geojson.load(file)

    # sake data
    df_sake = pd.read_csv('data/sake_list_final.csv', index_col='id')

    # flavor profiles by region data
    df_flavor = pd.read_csv('data/df_flavor.csv', index_col='region')

    # flavor profiles by area data
    df_flavor_area = pd.read_csv('data/df_flavor_area.csv', index_col='region')

    return map_pref_region, jsonData, df_sake, df_flavor, df_flavor_area

map_pref_region, jsonData, df_sake, df_flavor, df_flavor_area = load_data()

# add data to session state (so it can be used across all pages easily)
if "map_pref_region" not in st.session_state:
   st.session_state["map_pref_region"] = map_pref_region

if "jsonData" not in st.session_state:
   st.session_state["jsonData"] = jsonData

if "df_sake" not in st.session_state:
   st.session_state["df_sake"] = df_sake

if "df_flavor" not in st.session_state:
   st.session_state["df_flavor"] = df_flavor

if "df_flavor_area" not in st.session_state:
   st.session_state["df_flavor_area"] = df_flavor_area

# create sidebar?
# add_sidebar = st.sidebar.selectbox("select map", ("acidity", "gravity"))

# map variables
tile_light_gray = 'https://server.arcgisonline.com/arcgis/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
attr_light_gray = 'Esri, HERE, Garmin, (c) OpenStreetMap contributors, and the GIS user community'
test_coord = [39, 140]

# quick stats! plots and map
def make_futsu_plot_stacked():
    num_futsu = df_sake.is_futsu.sum()
    num_junmai = df_sake.is_junmai.sum()
    num_non_junmai = df_sake.is_non_junmai.sum()
    ast = num_futsu + num_junmai + num_non_junmai

    df_futsu_count = pd.DataFrame(
        {'futsu-shu': num_futsu / ast * 100,
        'junmai-shu': num_junmai / ast * 100,
        'non junmai-shu': num_non_junmai / ast * 100},
        index = ['% of sake']
    )

    width = 0.1

    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_xlim(-0.1, 0.15)

    p3 = ax.bar(0, df_futsu_count['non junmai-shu'], width, bottom=df_futsu_count['futsu-shu'] + df_futsu_count['junmai-shu'], label='non junmai-shu')
    p2 = ax.bar(0, df_futsu_count['junmai-shu'], width, bottom=df_futsu_count['futsu-shu'], label='junmai-shu')
    p1 = ax.bar(0, df_futsu_count['futsu-shu'], width, label='futsu-shu')

    # splitting between futsu-shu and special designation sake
    hline = df_futsu_count['futsu-shu'].sum()
    sds = df_futsu_count['junmai-shu'].sum() + df_futsu_count['non junmai-shu'].sum()
    ax.axhline(y = hline, linestyle='--')
    ax.text(0.065, sds / 2, s=f"Special Designation \nSake ({round(sds, 1)}%)")
    ax.text(0.065, hline - 13, s=f"Futsu-shu \n({round(hline, 1)}%)")
    ax.fill_between([-1, 1], 0, hline, alpha=0.1, color='g')
    ax.fill_between([-1, 1], hline, 100, alpha=0.1, color='orange')

    # add text
    ax.set_title("Ordinary Sake (Futsu-shu) vs. Special Designation Sake")
    ax.set_xticks([])
    ax.set_ylabel("% of all sake")

    ax.legend()

    # add labels
    ax.bar_label(p1, label_type='center', fmt="%.0f%%")
    ax.bar_label(p2, label_type='center', fmt="%.0f%%")
    ax.bar_label(p3, label_type='center', fmt="%.0f%%")

    st.pyplot(fig)

def make_sd_breakdown_plot_stacked():
    num_dai = df_sake.is_dai.sum()
    num_sd = df_sake.is_special_designation.sum()
    num_non_dai = num_sd - num_dai

    sd_counts = df_sake.sake_type_eng.value_counts().drop(['Futsū-shu', 'Kijō-shu'])
    sd_counts['Junmai-shu'] = sd_counts['Junmai-shu'] + 4
    sd_counts['Honjōzō-shu'] = sd_counts['Honjōzō-shu'] + 1
    sd_counts = sd_counts.drop(['Junmai-kei', 'Honjōzō-kei'])
    sd_counts = pd.DataFrame(sd_counts / num_sd * 100).T

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.2, 0.22)
    ax.set_xticks([])

    # making bar chunks for stacked bar plot
    width=0.1
    dai_count = sd_counts[['Junmai Daiginjō-shu', 'Daiginjō-shu']].sum(axis=1).sum()
    ginjo_count = sd_counts[['Junmai Ginjō-shu', 'Ginjō-shu']].sum(axis=1).sum()
    toku_count = sd_counts[['Tokubetsu Junmai-shu', 'Tokubetsu Honjōzō-shu']].sum(axis=1).sum()
    junhon_count = sd_counts[['Junmai-shu', 'Honjōzō-shu']].sum(axis=1).sum()

    p8 = ax.bar(0, sd_counts['Junmai Daiginjō-shu'], width, bottom=100 - sd_counts['Junmai Daiginjō-shu'].sum(), label='Junmai Daiginjo')
    p7 = ax.bar(0, sd_counts['Daiginjō-shu'], width, bottom=100 - dai_count, label='Daiginjo')
    p6 = ax.bar(0, sd_counts['Junmai Ginjō-shu'], width, bottom=100 - dai_count - sd_counts['Junmai Ginjō-shu'].sum(), label='Junmai Ginjo')
    p5 = ax.bar(0, sd_counts['Ginjō-shu'], width, bottom=100 - dai_count - ginjo_count, label='Ginjo')
    p4 = ax.bar(0, sd_counts['Tokubetsu Junmai-shu'], width, bottom=junhon_count + sd_counts['Tokubetsu Honjōzō-shu'].sum(), label='Tokubetsu Junmai')
    p3 = ax.bar(0, sd_counts['Tokubetsu Honjōzō-shu'], width, bottom=junhon_count, label='Tokubetsu Honjozo')
    p2 = ax.bar(0, sd_counts['Junmai-shu'], width, bottom=sd_counts['Honjōzō-shu'].sum(), label='Junmai')
    p1 = ax.bar(0, sd_counts['Honjōzō-shu'], width, label='Honjozo')

    # plotting chunks + adding chunk labels
    sd_bar_chunks = [p1, p2, p3, p4, p5, p6, p7, p8]

    for chunk in sd_bar_chunks:
        ax.bar_label(chunk, label_type='center', fmt="%.0f%%")


    ax.legend()
    # ax.legend(loc=(0.6, 0.725))

    # add labels
    ax.set_title('Special Designation Sake Breakdown')
    ylab_padding = 5
    ax.set_ylabel(f"(lower grade){' '*ylab_padding}% of special designation sake{' '*ylab_padding}(higher grade)")

    # adding background fill
    ax.axhline(y=junhon_count, linestyle='--')
    ax.axhline(y=toku_count + junhon_count, linestyle='--')
    ax.axhline(y=100 - dai_count, linestyle='--')

    ax.fill_between([-1, 1], 0, junhon_count, alpha=0.1, color='gray')
    ax.fill_between([-1, 1], junhon_count, toku_count + junhon_count, alpha=0.1, color='brown')
    ax.fill_between([-1, 1], toku_count + junhon_count, 100 - dai_count, alpha=0.1, color='green')
    ax.fill_between([-1, 1], 100 - dai_count, 100, alpha=0.1, color='orange')

    # add text
    text_x = -0.18
    ax.text(text_x, 15, f"Junmai and \nHonjozo ({round(junhon_count, 1)}%)")
    ax.text(text_x, 38, f"Tokubetsu ({round(toku_count, 1)}%)")
    ax.text(text_x, 57, f"Ginjo ({round(ginjo_count, 1)}%)")
    ax.text(text_x, 86, f"Daiginjo ({round(dai_count, 1)}%)")

    st.pyplot(fig)

def make_dai_map_area():

    area_locs = {
        'West': [32.5, 125.5],
        'West_Central': [33.3, 133.5],
        'Central': [36.0, 141.0],
        'North': [40.8, 142.2],
    }


    dict_area_dai_ratio = map_pref_region.drop_duplicates(subset='Area').set_index('Area').region_dai_ratio.to_dict()

    m = folium.Map(
        location=test_coord,
        zoom_start=5,
        tiles=tile_light_gray,
        attr=attr_light_gray
    )

    cp = folium.Choropleth(
        geo_data=jsonData,
        name="choropleth",
        data=map_pref_region,
        columns=["Area", "region_dai_ratio"],
        key_on="properties.AREA",
        fill_color="BuPu",
        fill_opacity=0.7,
        line_opacity=0.2,
        highlight=True
    )

    # removing legend
    for key in cp._children:
        if key.startswith("color_map"):
            del(cp._children[key])

    cp.add_to(m)

    folium.LayerControl().add_to(m)

    # and finally adding a tooltip/hover to the choropleth's geojson (match name with field name in the previous line)
    folium.GeoJsonTooltip(['PREFECTURE']).add_to(cp.geojson)

    # add title
    title_html = '''
                <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                '''.format("Highest Grade Sake: Percent of sake that is daiginjo or junmai daiginjo (by area)")

    m.get_root().html.add_child(folium.Element(title_html))

    # add labels to the map
    for area, loc in area_locs.items():
        folium.map.Marker(
            loc,
            icon=DivIcon(
                icon_size=(250,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 14pt">{area}: {round(dict_area_dai_ratio[area], 3) * 100}%</div>',
                )
            ).add_to(m)

    return m

def make_sake_type_plot():
    num_all_sake = df_sake.categories.notna().sum()
    df_pct_sake_types = pd.DataFrame(df_sake[['is_nigori', 'is_genshu', 'is_nama', 'is_kijoshu', 'is_koshu', 'is_muroka', 'is_sakabune', 'is_shiboritate']].sum() / num_all_sake * 100, columns=['% of sake']).sort_values(by='% of sake')
    df_pct_sake_types.index = [name[3:].capitalize() for name in df_pct_sake_types.index]

    fig, ax = plt.subplots(figsize=(6, 4))

    st_colors = cm.Blues(np.linspace(0,1,len(df_pct_sake_types)))

    bars = ax.barh(
        df_pct_sake_types.index,
        df_pct_sake_types["% of sake"],
        color=st_colors
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlim(0, 40)
    ax.set_title('Sake Types')
    ax.set_xlabel('% of sake')
    ax.set_ylabel('')

    st.pyplot(fig)

def make_key_measurement_boxplots():
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.suptitle("Distribution of Key Measurements", fontsize=16, y=.96)
    columns = {'rice_polishing_rate': 'Rice Polishing Rate',
             'abv_avg': 'ABV',
             'acidity_avg': 'Acidity',
             'gravity_avg': 'Sake Meter Value'}

    for col, ax in zip(columns.items(), axs.ravel()):
        column = col[0]
        title = col[1]
        median = df_sake[column].median()
        mx = df_sake[column].max()
        mn = df_sake[column].min()

        bplot = ax.boxplot(df_sake[column].dropna(), patch_artist=True)
        for patch in bplot['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_title(title)
        ax.axhline(median, color='orange', linestyle='--')
        ax.set_xticks([])
        ax.text(0.52, median + ((mx - mn) * .05), f'median: {median}', fontsize=12, color='blue')

    st.pyplot(fig)

def get_amakarado(smv, acidity):
    return (193593 / (1443 + smv)) - (1.16 * acidity) - 132.57

def make_plotdf_radar():
    df_radar = df_sake.loc[:, ['rice_polishing_rate', 'abv_avg', 'gravity_avg', 'acidity_avg']]
    df_radar['amakarado'] = df_radar.apply(lambda x: get_amakarado(x.gravity_avg, x.acidity_avg), axis=1)

    df_radar_norm = ((df_radar - df_radar.min()) / (df_radar.max() - df_radar.min())).join(df_sake[['is_nama', 'is_genshu', 'is_nigori']])

    df_median_measurements_by_sake_type = df_radar_norm.groupby(['is_nama', 'is_genshu', 'is_nigori']).agg(
        SMV = ('gravity_avg', 'median'),
        rice_polishing_rate = ('rice_polishing_rate', 'median'),
        ABV = ('abv_avg', 'median'),
        acidity = ('acidity_avg', 'median'),
        amakarado = ('amakarado', 'median'),
    )

    df_median_measurements_by_sake_type = df_median_measurements_by_sake_type.iloc[np.r_[1:3, 4]].reset_index(
        drop=True).rename(
        index={0: 'Nigori', 1: 'Genshu', 2: 'Nama'}
    )

    return df_median_measurements_by_sake_type

def make_sake_type_radar():
    # import and prepare data
    df_median_measurements_by_sake_type = make_plotdf_radar()

    categories = [cat.replace("_", " ") for cat in df_median_measurements_by_sake_type.columns]
    categories = [*categories, categories[0]]

    nigori = df_median_measurements_by_sake_type.loc['Nigori'].to_list()
    genshu = df_median_measurements_by_sake_type.loc['Genshu'].to_list()
    nama = df_median_measurements_by_sake_type.loc['Nama'].to_list()
    nigori = [*nigori, nigori[0]]
    genshu = [*genshu, genshu[0]]
    nama = [*nama, nama[0]]

    # prepare plot
    label_loc = np.linspace(start=0, stop=2 * np.pi, num = len(nigori))
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    ax.set_ylim(0.1, 0.68)

    # plot radar
    ax.plot(label_loc, nigori, label='Nigori')
    ax.plot(label_loc, genshu, label='Genshu')
    ax.plot(label_loc, nama, label='Nama')

    # add fill color
    ax.fill(label_loc, nigori, facecolor='b', alpha=0.1, label='_nolegend_')
    ax.fill(label_loc, genshu, facecolor='orange', alpha=0.1, label='_nolegend_')
    ax.fill(label_loc, nama, facecolor='g', alpha=0.1, label='_nolegend_')

    # remove tick labels
    ax.set_yticklabels([])

    # add text
    plt.title('Sake Type Comparison', size=20, y=1.05)
    plt.legend()

    st.pyplot(fig)

def make_plotdf_top_rice():
    # make dataframe of top 4 rice types as % + all other rice types put together
    total_rice_type = df_sake['rice_type'].value_counts().sum()
    top_rice_types = df_sake['rice_type'].value_counts().head(4)
    other_rice_types = total_rice_type - top_rice_types.sum()
    dict_top_rice = dict(round(top_rice_types / total_rice_type * 100, 1))
    dict_top_rice['others'] = round(other_rice_types / total_rice_type * 100, 1)
    plotdf_top_rice = pd.DataFrame.from_dict(dict_top_rice, orient='index').rename(columns={0: '% of sake'}).reset_index()

    # translate sake names to English
    top4_rice = {
                '山田錦': 'Yamada Nishiki',
                '五百万石': 'Gohyakumangoku',
                '雄町': 'Omachi',
                '美山錦': 'Miyama Nishiki',
            }
    plotdf_top_rice['index'] = plotdf_top_rice['index'].replace(top4_rice)

    # reverse order and reset index
    return plotdf_top_rice.reindex(index=plotdf_top_rice.index[::-1]).set_index('index')

def make_plotdf_top_yeast():
    total_yeast_type = df_sake['yeast'].value_counts().sum()
    top_yeast_types = df_sake['yeast'].value_counts().head(6)
    other_yeast_types = total_yeast_type - top_yeast_types.sum()
    dict_top_yeast = dict(round(top_yeast_types / total_yeast_type * 100, 1))
    dict_top_yeast['others'] = round(other_yeast_types / total_yeast_type * 100, 1)
    plotdf_top_yeast = pd.DataFrame.from_dict(dict_top_yeast, orient='index').rename(columns={0: '% of sake'}).reset_index()
    plotdf_top_yeast = plotdf_top_yeast.reindex(index=plotdf_top_yeast.index[::-1]).set_index('index')

    dict_top_yeasts = {
        "協会9号(熊本酵母・香露酵母)": "#9 Yeast - Kumamoto Kobo",
        "協会901号": "#901 Yeast",
        "協会701号": "#701 Yeast",
        "協会10号(明利小川酵母)": "#10 Yeast - Ogawa Kobo",
        "協会7号(真澄酵母)": "#7 Yeast - Masumi Kobo",
        "協会1801号": "#1801 Yeast"
    }

    return plotdf_top_yeast.rename(index=dict_top_yeasts)

def make_top_rice_stacked_plot():
    # make df
    plotdf_top_rice = make_plotdf_top_rice()

    # set up plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.09, 0.15)
    ax.set_xticks([])

    # make bar chunks
    o = plotdf_top_rice.loc['others', '% of sake']
    mn = plotdf_top_rice.loc['Miyama Nishiki', '% of sake']
    om = plotdf_top_rice.loc['Omachi', '% of sake']
    go = plotdf_top_rice.loc['Gohyakumangoku', '% of sake']
    yn = plotdf_top_rice.loc['Yamada Nishiki', '% of sake']

    width=0.1
    p5 = ax.bar(0, o, width, bottom=100 - o, label='Others')
    p4 = ax.bar(0, mn, width, bottom=100 - o - mn, label='Miyama Nishiki')
    p3 = ax.bar(0, om, width, bottom=yn + go, label='Omachi')
    p2 = ax.bar(0, go, width, bottom=yn, label='Gohyakumangoku')
    p1 = ax.bar(0, yn, width, label='Yamada Nishiki')

    # plotting chunks
    rice_bar_chunks = [p1, p2, p3, p4, p5]

    for chunk in rice_bar_chunks:
        ax.bar_label(chunk, label_type='center', fmt="%.0f%%")

    ax.legend()

    # add labels
    ax.set_title('Rice Types')
    ax.set_ylabel("% of all sake")

    # adding background fill
    ax.axhline(y=100-o, linestyle='--')

    ax.fill_between([-1, 1], 0, 100 - o, alpha=0.1, color='purple')
    ax.fill_between([-1, 1], 100 - o, 100, alpha=0.1, color='blue')

    # add text
    text_x = .07
    ax.text(text_x, 25, f"Top 4 Rice Types \n({round(yn + go + om + mn, 1)}%)")
    ax.text(text_x, 65, f"Other Rice Types \n({round(o, 1)}%)")

    st.pyplot(fig)

def make_top_yeast_plot_stacked():
    # make df
    plotdf_top_yeast = make_plotdf_top_yeast()

    # set up plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.09, 0.15)
    ax.set_xticks([])

    # make bar chunks
    o = plotdf_top_yeast.loc['others', '% of sake']
    _18 = plotdf_top_yeast.loc['#1801 Yeast', '% of sake']
    _7 = plotdf_top_yeast.loc['#7 Yeast - Masumi Kobo', '% of sake']
    _10 = plotdf_top_yeast.loc['#10 Yeast - Ogawa Kobo', '% of sake']
    _701 = plotdf_top_yeast.loc['#701 Yeast', '% of sake']
    _901 = plotdf_top_yeast.loc['#901 Yeast', '% of sake']
    _9 = plotdf_top_yeast.loc['#9 Yeast - Kumamoto Kobo', '% of sake']

    width=0.1
    p7 = ax.bar(0, o, width, bottom=100 - o, label='Others')
    p6 = ax.bar(0, _18, width, bottom=100 - o - _18, label='#1801 Yeast')
    p5 = ax.bar(0, _7, width, bottom=100 - o - _18 - _7, label='#7 Yeast')
    p4 = ax.bar(0, _10, width, bottom=_9 + _901 + _701, label='#10 Yeast')
    p3 = ax.bar(0, _701, width, bottom=_9 + _901, label='#701 Yeast')
    p2 = ax.bar(0, _901, width, bottom=_9, label='#901 Yeast')
    p1 = ax.bar(0, _9, width, label='#9 Yeast')

    # plotting chunks
    yeast_bar_chunks = [p1, p2, p3, p4, p5, p6, p7]

    for chunk in yeast_bar_chunks:
        ax.bar_label(chunk, label_type='center', fmt="%.0f%%")

    ax.legend()

    # add labels
    ax.set_title('Yeast Types')
    ax.set_ylabel("% of all sake")

    # adding background fill
    ax.axhline(y=100-o, linestyle='--')

    ax.fill_between([-1, 1], 0, 100 - o, alpha=0.1, color='purple')
    ax.fill_between([-1, 1], 100 - o, 100, alpha=0.1, color='blue')

    # add text
    text_x = .07
    ax.text(text_x, 25, f"Top 6 Yeast Types \n({round(100 - o, 1)}%)")
    ax.text(text_x, 58, f"Other Yeast Types \n({round(o, 1)}%)")

    st.pyplot(fig)


# other maps
def region_color_function(feature):
    """custom colors for region map"""
    if feature['properties']['region_code'] == str(0):
        return '#FF5F5A'
    elif feature['properties']['region_code'] == str(1):
        return '#FF5A9B'
    elif feature['properties']['region_code'] == str(2):
        return '#E95AFF'
    elif feature['properties']['region_code'] == str(3):
        return '#695AFF'
    elif feature['properties']['region_code'] == str(4):
        return '#5AFFFD'
    elif feature['properties']['region_code'] == str(5):
        return '#5AFF71'
    elif feature['properties']['region_code'] == str(6):
        return '#FDFF5A'
    else:
        return '#FFB45A'

def region_map():
    # base map
    m = folium.Map(
        location=test_coord,
        zoom_start=5,
        tiles=tile_light_gray,
        attr=attr_light_gray
    )

    # adding geojson
    gj = folium.GeoJson(
        jsonData,
        style_function=lambda feature: {
            'fillColor': region_color_function(feature),
            'color' : 'gray',
            'weight' : 1,
            }
        )

    # show prefecture name when hovering over it
    folium.GeoJsonTooltip(['PREFECTURE']).add_to(gj)

    gj.add_to(m)

    # add title
    title_html = '''
                <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                '''.format("Regions of Japan")

    m.get_root().html.add_child(folium.Element(title_html))

    # add region names to map
    region_locs = {
        'Kyūshū': [32.5, 127.0],
        'Chūgoku': [36.5, 130.5],
        'Shikoku': [33.3, 133.0],
        'Kansai': [34.3, 136.2],
        'Chūbu': [38.3, 135.8],
        'Kantō': [36.5, 141.0],
        'Tōhoku': [39.5, 142.2],
        'Hokkaidō': [42.8, 144.2]
    }

    for region, loc in region_locs.items():
        folium.map.Marker(
            loc,
            icon=DivIcon(
                icon_size=(250,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 14pt">{region}</div>',
                )
            ).add_to(m)

    return m

def area_color_function(feature):
    """custom colors for area map"""
    if feature['properties']['area_code'] == str(0):
        return '#FF5F5A'
    elif feature['properties']['area_code'] == str(1):
        return '#E95AFF'
    elif feature['properties']['area_code'] == str(2):
        return '#5AFFFD'
    else:
        return '#FDFF5A'

def area_map():
    m = folium.Map(
        location=test_coord,
        zoom_start=5,
        tiles=tile_light_gray,
        attr=attr_light_gray
    )

    gj = folium.GeoJson(
        jsonData,
        style_function=lambda feature: {
            'fillColor': area_color_function(feature),
            'color' : 'gray',
            'weight' : 1,
            }
        )
    folium.GeoJsonTooltip(['PREFECTURE']).add_to(gj)

    gj.add_to(m)

    # add title
    title_html = '''
                <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                '''.format("Areas of Japan")

    m.get_root().html.add_child(folium.Element(title_html))

    # add area names
    area_locs = {
        'West': [32.5, 127.0],
        'West Central': [33.3, 133.5],
        'Central': [36.0, 141.0],
        'North': [40.8, 142.2],
    }


    for area, loc in area_locs.items():
        folium.map.Marker(
            loc,
            icon=DivIcon(
                icon_size=(250,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 14pt">{area}</div>',
                )
            ).add_to(m)

    return m

def arrow_map():
    m = folium.Map(
        location=test_coord,
        zoom_start=5,
        tiles=tile_light_gray,
        attr=attr_light_gray
    )

    gj = folium.GeoJson(
        jsonData,
        style_function=lambda feature: {
            'fillColor': area_color_function(feature),
            'color' : 'gray',
            'weight' : 1,
            }
        )
    folium.GeoJsonTooltip(['NAME']).add_to(gj)

    gj.add_to(m)

    # triangle - northeast
    folium.RegularPolygonMarker(
        location=(43.5, 139.0),
        fill_color='blue',
        number_of_sides=3,
        radius=30,
        rotation=-50
    ).add_to(m)

    # text - northeast
    folium.map.Marker(
            location=(45, 129.5),
            icon=DivIcon(
                icon_size=(250,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 22pt; text-align: center">light <br> and dry?</div>',
                )
            ).add_to(m)

    # triangle - southwest
    folium.RegularPolygonMarker(
        location=(35.5, 130.0),
        fill_color='blue',
        number_of_sides=3,
        radius=30,
        rotation=10
    ).add_to(m)

    # text - southwest
    folium.map.Marker(
            location=(40.0, 123.5),
            icon=DivIcon(
                icon_size=(250,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 22pt; text-align: center">rich and <br> sweet?</div>',
                )
            ).add_to(m)

    # connect triangles
    points=[(43.0, 138.5), (36.0, 130.5)]
    folium.PolyLine(points).add_to(m)

    return m

def make_sake_count_charts(df_sake):
    # region prep
    regions = ['Kyūshū', 'Chūgoku', 'Shikoku', 'Kansai', 'Kantō', 'Chūbu', 'Tōhoku', 'Hokkaidō']
    r_mapping = {region: i for i, region in enumerate(regions)}
    region_counts = pd.DataFrame(df_sake.groupby('region').name.count()).rename({'name': 'num_sake'}, axis=1)
    r_key = region_counts.index.map(r_mapping)

    # area prep
    areas = ['West', 'West Central', 'Central', 'North']
    areas_dict = {'Kyūshū': 'West', 'Chūgoku': 'West',
                  'Shikoku': 'West Central', 'Kansai': 'West Central',
                  'Kantō': 'Central', 'Chūbu': 'Central',
                  'Tōhoku': 'North', 'Hokkaidō': 'North'}
    a_mapping = {area: i for i, area in enumerate(areas)}
    region_counts['area'] = region_counts.index.map(areas_dict)
    area_counts = pd.DataFrame(region_counts.groupby('area').num_sake.sum())
    a_key = area_counts.index.map(a_mapping)

    # charts
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    xlim = (0, 4750)
    xlabel = 'number of unique sake bottles'

    r_colors = cm.winter(np.linspace(0,1,len(region_counts)))
    region_counts.iloc[r_key.argsort()].num_sake.plot(kind='barh', legend=False, ax=axs[0], xlim=xlim, xlabel='', color=r_colors)
    axs[0].set_title('Number of Sake Bottles (by region - from northeast to southwest)')
    axs[0].set_xlabel(xlabel)

    a_colors = cm.winter(np.linspace(0,1,len(area_counts)))
    area_counts.iloc[a_key.argsort()].num_sake.plot(kind='barh', legend=False, ax=axs[1], xlim=xlim, xlabel='', color=a_colors)
    axs[1].set_title('Number of Sake Bottles (by area)')
    axs[1].set_xlabel(xlabel)

    plt.subplots_adjust(hspace=0.3)

    st.pyplot(fig)

# page layout

# st.title("Sake Map Explorer")
# st.subheader("Using Data to Explore Sake Flavor Profiles by Region")
st.title("Sake Dashboard")
st.subheader("Quick Stats!")

make_futsu_plot_stacked()
make_sd_breakdown_plot_stacked()

st.write("**Highest Grade Sake: Percent of sake that is daiginjo or junmai daiginjo (by area)**")
dai_map = make_dai_map_area()
folium_static(dai_map)

make_sake_type_plot()
make_key_measurement_boxplots()

make_sake_type_radar()

make_top_rice_stacked_plot()
make_top_yeast_plot_stacked()

st.write(
    """
    NOTE: All the data on this dashboard is for unique bottles of sake.  Total amount
    produced is not included.
    """
)

st.header("Introduction")
st.write(
    """
    Sake is a unique Japanese alcoholic beverage with a history going back thousands
    of years. One of the joys of exploring the world of sake is discovering the wide
    variety of styles and flavors of sake across Japan. This analysis aims to use
    data to demonstrate some trends in the flavor profiles of sake across Japan.
    """
)
st.header("Regions of Japan")
st.write(
    """
    Japan is divided into 47 prefectures across 4 main islands. These 47 prefectures
    can be grouped into 8 larger regions:
    """
)

m1 = region_map()
# st_folium(m1, width=725)
# NOTE: I want to use st_folium, but it doesn't add a space between the field name and value with the folium tooltip
folium_static(m1)

st.write(
    """
    I also sometimes group these 8 regions into 4 (unofficial) larger areas to better
    show the big picture trends of the data:
    """
)

m2 = area_map()
# st_folium(m2, width=725)
folium_static(m2)

st.header("Conventional wisdom states ...")
st.write(
    """
    The famous sake expert John Gauntner, in his book “Sake: A beyond-the-basics guide
    to understanding, tasting, selection & enjoyment,” stated "The farther northeast
    you go, **the more fine-grained and compact the sake gets.**  It is often light,
    delicate, and dry … And conversely, the farther west you go, **the more big-boned,
    rich, and broad the sake flavors get;** more sake in this part of Japan are heavier
    and sweet.” (193-195, emphasis added)  Let’s see if this trend can be verified with data! (and
    hopefully we can find some interesting caveats along the way)
    """
)

m3 = arrow_map()
folium_static(m3)

st.header("About the data")
st.write(
    """
    For this analysis, I collected data on more than 14,000 different bottles of sake from the website
    nihonshu.wiki.  Each bottle had information such as the brewery’s name and location, the
    rice and yeast used, the grade and style of sake, and chemical measurements such as abv,
    acidity, and so on.  Although many bottles had incomplete information, there is enough to
    find some interesting insights.
    """
)

st.write(
    """
    When grouping the data by region, the data is quite imbalanced.  For example, there's only about 200
    bottles from Hokkaiko, but more than 3,500 from Kansai.  Further grouping these regions into
    4 large areas helps balance the data better.
    """
)

make_sake_count_charts(df_sake)


st.header("Acknowledgments")
st.write(
    """
    Thanks to the following sources, which I used extensively in order to learn enough to
    do my analysis.  If you want to learn more about sake, I highly recommend all of them!
    """
)

st.markdown(
    """
    - Gaunter, John. Sake: A beyond-the-basics guide to understanding, tasting, selection, & enjoyment. Stone Bridge Press, 2014.
    - [Sake World website](https://sake-world.com/)
    - [nihonshu.wiki](https://www.nihonshu.wiki/)
    - [The Japanese Bar](https://thejapanesebar.com/)
    """
)

st.markdown(
    """
    **Please enjoy this data-driven dive into Japanese sake and geography!**
    """
)

# if __name__ == "__main__":
#     df_prefecture, jsonData = load_data()
#     print(df_prefecture.tail(3))
#     m = acidity_map()
#     m.save('test_map.html')
