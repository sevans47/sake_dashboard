# import dependencies
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup as bs

# configure page
st.set_page_config(
    page_title="Rice",
    page_icon='🌾',
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None
)

# get data
map_pref_region = st.session_state["map_pref_region"]
jsonData = st.session_state["jsonData"]
df_sake = st.session_state["df_sake"]
df_flavor = st.session_state['df_flavor']
df_flavor_area = st.session_state["df_flavor_area"]

# dataframe functions
def get_wikidf_pref():
    # get table from wikipedia
    wikiurl="https://en.wikipedia.org/wiki/Prefectures_of_Japan"
    table_class="wikitable sortable jquery-tablesorter"
    response=requests.get(wikiurl)
    soup = bs(response.text, 'html.parser')
    table=soup.find('table',{'class':"wikitable"})

    # convert table to df
    df_pref=pd.read_html(str(table))

    return pd.DataFrame(df_pref[0])

def make_top_rice_type_df(df_sake):
    rice_type_eng_dict = {
        '山田錦': 'Yamada Nishiki',
        '五百万石': 'Gohyakumangoku',
        '雄町': 'Omachi',
        '美山錦': 'Miyama Nishiki',
        'ひだほまれ': 'Hidahomare',
        '日本晴': 'Nihonbare',
        'レイホウ': 'Reihou',
        'アケボノ': 'Akebono',
        '八反錦': 'Hattan Nishiki',
        '山田錦 / 五百万石': 'Yamada Nishiki / Gohyakumangoku',
        '松山三井': 'Matsuyama Mitsui',
        '出羽燦々': 'Dewasansan',
        '秋田酒こまち': 'Akita Sake Komachi',
        '玉栄': 'Tamazakae',
        '吟風': 'Ginpu',
        '夢一献': 'Yumeikkon'
    }

    origin_to_pref = {
        '愛知県総合農業試験場': '愛知県',
        '山形県農業試験場': '山形県',
        '福岡県農業総合試験場': '福岡県',
        '宮崎県総合農業試験場': '宮崎県',
        '佐賀県農業試験研究センター': '佐賀県',
        '九州沖縄農業研究センター': '熊本県',
        '山形県立農業試験場': '山形県',
        '九州農業試験場': '熊本県',
        '中国': np.nan,
        'フランス': np.nan
    }

    # get Kumamoto as rice origin for Reihou type rice
    df_sake['rice_origin'] = np.where((df_sake.rice_type == 'レイホウ'), '熊本県', df_sake.rice_origin)

    # make dictionary for jp -> eng prefecture names
    wikidf_prefecture = get_wikidf_pref()
    dict_pref_jp_eng = dict(zip(wikidf_prefecture['Prefecture.1'], wikidf_prefecture['Prefecture']))

    # add columns
    top_rice_types = pd.DataFrame(df_sake.groupby(['rice_type', 'rice_origin']).name.count()).rename({'name': 'num_sake'}, axis=1).sort_values(by='num_sake', ascending=False).reset_index().head(16)
    top_rice_types['rice_type_eng'] = top_rice_types.rice_type.map(rice_type_eng_dict)
    top_rice_types['prefecture'] = top_rice_types.rice_origin.replace(origin_to_pref)
    top_rice_types['prefecture_eng'] = top_rice_types.prefecture.map(dict_pref_jp_eng)
    top_rice_types.loc[9, 'prefecture_eng'] = 'Hyōgo / Niigata'

    return top_rice_types.set_index('rice_type')[['rice_origin', 'prefecture', 'num_sake', 'rice_type_eng', 'prefecture_eng']]

def make_top4_rice_df():

    top4_rice = {
            '山田錦': 'Yamada Nishiki',
            '五百万石': 'Gohyakumangoku',
            '雄町': 'Omachi',
            '美山錦': 'Miyama Nishiki',
        }

    df_top4_rice = df_sake.groupby('rice_type').agg(
        num_sake = ('name', 'count'),
        num_company = ('company', 'nunique'),
        abv_avg = ('abv_avg', 'mean'),
        abv_count = ('abv_avg', 'count'),
        abv_std = ('abv_avg', 'std'),
        acidity_avg = ('acidity_avg', 'mean'),
        acidity_count = ('acidity_avg', 'count'),
        acidity_std = ('acidity_avg', 'std'),
        gravity_avg = ('gravity_avg', 'mean'),
        gravity_count = ('gravity_avg', 'count'),
        gravity_std = ('gravity_avg', 'std'),
        dry_light = ('dry_light', 'sum'),
        dry_rich = ('dry_rich', 'sum'),
        sweet_light = ('sweet_light', 'sum'),
        sweet_rich = ('sweet_rich', 'sum'),
    ).loc[list(top4_rice.keys())]

    # total, dry, sweet, light, rich count columns
    df_top4_rice['total'] = df_top4_rice[['dry_light', 'dry_rich', 'sweet_light', 'sweet_rich']].sum(axis=1)
    df_top4_rice['dry_ratio'] = df_top4_rice[['dry_light', 'dry_rich']].sum(axis=1) / df_top4_rice['total']
    df_top4_rice['sweet_ratio'] = 1.0 - df_top4_rice['dry_ratio']
    df_top4_rice['light_ratio'] = df_top4_rice[['dry_light', 'sweet_light']].sum(axis=1) / df_top4_rice['total']
    df_top4_rice['rich_ratio'] = 1.0 - df_top4_rice['light_ratio']

    # English name
    df_top4_rice['rice_type_eng'] = df_top4_rice.index.map(top4_rice)
#     df_top4

    cols = list(df_top4_rice.columns)
    cols = cols[-1:] + cols[:-1]

    return df_top4_rice[cols]

top_rice_types = make_top_rice_type_df(df_sake)
df_top4_rice = make_top4_rice_df()

# chart functions
def plot_top_rice_types(top_rice_types):
    fig, axs = plt.subplots(1, 2, figsize=(10,5), sharey=True)
    fig.suptitle("Number of Sake by Rice Type", fontsize=16, y=1.0)
    plt.subplots_adjust(wspace=0.05)

    top_rice_types = top_rice_types.drop('山田錦 / 五百万石', axis=0)
    top_rice_types['rice_type_pref'] = top_rice_types.rice_type_eng + " (" + top_rice_types.prefecture_eng + ")"

    top_rice_types.set_index('rice_type_pref').sort_values(by='num_sake').plot(
        kind='barh',
        ax=axs[0],
        legend=False,
        grid=True,
        fontsize=16
    )

    top_rice_types.set_index('rice_type_pref').sort_values(by='num_sake').plot(
        kind='barh',
        ax=axs[1],
        xlim=(0,750),
        legend=False,
        grid=True,
        fontsize=16
    )

    axs[0].set_ylabel("rice type (prefecture of origin)", fontsize=16)
    axs[0].set_title('Full Scale', fontsize=16)
    axs[0].set_xlabel("number of sake", fontsize=16)
    axs[1].set_title('Zoomed In', fontsize=16)
    axs[1].set_xlabel("number of sake", fontsize=16)

    st.pyplot(fig)

def top4_rice_stacked_bar(df_top4_rice):
    top4_rice = {
            '山田錦': 'Yamada Nishiki',
            '五百万石': 'Gohyakumangoku',
            '雄町': 'Omachi',
            '美山錦': 'Miyama Nishiki',
        }

    # remove underscores
    columns = {name: name.replace("_", " ") for name in df_top4_rice.columns}
    rice = {name: name.replace("_", " ") for name in df_top4_rice.index}
    df_top4_rice = df_top4_rice.rename(columns, axis=1)
    df_top4_rice = df_top4_rice.rename(rice, axis=0)
    df_top4_rice = df_top4_rice.set_index(df_top4_rice['rice type eng'])
    df_top4_rice.index.name = 'rice type'

    # sort rice so order will appear most popular to least popular from top to bottom
    rice_order = ['Miyama Nishiki', 'Omachi', 'Gohyakumangoku', 'Yamada Nishiki']
    r_mapping = {rice: i for i, rice in enumerate(rice_order)}
    r_key = df_top4_rice.index.map(r_mapping)

    # make stacked bar plots
    fig, axs = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    plt.subplots_adjust(wspace=0.1)

    df_top4_rice.iloc[r_key.argsort()][['dry ratio', 'sweet ratio']].mul(100).plot(
        kind='barh',
        stacked=True,
        ylabel="% of sake",
        legend=True,
        ax=axs[0],
        fontsize=16
    )

    df_top4_rice.iloc[r_key.argsort()][['light ratio', 'rich ratio']].mul(100).plot(
        kind='barh',
        stacked=True,
        ylabel="% of sake",
        legend=True,
        ax=axs[1],
        fontsize=16
    )

    # calculate overall dry and light ratios
    total = df_sake[['dry_light', 'dry_rich', 'sweet_light', 'sweet_rich']].sum().sum()
    dry_total = df_sake[['dry_light', 'dry_rich']].sum().sum()
    light_total = df_sake[['dry_light', 'sweet_light']].sum().sum()
    dry_rat = dry_total / total * 100
    light_rat = light_total / total * 100

    # plotting overal dry and light ratios
    axs[0].axvline(dry_rat, label=f'all sake: {int(dry_rat)}%', color='r', linestyle='dashed')
    axs[1].axvline(light_rat, label=f'all sake: {int(light_rat)}%', color='r', linestyle='dashed')

    # adding text
    axs[0].set_title("Dry : Sweet Ratio by Top 4 Rice Types", fontsize=20)
    axs[1].set_title("Light : Rich Ratio by Top 4 Rice Types", fontsize=20)
    axs[0].set_xlabel("% of sake", fontsize=16)
    axs[1].set_xlabel("% of sake", fontsize=16)
    axs[0].set_ylabel("rice type", fontsize=16)
    axs[1].set_ylabel("")

    axs[0].legend(loc='lower left', fontsize=16)
    axs[1].legend(loc='lower left', fontsize=16)



    st.pyplot(fig)

plot_top_rice_types(top_rice_types)

top4_rice_stacked_bar(df_top4_rice)
