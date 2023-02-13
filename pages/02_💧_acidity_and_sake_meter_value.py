import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import folium
from scipy import stats
from streamlit_folium import folium_static
from folium.features import DivIcon

st.set_page_config(
    page_title="Sake Dashboard: Acidity / SMV",
    page_icon='💧',
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





# functions for charts
def acid_smv_scatter_Japan(df_sake):
    # setup figure
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.xlim(-30, 30)
    plt.ylim(0.5, 2.5)

    # make plot
    plt.scatter(
        df_sake['gravity_avg'],
        df_sake['acidity_avg'],
        alpha=0.1)

    # plot median value
    plt.plot(
        df_sake['gravity_avg'].median(),
        df_sake['acidity_avg'].median(),
        marker='x',
        label='Median',
        markersize=12,
        markeredgewidth=3,
        markeredgecolor='red',
        alpha=0.8
    )
    plt.legend(loc='lower right')

    # add titles
    plt.title('Flavor Profiles of Bottles of Sake (all Japan)')#, fontsize=18)
    plt.ylabel('(light)                    Acidity                    (rich)')
    plt.xlabel('(sweet)                    Sake Meter Value                    (dry)')

    # create fill colors
    x = np.arange(-35, 35, 0.5)
    y1 = (0.5/22) * x + 1.65
    y2 = (-0.5/10) * x + 1.25

    ax.set_facecolor('#eafff5')
    plt.fill_between(x, y1, y2, where=y1 > y2, alpha=0.25, interpolate=True)
    plt.fill_between(x, y1, y2, where=y1 < y2, alpha=0.25, interpolate=True)
    plt.fill_between(x, y1, np.max(y2), alpha=0.25)
    plt.plot(x, y1, color='white')
    plt.plot(x, y2, color='white')

    # add all text
    plt.text(-8, 2.3, 'Dry-Rich', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))
    plt.text(-28, 1.6, 'Sweet-Rich', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))
    plt.text(-15, 0.75, 'Sweet-Light', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))
    plt.text(15, 1.25, 'Dry-Light', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))

    st.pyplot(fig)

def flavor_stacked_area(df_flavor_area):
    # remove underscores
    columns = {name: name.replace("_", " ") for name in df_flavor_area.columns}
    regions = {name: name.replace("_", " ") for name in df_flavor_area.index}
    df_flavor_area = df_flavor_area.rename(columns, axis=1)
    df_flavor_area = df_flavor_area.rename(regions, axis=0)

    # make stacked bar plot
    df_flavor_area[['dry light ratio', 'dry rich ratio', 'sweet light ratio', 'sweet rich ratio']].plot(
        kind='barh',
        stacked=True,
        figsize=(6,4)
    ).legend(loc='lower left')

    # add labels
    plt.title("Ratio of Flavor Profile Categories by Area")
    plt.xlabel("% of sake")
    plt.ylabel("area")
    fig = plt.gcf()

    st.pyplot(fig)

def make_smv_acid_matrix():
    # figure
    fig, ax = plt.subplots(figsize=(4,4))
    plt.plot()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(color='w')
    plt.axvline(color='w')

    # ticks
    plt.xticks([-0.5, 0.5])
    plt.yticks([-0.5, 0.5])
    ax.set_xticklabels(['low smv', 'high smv'])
    ax.set_yticklabels(['low acidity', 'high acidity'])

    # text
    # plt.xlabel('Sake Meter Value')
    # plt.ylabel('Acidity')
    plt.text(-0.5, 0.5, "Sweet-Rich", horizontalalignment='center', verticalalignment='center')
    plt.text(0.5, 0.5, "Dry-Rich", horizontalalignment='center', verticalalignment='center')
    plt.text(-0.5, -0.5, "Sweet-Light", horizontalalignment='center', verticalalignment='center')
    plt.text(0.5, -0.5, "Dry-Light", horizontalalignment='center', verticalalignment='center')
    # plt.title("Acidity / SMV Confusion Matrix")

    # color
    alpha=0.9
    ax.fill_between([-1, 0], -1, 0, alpha=alpha, color='#e9fdf4')  # sweet-light (light blue)
    ax.fill_between([-1, 0], 0, 1, alpha=alpha, color='#c1b69c')  # sweet-rich (brown)
    ax.fill_between([0, 1], -1, 0, alpha=alpha, color='#efdebc')  # dry_light (orange)
    ax.fill_between([0, 1], 0, 1, alpha=alpha, color='#e4c8c0')  # dry_rich (pink)

    st.pyplot(fig)

def skewed_smv_acid_matrix():
    # setup figure
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.xlim(-30, 30)
    plt.ylim(0.5, 2.5)

    # scatter plot that won't appear (for some reason, not using a scatter plot changes the colors)
    plt.scatter(0, 5)

    # add text
    plt.ylabel('(light)                    Acidity                    (rich)')
    plt.xlabel('(sweet)                    Sake Meter Value                    (dry)')

    # create fill colors
    x = np.arange(-35, 35, 0.5)
    y1 = (0.5/22) * x + 1.65
    y2 = (-0.5/10) * x + 1.25

    ax.set_facecolor('#eafff5')
    plt.fill_between(x, y1, y2, where=y1 > y2, alpha=0.25, interpolate=True)
    plt.fill_between(x, y1, y2, where=y1 < y2, alpha=0.25, interpolate=True)
    plt.fill_between(x, y1, np.max(y2), alpha=0.25)
    plt.plot(x, y1, color='white')
    plt.plot(x, y2, color='white')

    # add all text
    plt.text(-8, 2.3, 'Dry-Rich', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))
    plt.text(-28, 1.6, 'Sweet-Rich', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))
    plt.text(-15, 0.75, 'Sweet-Light', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))
    plt.text(15, 1.25, 'Dry-Light', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))

    st.pyplot(fig)

def make_straight_vs_skewed():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # saving lines
    x = np.arange(-35, 35, 0.5)
    y1 = (0.5/22) * x + 1.65
    y2 = (-0.5/10) * x + 1.25

    m = (0.5/22) + (0.5/10)
    b = 1.25 - 1.65
    x_int = b / m
    y_int = (0.5/22) * x_int + 1.65

    # straight plot
    axs[0].plot()
    axs[0].set_xlim(-30, 30)
    axs[0].set_ylim(0.5, 2.5)
    axs[0].set_title("Straight Plot", fontsize=16)

    axs[0].axhline(y=y_int, color='w')
    axs[0].axvline(x=x_int, color='w')

    blank1 = 16
    axs[0].set_ylabel(f'(light){" "*blank1}Acidity{" "*blank1}(rich)')
    axs[0].set_xlabel(f'(sweet){" "*blank1}Sake Meter Value{" "*blank1}(dry)')

    alpha=0.9
    axs[0].fill_between([-30, x_int], 0.5, y_int, alpha=alpha, color='#e9fdf4') # sweet-light (light blue)
    axs[0].fill_between([-30, x_int], y_int, 2.5, alpha=alpha, color='#c1b69c') # sweet-rich (brown)
    axs[0].fill_between([x_int, 30], 0.5, y_int, alpha=alpha, color='#efdebc') # dry_light (orange)
    axs[0].fill_between([x_int, 30], y_int, 2.5, alpha=alpha, color='#e4c8c0') # dry_rich (pink)

    axs[0].text((30 + x_int) / 2, (2.5 + y_int) / 2, 'Dry-Rich', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2), horizontalalignment='center', verticalalignment='center')
    axs[0].text((-30 + x_int) / 2, (2.5 + y_int) / 2, 'Sweet-Rich', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2), horizontalalignment='center', verticalalignment='center')
    axs[0].text((-30 + x_int) / 2, (0.5 + y_int) / 2, 'Sweet-Light', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2), horizontalalignment='center', verticalalignment='center')
    axs[0].text((30 + x_int) / 2, (0.5 + y_int) / 2, 'Dry-Light', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2), horizontalalignment='center', verticalalignment='center')

    # skewed plot
    axs[1].scatter(0, 5)
    axs[1].set_xlim(-30, 30)
    axs[1].set_ylim(0.5, 2.5)
    axs[1].set_title("Skewed Plot", fontsize=16)

    blank2 = 20
    axs[1].set_ylabel(f'(light){" "*blank2}Acidity{" "*blank2}(rich)')
    axs[1].set_xlabel(f'(sweet){" "*blank2}Sake Meter Value{" "*blank2}(dry)')

    axs[1].set_facecolor('#eafff5')
    axs[1].fill_between(x, y1, y2, where=y1 > y2, alpha=0.25, interpolate=True)
    axs[1].fill_between(x, y1, y2, where=y1 < y2, alpha=0.25, interpolate=True)
    axs[1].fill_between(x, y1, np.max(y2), alpha=0.25)
    axs[1].plot(x, y1, color='white')
    axs[1].plot(x, y2, color='white')

    axs[1].text(-8, 2.3, 'Dry-Rich', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))
    axs[1].text(-28, 1.6, 'Sweet-Rich', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))
    axs[1].text(-15, 0.75, 'Sweet-Light', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))
    axs[1].text(15, 1.25, 'Dry-Light', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))

    # plot two sake bottles
    smv = -12
    acid_1 = 1.65
    acid_2 = acid_1 + 0.5

    for i in range(2):
        axs[i].plot(
            smv,
            acid_1,
            marker='x',
            label='sake 1',
            markersize=15,
            markeredgewidth=3,
            markeredgecolor='red',
            alpha=0.8
        )
        axs[i].plot(
            smv,
            acid_2,
            marker='x',
            label='sake 2',
            markersize=15,
            markeredgewidth=3,
            markeredgecolor='blue',
            alpha=0.8
        )
        # axs[i].legend(loc='upper right')

    st.pyplot(fig)

def smv_acid_all_regions():
    # saving variables
    x = np.arange(-35, 35, 0.5)
    y1 = (0.5/22) * x + 1.65
    y2 = (-0.5/10) * x + 1.25
    regions = ['Hokkaidō', 'Tōhoku', 'Chūbu', 'Kantō', 'Kansai', 'Shikoku', 'Chūgoku', 'Kyūshū']

    # setup figure
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(16, 24))
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle("Acidity and Sake Meter Value by Region", fontsize=22, y=0.92)

    for region, ax in zip(regions, axs.ravel()):
        # plot acidity and smv
        ax.scatter(
            df_sake[df_sake['region'] == region]['gravity_avg'],
            df_sake[df_sake['region'] == region]['acidity_avg'],
            alpha=0.15
        )

        # plot region's median value
        ax.plot(
            df_sake[df_sake['region'] == region]['gravity_avg'].median(),
            df_sake[df_sake['region'] == region]['acidity_avg'].median(),
            marker='x',
            label=f"{region}'s median",
            markersize=18,
            markeredgewidth=3,
            markeredgecolor='k',
            alpha=0.8
        )

        # plot overall median value
        ax.plot(
            df_sake['gravity_avg'].median(),
            df_sake['acidity_avg'].median(),
            marker='x',
            label='Overall median',
            markersize=18,
            markeredgewidth=3,
            markeredgecolor='red',
            alpha=0.8
        )
        ax.legend(loc='lower right')

        # set limit
        ax.set_xlim(-30, 30)
        ax.set_ylim(0.5, 2.5)

        # add titles
        ax.set_title(region, fontsize=22)
        ax.set_ylabel('(light)                    Acidity                    (rich)')
        ax.set_xlabel('(sweet)                    Sake Meter Value                    (dry)')

        # create fill colors
        ax.set_facecolor('#eafff5')
        ax.fill_between(x, y1, y2, where=y1 > y2, alpha=0.25, interpolate=True)
        ax.fill_between(x, y1, y2, where=y1 < y2, alpha=0.25, interpolate=True)
        ax.fill_between(x, y1, np.max(y2), alpha=0.25)
        ax.plot(x, y1, color='white')
        ax.plot(x, y2, color='white')

        # add all text
        ax.text(-8, 2.3, 'Dry-Rich', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))
        ax.text(-28, 1.6, 'Sweet-Rich', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))
        ax.text(-15, 0.75, 'Sweet-Light', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))
        ax.text(15, 1.25, 'Dry-Light', fontsize=14, bbox = dict(facecolor = 'red', alpha = 0.2))

    st.pyplot(fig)

def flavor_stacked_region(df_flavor):
    # remove underscores
    columns = {name: name.replace("_", " ") for name in df_flavor.columns}
    regions = {name: name.replace("_", " ") for name in df_flavor.index}
    df_flavor = df_flavor.rename(columns, axis=1)
    df_flavor = df_flavor.rename(regions, axis=0)

    # sort regions
    regions = ['Kyūshū', 'Chūgoku', 'Shikoku', 'Kansai', 'Kantō', 'Chūbu', 'Tōhoku', 'Hokkaidō']
    r_mapping = {region: i for i, region in enumerate(regions)}
    r_key = df_flavor.index.map(r_mapping)

    # make stacked bar plot
    df_flavor.iloc[r_key.argsort()][['dry light ratio', 'dry rich ratio', 'sweet light ratio', 'sweet rich ratio']].plot(
        kind='barh',
        stacked=True,
        figsize=(6,4),
        legend=True
    ).legend(loc='lower left')

    # add labels
    plt.title("Ratio of Flavor Profile Categories by Region")
    plt.xlabel("% of sake")
    plt.ylabel("area")

    fig = plt.gcf()

    st.pyplot(fig)

def flavor_stacked_region_2(df_flavor):
    # remove underscores
    columns = {name: name.replace("_", " ") for name in df_flavor.columns}
    regions = {name: name.replace("_", " ") for name in df_flavor.index}
    df_flavor = df_flavor.rename(columns, axis=1)
    df_flavor = df_flavor.rename(regions, axis=0)

    # sort regions
    regions = ['Kyūshū', 'Chūgoku', 'Shikoku', 'Kansai', 'Kantō', 'Chūbu', 'Tōhoku', 'Hokkaidō']
    r_mapping = {region: i for i, region in enumerate(regions)}
    r_key = df_flavor.index.map(r_mapping)

    # make stacked bar plot
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    df_flavor.iloc[r_key.argsort()][['dry ratio', 'sweet ratio']].mul(100).plot(
        kind='barh',
        stacked=True,
        legend=True,
        fontsize=16,
        ax=axs[0]
    )#.legend(loc='lower left', fontsize=16)


    df_flavor.iloc[r_key.argsort()][['light ratio', 'rich ratio']].mul(100).plot(
        kind='barh',
        stacked=True,
        legend=True,
        fontsize=16,
        ax=axs[1]
    )#.legend(loc='lower left', fontsize=16)

    # calculate overall dry and light ratios
    total = df_sake[['dry_light', 'dry_rich', 'sweet_light', 'sweet_rich']].sum().sum()
    dry_total = df_sake[['dry_light', 'dry_rich']].sum().sum()
    light_total = df_sake[['dry_light', 'sweet_light']].sum().sum()
    dry_rat = dry_total / total * 100
    light_rat = light_total / total * 100

    # plotting overal dry and light ratios
    axs[0].axvline(dry_rat, label=f'all sake: {int(dry_rat)}%', color='r', linestyle='dashed')
    axs[1].axvline(light_rat, label=f'all sake: {int(light_rat)}%', color='r', linestyle='dashed')

    # incorrect median lines! - not median, but ratio
    # dry_rat = df_flavor['dry ratio'].mul(100).median()
    # light_rat = df_flavor['light ratio'].mul(100).median()
    # axs[0].axvline(dry_rat, label=f'median ({int(dry_rat)}%)', color='r', linestyle='dashed')
    # axs[1].axvline(light_rat, label=f'median ({int(light_rat)}%)', color='r', linestyle='dashed')

    # legends
    axs[0].legend(loc='lower left', fontsize=16)
    axs[1].legend(loc='lower left', fontsize=16)

    # adding text
    axs[0].set_title("Dry : Sweet Ratio by Region", fontsize=20)
    axs[0].set_xlabel("% of sake", fontsize=16)
    axs[0].set_ylabel(f"(west){' '*20}region{' '*20}(north)", fontsize=16)
    axs[1].set_title("Light : Rich Ratio by Region", fontsize=20)
    axs[1].set_xlabel("% of sake", fontsize=16)
    axs[1].set_ylabel("")

    plt.subplots_adjust(wspace=0.3)

    st.pyplot(fig)

def num_of_sake_bottles_asmv(df_flavor, df_flavor_area):
    # putting regions in the right order
    regions = ['Kyūshū', 'Chūgoku', 'Shikoku', 'Kansai', 'Kantō', 'Chūbu', 'Tōhoku', 'Hokkaidō']
    r_mapping = {region: i for i, region in enumerate(regions)}
    r_key = df_flavor.index.map(r_mapping)

    # remove underscores
    areas = {name: name.replace("_", " ") for name in df_flavor_area.index}
    df_flavor_area = df_flavor_area.rename(areas, axis=0)

    # putting areas in the right order
    areas = ['West', 'West Central', 'Central', 'North']
    a_mapping = {area: i for i, area in enumerate(areas)}
    a_key = df_flavor_area.index.map(a_mapping)

    # charts
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    xlim = (0, 1750)
    xlabel = 'number of sake bottles'

    r_colors = cm.winter(np.linspace(0,1,len(df_flavor)))
    df_flavor.iloc[r_key.argsort()].total.plot(kind='barh', legend=False, ax=axs[0], xlim=xlim, xlabel='', color=r_colors)
    axs[0].set_title('Number of Sake Bottles with Acidity and SMV info (by Region)')
    axs[0].set_xlabel(xlabel)

    a_colors = cm.winter(np.linspace(0,1,len(df_flavor_area)))
    df_flavor_area.iloc[a_key.argsort()].total.plot(kind='barh', legend=False, ax=axs[1], xlim=xlim, xlabel='', color=a_colors)
    axs[1].set_title('Number of Sake Bottles with Acidity and SMV info (by Area)')
    axs[1].set_xlabel(xlabel)

    plt.subplots_adjust(hspace=0.3)

    st.pyplot(fig)

def flavor_bar_area_charts(df_flavor_area):
    # remove underscores
    columns = {name: name.replace("_", " ") for name in df_flavor_area.columns}
    areas = {name: name.replace("_", " ") for name in df_flavor_area.index}
    df_flavor_area = df_flavor_area.rename(columns, axis=1)
    df_flavor_area = df_flavor_area.rename(areas, axis=0)

    # make bar plots
    fig, axs = plt.subplots(1, 4, figsize=(10,2.5), sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.15)
    fig.suptitle("Ratio of Flavor Profile Categories (by Area)", fontsize=16, y=1.1)
    col = ['dry light ratio', 'dry rich ratio', 'sweet light ratio', 'sweet rich ratio']

    for area, ax in zip(list(areas.values()), axs.ravel()):
        bars = ax.barh(
            col,
            df_flavor_area.loc[area, col].mul(100)
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=1.5)
        ax.set_xlim(0, 100)
        ax.set_title(f"{area}")
        ax.set_xlabel("% of sake")

    st.pyplot(fig)

def pval_df_one_sample(df_flavor_area):

    # copy the dataframe
    df_fa = df_flavor_area.copy()

    # get necessary numbers
    west_total = df_fa.loc["West", "total"]
    north_total = df_fa.loc["North", "total"]

    west_dry = df_fa.loc["West", "dry"]
    north_dry = df_fa.loc["North", "dry"]
    west_light = df_fa.loc["West", "light"]
    north_light = df_fa.loc["North", "light"]

    west_dry_dist = [1] * west_dry + [0] * (west_total-west_dry)
    north_dry_dist = [1] * north_dry + [0] * (north_total-north_dry)
    west_light_dist = [1] * west_light + [0] * (west_total-west_light)
    north_light_dist = [1] * north_light + [0] * (north_total-north_light)

    all_dry_ratio = df_fa.loc[:, "dry"].sum() / df_fa.loc[:, "total"].sum()
    all_light_ratio = df_fa.loc[:, "light"].sum() / df_fa.loc[:, "total"].sum()

    # create dictionary to loop through
    dist_dict = {
        'west dry distribution': [west_dry_dist, all_dry_ratio],
        'north dry distribution': [north_dry_dist, all_dry_ratio],
        'west light distribution': [west_light_dist, all_light_ratio],
        'north light distribution': [north_light_dist, all_light_ratio]
    }

    # create empty dataframe to put values into
    pval_df = pd.DataFrame(columns=['p-value'], index=dist_dict.keys())


    # get p-values
    for k, v in dist_dict.items():
        pval = stats.ttest_1samp(v[0], v[1])[1]
        pval_df.loc[k, 'p-value'] = round(pval, 6)

    # conditional formatting for styling df
    def cond_formatting(x):
        if x < 0.05:  # significance threshold
            return 'background-color: lightgreen'
        else:
            return None
    pval_df = pval_df.style.applymap(cond_formatting)

    # display df
    st.dataframe(pval_df)

def pval_df_two_sample(df_flavor_area):

    # copy the dataframe
    df_fa = df_flavor_area.copy()

    # get necessary numbers
    west_total = df_fa.loc["West", "total"]
    north_total = df_fa.loc["North", "total"]

    west_dry = df_fa.loc["West", "dry"]
    north_dry = df_fa.loc["North", "dry"]
    west_light = df_fa.loc["West", "light"]
    north_light = df_fa.loc["North", "light"]

    west_dry_dist = [1] * west_dry + [0] * (west_total-west_dry)
    north_dry_dist = [1] * north_dry + [0] * (north_total-north_dry)
    west_light_dist = [1] * west_light + [0] * (west_total-west_light)
    north_light_dist = [1] * north_light + [0] * (north_total-north_light)

    # create dictionary to loop through
    dist_dict_2 = {
        'west vs. north dry distributions': [west_dry_dist, north_dry_dist],
        'west vs. north light distributions': [west_light_dist, north_light_dist],
    }

    # create empty dataframe to put values into
    pval_df_2 = pd.DataFrame(columns=['p-value'], index=dist_dict_2.keys())

    # get p-values
    for k, v in dist_dict_2.items():
        pval = stats.ttest_ind(v[0], v[1])[1]
        pval_df_2.loc[k, 'p-value'] = round(pval, 6)

    # conditional formatting for styling df
    def cond_formatting(x):
        if x < 0.05:  # significance threshold
            return 'background-color: lightgreen'
        else:
            return None
    pval_df_2 = pval_df_2.style.applymap(cond_formatting)

    # display the dataframe
    st.dataframe(pval_df_2)

# map variables
tile_light_gray = 'https://server.arcgisonline.com/arcgis/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
attr_light_gray = 'Esri, HERE, Garmin, (c) OpenStreetMap contributors, and the GIS user community'
test_coord = [39, 140]
region_locs = {
    'Kyūshū': [32.5, 125.2],
    'Chūgoku': [36.5, 130.5],
    'Shikoku': [33.3, 133.0],
    'Kansai': [34.3, 136.2],
    'Chūbu': [38.3, 135.8],
    'Kantō': [36.5, 141.0],
    'Tōhoku': [39.5, 142.2],
    'Hokkaidō': [42.8, 144.2]
}
area_locs = {
    'West': [32.5, 125.5],
    'West_Central': [33.3, 133.5],
    'Central': [36.0, 141.0],
    'North': [40.8, 142.2],
}

# map functions
def dry_map_region():
    dry_bins = [.80, .87, .895, .92, .95, .98, 1.00]
    dict_reg_dry_ratio = df_flavor['dry_ratio'].to_dict()

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
        columns=["Region", "region_dry_ratio"],
        key_on="properties.REGION",
        bins=dry_bins,
        fill_color="BuPu",
        fill_opacity=0.7,
        line_opacity=0.2,
    #     legend_name="% of sake with dry flavor profile",
        highlight=True
    )#.add_to(m)

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
                '''.format("Percent of sake with a dry flavor profile (as opposed to sweet)")

    m.get_root().html.add_child(folium.Element(title_html))

    # add labels to the map
    for region, loc in region_locs.items():
        folium.map.Marker(
            loc,
            icon=DivIcon(
                icon_size=(250,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 14pt">{region}: {int(round(dict_reg_dry_ratio[region], 2) * 100)}%</div>',
                )
            ).add_to(m)

    return m

def light_map_region():
    light_bins = [0.73, 0.77, 0.79, 0.82, 0.85, 0.88, 0.92]
    dict_reg_light_ratio = df_flavor['light_ratio'].to_dict()

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
        columns=["Region", "region_light_ratio"],
        key_on="properties.REGION",
        bins=light_bins,
        fill_color="BuPu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="% of sake with light flavor profile",
        highlight=True
    )#.add_to(m)

    # removing legend
    for key in cp._children:
        if key.startswith("color_map"):
            del(cp._children[key])

    cp.add_to(m)

    folium.LayerControl().add_to(m)

    # and finally adding a tooltip/hover to the choropleth's geojson (match name with field name in the previous line)
    folium.GeoJsonTooltip(['REGION', 'light:rich ratio (region)']).add_to(cp.geojson)

    # add title
    title_html = '''
                <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                '''.format("Percent of sake with a light flavor profile (as opposed to rich)")

    m.get_root().html.add_child(folium.Element(title_html))

    # add labels to the map
    for region, loc in region_locs.items():
        folium.map.Marker(
            loc,
            icon=DivIcon(
                icon_size=(250,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 14pt">{region}: {int(round(dict_reg_light_ratio[region], 2) * 100)}%</div>',
                )
            ).add_to(m)

    return m

def dry_map_area():
    dry_bins = [.80, .87, .895, .92, .95, .98, 1.00]
    dict_area_dry_ratio = df_flavor_area['dry_ratio'].to_dict()

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
        columns=["Area", "area_dry_ratio"],
        key_on="properties.AREA",
        bins=dry_bins,
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=0.2,
    #     legend_name="dry:sweet ratio",
        highlight=True
    )#.add_to(m)

    # removing legend
    for key in cp._children:
        if key.startswith("color_map"):
            del(cp._children[key])

    cp.add_to(m)

    folium.LayerControl().add_to(m)

    # and finally adding a tooltip/hover to the choropleth's geojson (match name with field name in the previous line)
    folium.GeoJsonTooltip(['PREFECTURE']).add_to(cp.geojson)

    # add title - not working with I guess either folium static or streamlit, for some reason ...
    title_html = '''
                <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                '''.format("Percent of sake with a dry flavor profile (by area)")

    m.get_root().html.add_child(folium.Element(title_html))

    # add labels to the map
    for area, loc in area_locs.items():
        folium.map.Marker(
            loc,
            icon=DivIcon(
                icon_size=(250,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 14pt">{area.replace("_", " ")}: {int(round(dict_area_dry_ratio[area], 2)*100)}%</div>',
                )
            ).add_to(m)

    return m

def light_map_area():
    light_bins = [0.73, 0.77, 0.79, 0.82, 0.85, 0.88, 0.92]
    dict_area_light_ratio = df_flavor_area['light_ratio'].to_dict()

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
        columns=["Area", "area_light_ratio"],
        key_on="properties.AREA",
        bins=light_bins,
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=0.2,
    #     legend_name="light:rich ratio",
        highlight=True
    )#.add_to(m)

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
                '''.format("Percent of sake with a light flavor profile (by area)")

    m.get_root().html.add_child(folium.Element(title_html))

    # add labels to the map
    for area, loc in area_locs.items():
        folium.map.Marker(
            loc,
            icon=DivIcon(
                icon_size=(250,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 14pt">{area.replace("_", " ")}: {int(round(dict_area_light_ratio[area], 2)*100)}%</div>',
                )
            ).add_to(m)

    return m


# page layout
st.title("Acidity and Sake Meter Value")
st.header("TL;DR")
st.write(
    """
    Although it's true that western Japan has a higher percentage of sweet and rich
    sake, most sake across the board is dry and light.
    """
)

acid_smv_scatter_Japan(df_sake)
flavor_stacked_area(df_flavor_area)

st.markdown("""---""")
st.header("Pt 1: Chemistry -> Flavor")
st.write(
    """
    After sake is made, two chemical measurments are commonly taken:
    - **Acidity (酸度)**: How acidic the sake is.  Unsurprisingly, higher acidity = more acidic.
    - **Sake Meter Value (SMV) (日本酒度)**: Basically, how much sugar is in the sake.  Surprisingly, higher SMV = less sugar.

    These values correspond with two flavors:
    - **Acidity = Richness**.  Low acidity means *light* sake, and high acidity means *rich* sake.
    - **SMV = Sweetness**.  Low SMV means *sweet* sake, and high SMV means *dry* sake.
    """
)

make_smv_acid_matrix()

st.write(
    """
    In other words, depending on the acidity and SMV of a sake, its flavor profile can be categorized in
    one of four ways: Sweet-Light, Sweet-Rich, Dry-Light, or Dry-Rich.  Acidity usually ranges
    between 1.0 and 2.0, and SMV between -5 and 10.  Categorizing should be pretty straightforward, right?

    *Not so fast!*  It's not quite so simple, because higher acidity makes sake taste drier.  If we plot
    the above matrix, it needs to be skewed to account for this\*:
    """
)

skewed_smv_acid_matrix()

st.write(
    """
    To better illustrate this, notice the blue and red X's in the two charts below.  These
    X's represent two bottles of sake that have the same SMV, but the blue X has a higher
    acidity.  In the straight (ie incorrect) plot on the left, they fall into the same category
    (Sweet-Rich).  In the skewed (ie correct) plot on the left, they are in two different
    categories:
    """
)

make_straight_vs_skewed()

st.write(
    """
    The blue X, having higher acidity, should actually be classified as Dry instead of Sweet, like
    in the chart on the right.  Remember that a low SMV equates to being sweeter (sweet and low!).
    Although these bottles of sake have a rather low SMV of -12, the high acidity of the blue X
    counteracts the sweetness, making it actually more Dry than Sweet.

    This highlights a bigger point about classifying sake - it's hard to judge the flavor of a
    bottle of sake based on just one criteria.  If I saw a bottle of sake with an SMV of -12,
    without knowing anything else about it, I'd guess it's pretty sweet.  But as we just saw,
    it's not necessarily so!\*\*
    """
)

st.header("Pt 2: Flavor Profile Categories by Region")

st.write(
    """
    So does sake tend to be more "Dry-Light" in northern Japan (Hokkaido, Tohoku) and
    "Sweet-Rich" in Western Japan (Kyushu, Chugoku)?  Let's see what percentage of sake
    is Dry (left chart) and Light (right chart) in each region:
    """
)

# flavor_stacked_region(df_flavor)
flavor_stacked_region_2(df_flavor)

st.write(
    """
    There's a very slight trend in the left chart, with about 96% of sake being dry
    in Hokkaido, going down to a low 88% in Kyushu.  6 of the 8 regions are
    within + / - 2% of the 90% median.

    In the chart on the right, there's a lot more variety and even less of a discernable
    trend.  Hokkaido has the highest percentage of light sake at 90%, but this time
    Kansai has the lowest rate with 73% (Kyushu is second lowest at 75%).

    Let's see what mapping the same data looks like.
    """
)

m1 = dry_map_region()
st.markdown("**Percent of sake with a dry flavor profile (as opposed to sweet):**")
folium_static(m1)

m2 = light_map_region()
st.markdown("**Percent of sake with a light flavor profile (as opposed to rich):**")
folium_static(m2)

st.write(
    """
    In the 2nd map, it's worth noting that 5 of the 8 regions are below 80%, and the
    remaining three (Hokkaido, Kanto, and Shikoku) have the least amount of data in the data set.
    Let's try grouping regions together to better balance the data and see if there is
    a clearer trend across Japan.
    """
)

num_of_sake_bottles_asmv(df_flavor, df_flavor_area)

# st.write(
#     """
#     Western Japan's median acidity is a bit higher than the rest of Japan, but the
#     SMV is not so different, with Kyushu being a bit lower than other regions (ie sweeter).

#     The other thing to note is Shikoku's SMV is a bit higher (ie drier) than the rest of Japan.
#     If conventional wisdom says that western Japan has more Sweet sake, Shikoku
#     would be the exception to that.

#     The other thing to note is that there isn't actually a huge amount of variety.  The black
#     and red X's are super close together (or overlapping) for each chart.  Some of the clusters
#     of blue dots are more oval shaped (Tohoku), and some are rounder (Kansai), but nothing
#     stands out as being dramatically different.
#     """
# )



st.header("Pt 3: Grouping regions into areas")

m3 = dry_map_area()
st.markdown("**Percent of sake with a dry flavor profile (by area):**")
folium_static(m3)

m4 = light_map_area()
st.markdown("**Percent of sake with a light flavor profile (by area):**")
folium_static(m4)

st.write(
    """
    Hey, there's kind of a trend!  Especially if you just compare North to West.
    Imagine visiting a hypothetical sake shop with sections for each area of Japan,
    each section having exactly 100 different bottles to choose from. The northern sake
    section of the shop should have 2% more dry sake than the western section, and 3% more
    light sake. Neat!

    However, the vast majority of sake in this shop would be Dry, especially Dry-Light.
    You'd need to search around a bit to find any Sweet-Rich sake, even in the West section:
    """
)

flavor_bar_area_charts(df_flavor_area)

st.subheader("Statistical Significance")

st.write(
    """
    This begs the question: Is any of this variation between areas significant?  Or is the
    variation just random fluctuations?

    Running a T-test to calculate the p-value of dry / light sake by region can tell us
    whether or not we can reject the null hypothesis that it's all just random variation!
    """
)

pval_df_one_sample(df_flavor_area)

st.write(
    """
    The above chart shows the p-values for north and west areas.  It compares their distributions
    of dry and light sake compared to the mean dry / light ratios for all of Japan.  The only
    value that is below the significance threshold of 0.05 is West's distribution of dry sake.
    Therefore, we can reject the null hypothesis that there is no significant difference in
    the amount of dry sake in Western Japan compared to the rest of Japan (we know from before that
    it is less than the rest of Japan).  However, we can't say that for the other distributions
    of sake.  Oh well!

    What about comparing West and North Japan to each other, instead of to all of Japan?
    Let's just run a two sample T-test and find out:
    """
)

pval_df_two_sample(df_flavor_area)

st.write(
    """
    This time, the difference in dry sake is not significant, but the difference in their
    distributions of light sake IS signficant!

    So if you wanted to make some statistical statements about dry and light sake in North and West Japan,
    you could say two things:
    """
)

st.markdown(
    """
    1. West Japan's distribution of dry sake is significantly less than the rest of Japan.
    2. West Japan's distribution of light sake is also significantly less than that of\
    North Japan, but not when compared to all of Japan.
    """
)

st.header("Conclusion")

st.write(
    """
    Sake's flavor is pretty homogenous throughout Japan.  Based on Acidity and Sake Meter Value, most
    sake has a dry-light flavor profile no matter where you are.  The one exception appears to be West
    Japan, which has a significantly lower ratio of dry:sweet sake than the rest of Japan.  Furthermore,
    compared to North Japan, West Japan has a significantly lower ratio of light:rich sake (but not
    when compared to all of Japan).
    """
)

# st.write(
#     """
#     Perhaps this lack of sweet sake is why there is actually a special type
#     of extra sweet sake, but no special extra dry sake.  sake that is extra sweet, called Kijoshu.  However, as far as I know, there is no special
#     category for extra dry sake, because having a sweet sake is so much rarer.

#     (By the way, Kijoshu is kind of like fortified sake.  As a point of reference, the
#     median SMV for all sake in this dataset is +3.0, but for kijoshu it's -47.0!)
#     """
# )

st.markdown("""---""")
st.subheader("Notes")
st.markdown("""
<style>
.note-font {
    font-size:16px;
}
</style>
""", unsafe_allow_html=True)

note = """
*I've seen this chart parroted around the internet several times, but haven't been able
to find a source for it.  Please get in touch if you know what the official name for this
chart is, or where it originated!  I'd love to find out.

**There's a calculation called Amakarado which is a combination of SMV and Acidity.  It's
supposed to give a better indication of the sweetness of a sake than SMV alone.  Also, it
is much more intuitive than SMV in that a high Amakarado means sweeter sake, and vice versa.
For a bit of context, in the straight / skewed plot comparisons above, the Amakarado of the red X is about 0.8, and
the blue X is 0.2.  Amakarado is a pretty useful concept, but unfortunately is almost never used.
"""
st.markdown(f'<p class="note-font">{note}</p>', unsafe_allow_html=True)
