import os
import pandas as pd
import re
import requests
import time
from bs4 import BeautifulSoup as bs

def sake_scraper(start_num: int, stop_num: int, csv_directory=None):
    """
    Web Scraper for nihonshu.wiki. Writes scraped data as csv to local disk.

    Parameters:
    -----------
    start_num       (int) : web page number to start scraping from
    stop_num        (int) : web page number to stop scraping
    csv_directory   (str) : path to local directory for writing the csv file
    """
    sake_dict = {"name": [], "name_kana": [], "name_romaji": [], "sake_type": [], "rice_type": [], "rice_origin": [],
                 "yeast": [], "rice_polishing_rate": [], "abv": [], "acidity": [], "amino": [], "gravity": [],
                 "volume": [], "prefecture": [], "city": [], "company": [], "address": [], "website": [], "brand": [],
                 "categories": [], "id": []
                }
    url = "https://www.nihonshu.wiki/products/detail/"
    blank_page_counter = 0

    # go through each product webpage on nihonshu.wiki
    for i in range(start_num, stop_num + 1):
        time.sleep(3)  # wait 3 seconds to avoid overwhelming the website
        response = requests.get(url+str(i))
        soup = bs(response.content, 'html.parser')

        # add product information to the sake_dict
        main = soup.find(id='item_detail_area')
        if main == None:
            blank_page_counter += 1
            continue
        sake_dict["name"].append(re.sub(r"\u3000", " ", main.find(id='detail_description_box__name').string))
        sake_dict["name_kana"].append(main.find(id='detail_description_box__kana').string)
        sake_dict["name_romaji"].append(main.find(id='detail_description_box__kana_romaji').string)
        sake_dict["sake_type"].append(main.find("dt", string="特定名称・分類").find_next_sibling("dd").text.strip("\n\r "))

        # rice type and origin - will include multiple types of rice
        rice_tag = main.find("dt", string="主使用米").find_next_sibling("dd").text.strip("\n\r ")
        if rice_tag == "-":
            sake_dict["rice_type"].append("-")
            sake_dict["rice_origin"].append("-")
        else:
            sake_dict["rice_type"].append(" / ".join([val.string.strip("\n\r ") for val in soup.find(id='item_detail_area').find("dt", string="主使用米").find_next_sibling("dd").select("a")]))
            try:
                sake_dict["rice_origin"].append(" / ".join([val.select("dd")[0].text.strip('\r\n ') for val in soup.find(id='item_detail_area').find("dt", string="主使用米").find_next_sibling("dd").select("div")]))
            except IndexError:
                sake_dict["rice_origin"].append("-")

        sake_dict["yeast"].append(main.find("dt", string="使用酵母").find_next_sibling("dd").text.strip("\n\r "))
        sake_dict["rice_polishing_rate"].append(main.find("dt", string="精米歩合").find_next_sibling("dd").text.strip("%\n\r "))
        sake_dict["abv"].append(re.sub(r"[  *\n\r]", "", main.find("dt", string="アルコール度数").find_next_sibling("dd").text.strip("度\n\r ")))
        sake_dict["acidity"].append(main.find("dt", string="酸度").find_next_sibling("dd").text.strip("\n\r "))
        sake_dict["amino"].append(main.find("dt", string="アミノ酸度").find_next_sibling("dd").text.strip("\n\r "))
        sake_dict["gravity"].append(main.find("dt", string="日本酒度").find_next_sibling("dd").text.strip("\n\r "))
        try:
            sake_dict["volume"].append(re.sub(r"[  *\n\r]", "", main.find("dt", string="容量").find_next_sibling("dd").text.strip("\n\r ")))
        except AttributeError:
            sake_dict["volume"].append("-")
        if main.find("dt", string="蔵元").find_next_sibling("div").text.strip("\n\r ") == "-":
            sake_dict["prefecture"].append("-")
            sake_dict["city"].append("-")
            sake_dict["company"].append("-")
            sake_dict["address"].append("-")
            sake_dict["website"].append("-")
        else:
            sake_dict["prefecture"].append(main.find("dt", string="蔵元").find_next_sibling("div").select("a")[0].text)
            sake_dict["city"].append(main.find("dt", string="蔵元").find_next_sibling("div").select("a")[1].text)
            sake_dict["company"].append(main.find("dt", string="蔵元").find_next_sibling("div").select("a")[2].text)
            sake_dict["address"].append(re.sub(r"[  *\n\r]", "", main.find("dt", string="蔵元").find_next_sibling("div").find("div", id="tips").select("dd")[0].text.strip("\n\r ")))
            sake_dict["website"].append(main.find("dt", string="蔵元").find_next_sibling("div").find("dt", string="WEB").find_next_sibling("dd").text.strip("\n\r "))
        sake_dict["brand"].append(main.find("dt", string="銘柄").find_next_sibling("dd").text.strip("\n\r "))

        # added categories section
        sake_dict["categories"].append(" / ".join([re.sub('\n', '>', val) for val in main.find(id='relative_category_box').text.strip('\n関連カテゴリ').split('\n\n\n')]))

        # added product id from site
        sake_dict["id"].append(i)

        print(f"scraped product {i}")

    print(f"number of blank pages: {blank_page_counter}")

    # save as csv
    sake_df = pd.DataFrame(sake_dict)
    filename = f'sake_df_{start_num}_{stop_num}.csv'
    filepath = os.path.join(csv_directory, filename)
    sake_df.to_csv(filepath, index=False)
    print("csv saved")
