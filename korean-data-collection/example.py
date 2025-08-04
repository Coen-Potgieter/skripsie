from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time
from bs4 import BeautifulSoup
import json
import os
import sys
from datetime import datetime, timedelta


def save_json(content, file):
    json_str = json.dumps(content, indent=4)
    with open(file, mode="w") as json_file:
        json_file.write(json_str)


def load_json(file):
    with open(file, mode="r") as tf:
        return json.load(tf)


def move_file(file2move, location):
    num = len(os.listdir(location))
    os.rename(file2move, location + f"/{num}.html")


def site_extract(link, file_path, login=True, filter_date=None):

    def process_filter_date_offset():

        year = int(filter_date[:4])
        month = int(filter_date[5:7])
        day = int(filter_date[8:11])

        num_days_per_month = [32, 29, 32, 31, 32, 31, 32, 32, 31, 32, 31, 32]

        year_offset = year - 1979
        month_offset = 13 - month
        days_offset = num_days_per_month[month - 1] - day
        return (year_offset, month_offset, days_offset)

    def click_read_more():

        class_ = "x1i10hfl xjbqb8w x1ejq31n xd10rxx x1sy0etr x17r0tee x972fbf xcfux6l x1qhh985 xm0m39n x9f619 x1ypdohk xt0psk2 xe8uvvx xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x16tdsg8 x1hl2dhg xggy1nq x1a2a7pz xt0b8zv xzsf02u x1s688f"

        try:
            buttons = driver.find_elements(
                By.XPATH, f"//div[@class='{class_}'][@role='button']"
            )
            # time.sleep(2)

        except:
            print("No buttons")
        else:
            for btn in buttons:

                if btn.text != "See more":
                    continue

                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center', behavior: 'instant'});",
                    btn,
                )
                # time.sleep(1)
                try:
                    btn.click()
                    # time.sleep(1)
                except:
                    print("couldnt click")
                    pass

    def apply_filter():
        filter_xpath = '//div[@class="x1i10hfl xjbqb8w x1ejq31n xd10rxx x1sy0etr x17r0tee x972fbf xcfux6l x1qhh985 xm0m39n x1ypdohk xe8uvvx xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x16tdsg8 x1hl2dhg xggy1nq x1o1ewxj x3x9cwd x1e5q0jg x13rtm0m x87ps6o x1lku1pv x1a2a7pz x9f619 x3nfvp2 xdt5ytf xl56j7k x1n2onr6 xh8yej3"][@role="button"][@aria-label="Filters"]'
        filters_btn = driver.find_element(By.XPATH, filter_xpath)
        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center', behavior: 'instant'});",
            filters_btn,
        )
        filters_btn.click()
        time.sleep(0.5)
        Xpath4ComboBox = "//div[@class='x1i10hfl xjqpnuy xa49m3k xqeqjp1 x2hbi6w xdl72j9 x2lah0s xe8uvvx x2lwn1j xeuugli x1hl2dhg xggy1nq x1t137rt x1q0g3np x87ps6o x1lku1pv x78zum5 x1a2a7pz x6s0dn4 xjyslct x1qhmfi1 xhk9q7s x1otrzb0 x1i1ezom x1o6z2jb x13fuv20 xu3j5b3 x1q0q8m5 x26u7qi x972fbf xcfux6l x1qhh985 xm0m39n x9f619 x1ypdohk x1qughib xdj266r x11i5rnm xat24cr x1mh8g0r x889kno xn6708d x1a8lsjc x1ye3gou x1n2onr6 x1yc453h x1ja2u2z'][@role='combobox'][@aria-expanded='false']"

        num_up_arrows = process_filter_date_offset()
        for combo_idx in range(3):
            combo = driver.find_elements(By.XPATH, Xpath4ComboBox)[-1]
            combo.click()
            for _ in range(num_up_arrows[combo_idx]):
                ActionChains(driver).send_keys(Keys.UP).perform()
            ActionChains(driver).send_keys(Keys.ENTER).perform()

        time.sleep(1)
        Xpath_done = "//div[@role='button'][@class='x1i10hfl xjbqb8w x1ejq31n xd10rxx x1sy0etr x17r0tee x972fbf xcfux6l x1qhh985 xm0m39n x1ypdohk xe8uvvx xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x16tdsg8 x1hl2dhg xggy1nq x1o1ewxj x3x9cwd x1e5q0jg x13rtm0m x87ps6o x1lku1pv x1a2a7pz x9f619 x3nfvp2 xdt5ytf xl56j7k x1n2onr6 xh8yej3'][@aria-label='Done']"
        driver.find_element(By.XPATH, Xpath_done).click()
        time.sleep(2)

    def perform_login():
        driver.get("https://www.facebook.com/login/")
        driver.find_element(By.ID, "email").send_keys(username)
        driver.find_element(By.ID, "pass").send_keys(passwrd)
        driver.find_element(By.ID, "loginbutton").click()
        driver.get(link)

    username = "user"
    passwrd = "password"

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-notifications")
    driver = webdriver.Chrome(options)

    # login
    if login:
        perform_login()
    else:
        driver.get(link)
        exit_btn = driver.find_element(By.CSS_SELECTOR, "[aria-label='Close']")
        exit_btn.click()

    if not filter_date is None:
        apply_filter()

    driver.maximize_window()
    timer = 1
    while 1:

        if timer % 5000 == 0:
            click_read_more()

        if timer % 10_000 == 0:

            with open(file_path, "w") as tf:
                tf.write(driver.page_source)
            timer = 0

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        timer += 1
        print(timer)


def last_date(file2html):

    with open(file2html, "r") as tf:
        content = tf.read()

    soup = BeautifulSoup(content, "html.parser")
    unfiltered_dates = soup.select('[id^="SvgT"]')
    dates = [
        elem
        for elem in unfiltered_dates
        if (elem.text != "Learn More") and (elem.text != "")
    ]

    start_date = dates[0].text.strip()
    end_date = dates[-1].text.strip()
    print(start_date + " -> " + end_date)
    return end_date


def html_extract(folder, json_path):

    htmls = os.listdir(folder)

    try:
        htmls.remove(".DS_Store")
    except:
        pass

    post_info = []

    for idx, elem in enumerate(htmls):

        path2html = folder + "/" + elem
        with open(path2html, "r", encoding="latin-1") as tf:
            content = tf.read()

        soup = BeautifulSoup(content, "html.parser")
        unfiltered_posts = soup.find_all(
            class_="x193iq5w xeuugli x13faqbe x1vvkbs xlh3980 xvmahel x1n0sxbx x1lliihq x1s928wv xhkezso x1gmr53x x1cpjm7i x1fgarty x1943h6x x4zkp8e x3x7a5m x6prxxf xvq8zen xo1l8bm xzsf02u x1yc453h"
        )
        posts = unfiltered_posts[3:]

        unfiltered_dates = soup.select('[id^="SvgT"]')
        dates = [
            elem
            for elem in unfiltered_dates
            if (elem.text != "Learn More") and (elem.text != "")
        ]

        for post, date in zip(posts, dates):
            text = post.text
            post_info.append({"Post": text, "Date": date.text})
        print(f"{idx+1}/{len(htmls)} htmls done")
    save_json(post_info, json_path)


def remove_duples(path2json):
    posts = load_json(path2json)

    seen_text = []
    unique = []
    duples = 0
    for elem in posts:
        if elem["Post"] in seen_text:
            duples += 1
        else:
            seen_text.append(elem["Post"])
            unique.append(elem)

    print(duples)
    if duples != 0:
        save_json(unique, path2json)


def clean_json(path2json, date_of_scrape=None):
    remove_duples(path2json)

    if date_of_scrape is None:
        date_of_scrape = datetime.now()

    # reformat dates
    posts = load_json(path2json)
    for idx, elem in enumerate(posts):
        date = elem["Date"]

        if (date.split()[0] == "Call") or (date.split()[0] == "Learn"):
            posts[idx]["Date"] = posts[idx - 1]["Date"]
            continue
        else:
            posts[idx]["Date"] = reformat_date(date, date_of_scrape)
    save_json(posts, path2json)


def reformat_date(date2reformat, date_of_scrape=None):

    if date_of_scrape is None:
        datetime.now()

    month_dict = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12",
    }

    old_date = date2reformat
    splitted = old_date.replace(",", "").split()

    if not splitted[-1] in ["2023", "2022", "2021", "2020", "2019", "2018", "2017"]:

        if splitted[0].lower() in month_dict.keys():
            year = "2024"
            month = month_dict[splitted[0].lower()]
            day = splitted[1]
            if len(day) < 2:
                day = "0" + day

            final = f"{year}-{month}-{day}"
        elif (splitted[1] == "day") or (splitted[1] == "days"):
            if splitted[1] == "day":
                days_offset = 1
            else:
                days_offset = int(splitted[0])

            final = str(date_of_scrape - timedelta(days=days_offset))[:10]
        else:
            final = str(date_of_scrape)[:10]

    else:
        year = splitted[2]
        month = month_dict[splitted[0].lower()]
        day = splitted[1]
        if len(day) < 2:
            day = "0" + day
        final = f"{year}-{month}-{day}"

    return final


def extract_handler(link, folder_path, login=True, start_date=None):

    if start_date is None:
        start_date = str(datetime.now())[:10]

    filter_date = start_date
    temp_folder_path = "Assets/HTMLS/Temp-htmls/temp.html"
    while 1:
        try:
            site_extract(link, temp_folder_path, login=login, filter_date=filter_date)
        except:
            end_date = last_date(temp_folder_path)
            filter_date = reformat_date(end_date)
            move_file(temp_folder_path, folder_path)


def main():

    profile1 = "https://www.facebook.com/ReactionUnitSouthAfrica"
    profile2 = "https://www.facebook.com/crimewatch202/"
    profile3 = "https://www.facebook.com/SAPoliceService/"

    profile2scrape = profile3
    extract_handler(
        profile2scrape, "Assets/HTMLS/SAPS", login=True, start_date="2018-04-07"
    )

    # site_extract(profile2scrape, "Assets/HTMLS/4Testing", login=True,
    #              filter_date=None)

    html0 = "Assets/HTMLS/4Testing"
    html1 = "Assets/HTMLS/Reaction-Unit"
    html2 = "Assets/HTMLS/Crime-Watch202"
    html3 = "Assets/HTMLS/SAPS"
    html2use = html3
    # html_extract(html2use, "Assets/JSONS/SAPS.json")

    json1 = "Assets/JSONS/RU.json"
    json2 = "Assets/JSONS/CW202.json"
    json3 = "Assets/JSONS/SAPS.json"
    json2use = json3

    # clean_json(json2use)
    print(len(load_json(json2use)))


if __name__ == "__main__":
    main()
