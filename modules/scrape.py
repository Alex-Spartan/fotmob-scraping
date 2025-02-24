from selenium import webdriver
from bs4 import BeautifulSoup
from modules import get_stats as gs
import time
def scrapeData(url="https://www.fotmob.com/en-GB/teams/8543/overview/lazio"):

    html_url = url


    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    def extract_codes_selenium(url):
        driver.get(url)
        time.sleep(4)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        section = soup.find('section', class_='css-1o7ul5e-TeamFormContainer e1pube2c1')
        if section:
            print("Section found!")
            codes = [a['href'].split('#')[-1] for a in section.find_all('a', href=True) if '#' in a['href']]
            return codes
        else:
            print("Section not found")
            return []

        driver.quit()

    team = html_url.split('/')[-1].capitalize()

    codes = extract_codes_selenium(html_url)

    gs.get_stats(codes, team)