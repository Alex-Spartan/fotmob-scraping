from selenium import webdriver
from bs4 import BeautifulSoup
from modules import get_stats as gs
import time

def scrapeData(url1):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)


    def extract_codes_from_second_link(url):
        driver.get(url)
        time.sleep(4)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Scrape all a tags from the div with a specific class
        div = soup.find('div', class_='css-1mvr87o-FixturesCardCSS ep1hw9x0')  # Replace 'specific-class-name' with the actual class name
        if div:
            codes = [a['href'].split('#')[-1] for a in div.find_all('a', href=True) if '#' in a['href']]
            return codes[-20:]
        else:
            print("Div2 not found")
            return []

    codes = extract_codes_from_second_link(url1)


    driver.quit()

    team = url1.split('/')[-1].split('?')[0].capitalize()
    gs.get_stats(codes, team)
