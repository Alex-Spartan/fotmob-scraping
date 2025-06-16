from selenium import webdriver
from bs4 import BeautifulSoup
from modules import get_stats as gs
import time

def scrapeData(url1):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    team = None

    def extract_codes_from_second_link(url):
        nonlocal team  # Use the team variable from the enclosing scope
        driver.get(url)
        time.sleep(4)

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        span = soup.find('span', class_='css-2irgse-TextOnDesktop e1i7jfg81')
        if span:
            team = span.text
        else:
            print("Span not found")
        
        # Replace div class with the correct one
        div = soup.find('div', class_='css-1mvr87o-FixturesCardCSS e16yw8af0')
        if div:
            codes = [a['href'].split('#')[-1] for a in div.find_all('a', href=True) if '#' in a['href']]
            return codes
        else:
            print("Div2 not found")
            return []

    codes = extract_codes_from_second_link(url1)

    driver.quit()

    gs.get_stats(codes, team)
