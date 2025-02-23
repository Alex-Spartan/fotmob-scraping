from selenium import webdriver
from bs4 import BeautifulSoup
from modules import get_top_stats as gts
import time


match_id = "4535454"  # Example Match ID

html_url = "https://www.fotmob.com/en-GB/teams/8543/overview/lazio"


# Set up Selenium
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in headless mode (no GUI)
driver = webdriver.Chrome(options=options)

def extract_codes_selenium(url):
    driver.get(url)
    time.sleep(4)  # Wait for JavaScript to load content

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Extract section
    section = soup.find('section', class_='css-1o7ul5e-TeamFormContainer e1pube2c1')
    if section:
        print("Section found!")
        codes = [a['href'].split('#')[-1] for a in section.find_all('a', href=True) if '#' in a['href']]
        return codes
    else:
        print("Section not found")
        return []

    driver.quit()

html_url = "https://www.fotmob.com/en-GB/teams/8636/overview/inter"  # Change this
codes = extract_codes_selenium(html_url)

print(codes)
gts.get_top_stats(codes)