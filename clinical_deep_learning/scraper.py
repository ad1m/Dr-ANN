__author__ = 'Adamlieberman'
from bs4 import BeautifulSoup
import requests


'''
Scrape ICD-9 descriptions for a given ICD-9 code
'''

def scrape_icd9(code):
    link = "https://www.findacode.com/code.php?set=ICD9&c="+str(code)
    html = requests.get(link).text
    soup = BeautifulSoup(html,"html.parser")
    blockquote = soup.find("div",{"class":"sectionbody"})
    ls = list(blockquote)
    count = 1
    for i in ls[1]:
        if count == 3:
            description = i.replace("-","").lstrip()
            break
        count = count + 1
    return description
if __name__=="__main__":
    code = 250.01
    description = scrape_icd9(code)
    print description
