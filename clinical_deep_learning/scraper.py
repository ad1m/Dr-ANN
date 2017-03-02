__author__ = 'Adamlieberman'
from bs4 import BeautifulSoup
import requests


'''
Scrape ICD-9 descriptions for a given ICD-9 code
'''

def scrape_icd9(codes):
    all_descriptions = []
    for c in codes:
        link = "https://www.findacode.com/code.php?set=ICD9&c="+str(c)
        html = requests.get(link).text
        soup = BeautifulSoup(html,"html.parser")
        blockquote = soup.find("div",{"class":"sectionbody"})
        ls = list(blockquote)
        count = 1
        for i in ls[1]:
            if count == 3:
                description = i.replace("-","").lstrip()
                all_descriptions.append(description)
                break
            count = count + 1
    return all_descriptions
if __name__=="__main__":
    codes = [250.01,250.02,250.03]
    description = scrape_icd9(codes)
    print description
