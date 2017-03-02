__author__ = 'Adamlieberman'
from flask import Flask, render_template, request
from scraper import scrape_icd9
app = Flask(__name__)

@app.route('/')
def input_page():
    return render_template('input.html')

@app.route('/',methods=['POST'])
def input_page_post():
    clinical_note = request.form['note']
    process_text = clinical_note.lower()
    #Pass the processed text into the deep learning model
    #return prediction of top 3 ICD-9 codes
    #descriptions
    #pass this to the results.html file along with the scraped data

if __name__ == "__main__":
    app.run(debug=True)