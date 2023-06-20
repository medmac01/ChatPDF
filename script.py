import requests
from bs4 import BeautifulSoup
import re
import os, json

PATH = "/Users/medmachrouh/Desktop/PFA/Wiki-Scraping/sites"
OUT_PATH = "/Users/medmachrouh/Desktop/PFA/Wiki-Scraping/data.json"

def scrape(url):
	print(f"Getting {url}")
	response = requests.get(url.replace('\n',''))
	html_content = response.text

	soup = BeautifulSoup(html_content, "html.parser")

	# Find the main content div using its id or class
	content_div = soup.find("div", id="mw-content-text")

	# Find all the paragraphs within the main content div
	paragraphs = content_div.find_all("p")

	out = []
	# Extract the text from the paragraphs
	for paragraph in paragraphs:
	    text = paragraph.get_text()
	    # Process and store the extracted text as desired
	    element = {'source':url,'text':text}
	    out.append(element)

	return out

if __name__ == '__main__':

	data = []
	with open(PATH) as f:
		for i in range(145):
			url = f.readline()
			data.extend(scrape(url))


	with open(OUT_PATH,'w') as out:
		json.dump(data,out,indent=2)
		print('Saved')

