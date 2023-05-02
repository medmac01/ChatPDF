import os
import json
import PyPDF2
import textwrap
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('sentence-transformers/gtr-t5-large')
def gtr_embedding(content):
    # response = openai.Embedding.create(input=content,engine=engine)
    # vector = response['data'][0]['embedding']  # this is a normal list
    vector = model.encode(content)
    return vector


"""
This code iterates over a list of pdfs, extracts text and divide it into chunks. Then it saves the chunks into a json file.
"""
pdf_dir = os.path.join(os.getcwd(), 'Dataset')
json_file = os.getcwd()+'/index_2.json'

pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

output = []

for pdf_file in pdf_files:
    with open(os.path.join(pdf_dir, pdf_file), "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        print(f'Reading pdf: {f}')
        num_pages = len(pdf_reader.pages)
        print(f'This pdf has {num_pages} pages')
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            content = page.extract_text()
            chunks = textwrap.wrap(content, 1000)
            
            for chunk in chunks:
                embed = gtr_embedding(chunk)
                output.append({
                    "filename": pdf_file,
                    "page_number": page_num+1,
                    "content": chunk,
                    "embedding" : np.array(embed).tolist()
                })

with open(json_file, "w") as f:
    json.dump(output, f)