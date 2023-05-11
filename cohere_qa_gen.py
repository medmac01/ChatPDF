import json
import re
import cohere
import time


co = cohere.Client('U0pNRE1uLERaCJJFPOQr5MKYLZ1rl5UkmcL5r1VA')

# Load existing JSON data from the file
with open("index_2.json", "r") as json_file:
    passages_data = json.load(json_file)

# Initialize the list to store the generated QA pairs
generated_qa_pairs = []

counter = 0
# Generate QA pairs for each passage
for entry in passages_data:
    try:
        # if counter % 4 == 0 and counter != 0:
        #     print("Sleeping for 60 seconds")
        #     time.sleep(60)
        #     continue

        context = entry["content"]

        response = co.generate(
            model='command',
            prompt=f'Generer cinq question (pas qcm) en relation au context du passage suivant, chaque question doit avoir sa réponse sous la forme suivante:\n\n{{ question: \"\",\nreponse: \"\"}}\n\nPassage: {context}\n\nQuestion:',
            max_tokens=300,
            temperature=0.7,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
        print(f'Attempt {counter} before sleeping')
        counter = counter + 1

        # Extract the generated question and answer from the response using regex
        generated_text = response.generations[0].text.strip()

        # Split the generated text into question and answer parts
        parts = re.split(r"Réponse:|Answer:|reponse:|answer:|Reponse:|réponse:", generated_text)
        if len(parts) == 2:
            generated_question = parts[0].strip()
            generated_answer = parts[1].strip()
        else:
            generated_question = generated_text
            generated_answer = ""

        # Create a new QA pair
        qa_pair = {
            "context": context,
            "question": generated_question,
            "answer": generated_answer
        }

        # Append the QA pair to the list
        generated_qa_pairs.append(qa_pair)

    # Append the generated QA pairs to the existing JSON data
        with open("qa_dataset.json", "r") as json_file:
            existing_data = json.load(json_file)

        existing_data.extend(generated_qa_pairs)

        # Write the updated data back to the JSON file
        with open("qa_dataset.json", "w") as json_file:
            json.dump(existing_data, json_file, indent=4)

        print("QA pairs generated and added to the JSON dataset.")
    except cohere.error.CohereAPIError:
        print('API tired')  
        time.sleep(65)

        continue    
