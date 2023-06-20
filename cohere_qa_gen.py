import json
import re
import cohere, openai
import time
import requests


# Load existing JSON data from the file
with open("data.json", "r") as json_file:
    passages_data = json.load(json_file)

# Initialize the list to store the generated QA pairs
checkpoint = passages_data[4350:]
counter = 0
print(len(passages_data))
# Generate QA pairs for each passage
for entry in checkpoint:
    try:
        generated_qa_pairs = []
        context = entry["text"]
        print(f'Trying this passage {context[:100]}')
        # response = openai.Completion.create(
        #       model="text-davinci-003",
        #       prompt="Ce passages est tiré des livres dont leur thèmes est l'histoire du Maroc. Generer une question historique (pas qcm) en relation au passage suivant, chaque question doit avoir sa réponse dans le passage et doit avoir sa réponse sous la forme suivante:\nQuestion:\nReponse:\n\nPassage :{passage}\n\nQuestion:",
        #       temperature=0.7,
        #       max_tokens=256,
        #       top_p=1,
        #       frequency_penalty=0,
        #       presence_penalty=0
        #     )

        # response = co.generate(
        #     model='command',
        #     prompt=f'Generate a question related to the following passage, each question should have also its answer.The answer should be elaborated in a sentence or a short paragraph, but not a single word.\n\nPassage: {context}\n\nQuestion:',
        #     max_tokens=300,
        #     temperature=0.7,
        #     k=0,
        #     stop_sequences=[],
        #     return_likelihoods='NONE'
        # )



        API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
    
        output = query({
            "inputs": f"Generate a question related to the following passage, each question should have also its answer.The answer should be elaborated in a sentence or a short paragraph, but not a single word.\n\nPassage: {context}\n\nQuestion:",
            "parameters": {"max_new_tokens":600}
        })
        # print(output)

        print(f'Attempt {counter} before sleeping')
        counter = counter + 1

        # Extract the generated question and answer from the response using regex
        # generated_text = response.generations[0].text.strip()
        output = output[0]['generated_text']

        # Split the generated text into question and answer parts
        # parts = re.split(r"Réponse:|Answer:|reponse:|answer:|Reponse:|réponse:|ANSWER:", generated_text)
        # if len(parts) == 2:
        #     generated_question = parts[0].strip()
        #     generated_answer = parts[1].strip()
        # else:
        #     generated_question = generated_text
        #     generated_answer = ""

        generated_question = output[output.find("Question:"):output.find("Answer:")-1]
        generated_answer = output[output.find("Answer:"):]
        # Create a new QA pair
        qa_pair = {
            "context": context,
            "question": generated_question,
            "answer": generated_answer
        }

        # Check if the QA pair already exists in the JSON file
        with open("qa_dataset.json", "r") as json_file:
            existing_data = json.load(json_file)

        # existing_questions = [qa["question"] for qa in existing_data]
        # if generated_question in existing_questions:
        #     print("QA pair already exists. Skipping...")
        #     continue

        # Append the QA pair to the list
        generated_qa_pairs.append(qa_pair)

        # Append the generated QA pairs to the existing JSON data
        existing_data.extend(generated_qa_pairs)

        # Write the updated data back to the JSON file
        with open("qa_dataset.json", "w") as json_file:
            json.dump(existing_data, json_file, indent=4)

        print("QA pairs generated and added to the JSON dataset.")
    except cohere.error.CohereAPIError:
        print('API tired')
        time.sleep(65)
        continue
