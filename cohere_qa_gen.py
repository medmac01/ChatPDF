import json
import re
import cohere

# Your existing code
co = cohere.Client('U0pNRE1uLERaCJJFPOQr5MKYLZ1rl5UkmcL5r1VA')
response = co.generate(
  model='command',
  prompt=f'Generer cinq question (pas qcm) en relation au context du passage suivant, chaque question doit avoir sa réponse sous la forme suivante:\n\n{ question: \"\",\nreponse: \"\"}\n\nPassage :{passage}\n\nQuestion:',
  max_tokens=300,
  temperature=0.7,
  k=0,
  stop_sequences=[],
  return_likelihoods='NONE')

# Extract the generated question and answer from the response using regex
generated_text = response.generations[0].text.strip()
parts = re.split(r"Réponse:|Answer:|reponse:|answer:", generated_text)
if len(parts) == 2:
    generated_question = parts[0].strip()
    generated_answer = parts[1].strip()
else:
    generated_question = generated_text
    generated_answer = ""

# Load existing JSON data from the file
with open("qa_dataset.json", "r") as json_file:
    existing_data = json.load(json_file)

# Append the generated question and answer to the existing JSON data
new_question = {
    "question": generated_question,
    "reponse": generated_answer
}
existing_data.append(new_question)

# Write the updated data back to the JSON file
with open("qa_dataset.json", "w") as json_file:
    json.dump(existing_data, json_file, indent=4)

print("Question added to the JSON dataset.")
