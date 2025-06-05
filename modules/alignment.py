from transformers import pipeline
import re

# Create the NER pipeline
ner_pipe = pipeline('ner', model="Davlan/bert-base-multilingual-cased-ner-hrl")

# Define regex patterns for money and phone numbers
money_pattern = r'\$\d+(?:\.\d+)?|\d+\s?(?:dollars|rupees|INR|USD|EUR|€|£|₹)'
phone_pattern = r'\+?\d{1,4}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'

# Example text with money and phone numbers
text = " where in the city of patiala with pahuldeep singh"

# Run NER for standard entities
ner_results = ner_pipe(text)

# Find money mentions
money_matches = []
for match in re.finditer(money_pattern, text):
    money_matches.append({
        'entity': 'MONEY',
        'word': match.group(),
    })

# Find phone numbers
phone_matches = []
for match in re.finditer(phone_pattern, text):
    phone_matches.append({
        'entity': 'PHONE',
        'word': match.group(),
    })

# # Combine all results
# all_results = ner_results + money_matches + phone_matches

# # Sort by position in text
# all_results.sort(key=lambda x: x['start'] if 'start' in x else x.get('start_pos', 0))

# print(all_results)

