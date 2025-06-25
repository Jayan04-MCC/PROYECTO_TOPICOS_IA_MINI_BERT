from transformers import AutoTokenizer
import csv
import os

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
frase = input("Ingresa la frase a tokenizar: ")
tokens = tokenizer(frase, return_tensors=None, add_special_tokens=True)

import os

output_path = os.path.join(os.getcwd(), "tokens.csv")
with open(output_path, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(tokens['input_ids'])

print(f"Tokens guardados en '{output_path}'")