from transformers import AutoTokenizer
import csv
import os

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Pedir la primera frase
frase1 = input("Ingresa la primera frase a tokenizar: ")
tokens1 = tokenizer(frase1, return_tensors=None, add_special_tokens=True)

# Pedir la segunda frase
frase2 = input("Ingresa la segunda frase a tokenizar: ")
tokens2 = tokenizer(frase2, return_tensors=None, add_special_tokens=True)

# Guardar tokens de la primera frase en tokens1.csv
output_path1 = os.path.join(os.getcwd(), "tokens1.csv")
with open(output_path1, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(tokens1['input_ids'])

print(f"Tokens de la primera frase guardados en '{output_path1}'")

# Guardar tokens de la segunda frase en tokens2.csv
output_path2 = os.path.join(os.getcwd(), "tokens2.csv")
with open(output_path2, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(tokens2['input_ids'])

print(f"Tokens de la segunda frase guardados en '{output_path2}'")