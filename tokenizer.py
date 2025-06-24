from transformers import AutoTokenizer
import csv

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
frase = input("Ingresa la frase a tokenizar: ")
tokens = tokenizer(frase, return_tensors=None, add_special_tokens=True)

with open(r"D:\PROYECTO-TOPICOS-IA\tokens.csv", "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(tokens['input_ids'])

print("Tokens guardados en 'D:\\PROYECTO-TOPICOS-IA\\tokens.csv'")