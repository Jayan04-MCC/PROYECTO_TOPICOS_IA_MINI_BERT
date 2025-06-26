from transformers import AutoTokenizer
import csv
import os

# Test phrase for Day 3
frase = "Hello world, this is a test."

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokens = tokenizer(frase, return_tensors=None, add_special_tokens=True)

output_path = os.path.join(os.getcwd(), "tokens.csv")
with open(output_path, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(tokens['input_ids'])

print(f"Test tokens for '{frase}': {tokens['input_ids']}")
print(f"Tokens guardados en '{output_path}'")