import ast
import json
from FlagEmbedding import BGEM3FlagModel
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# Read the file content
with open('message.txt', 'r', encoding='utf-8') as file:
    file_content = file.read()

# Parse the content as a Python list
items = ast.literal_eval(file_content)

# Initialize the model
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) # Use FP16 for faster computation

# Generate embeddings for each item
embeddings = model.encode(items, batch_size=12, max_length=8192)['dense_vecs']

# Prepare data for JSON
data = {item: embedding.tolist() for item, embedding in zip(items, embeddings)}

# Write data to JSON file
with open('embeddings.json', 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=4, ensure_ascii=False)

print("Embeddings have been saved to 'embeddings.json'")